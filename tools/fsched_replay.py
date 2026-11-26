"""Numerical inputs for fsched_sim. No tokenizer, network, GPU or model required."""
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import median


BENCH = Path(__file__).resolve().parents[1] / "bench"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def priority(headers, fractional=False):
    values = [str(v) for k, v in headers.items() if k.lower() == "x-sglang-priority"]
    if not values:
        return 1
    if fractional and len(values) == 1:
        w = float(values[0])
        if not math.isfinite(w) or w <= 0:
            raise ValueError("priority must be finite and positive")
        return w
    if len(values) != 1 or not values[0].isascii() or not values[0].isdecimal():
        raise ValueError("X-SGLang-Priority must be a positive integer")
    n = int(values[0])
    if n < 1:
        raise ValueError("X-SGLang-Priority must be positive")
    return n


@dataclass(frozen=True)
class Request:
    name: str
    arrival: float
    prompt: int
    requested: int
    eos: int  # hidden from policies; observed/synthetic completion length
    weight: float = 1
    # Named, immutable radix edges, ordered root to leaf; private tail implicit.
    prefix: tuple = ()
    cached: int = 0  # known pre-existing hit, not earned by an active request
    after_previous: float | None = None  # closed-loop client think time

    def __post_init__(self):
        if (not math.isfinite(self.arrival) or self.arrival < 0
                or not math.isfinite(self.weight) or self.weight <= 0
                or not 0 <= self.eos <= self.requested or self.prompt < 0
                or bool(self.eos) != bool(self.requested)
                or any(n <= 0 for _, n in self.prefix)
                or not 0 <= self.cached <= self.prompt
                or sum(n for _, n in self.prefix) + self.cached > self.prompt
                or (self.after_previous is not None and self.after_previous < 0)):
            raise ValueError(f"invalid request: {self.name}")

    def edges(self):
        path = ()
        for label, length in self.prefix:
            path += (label,)
            yield ("shared", path), length
        tail = self.prompt - self.cached - sum(n for _, n in self.prefix)
        if tail:
            yield ("private", self.name), tail


def interp(points, x):
    if x <= points[0][0]:
        return points[0][1]
    for (a, u), (b, v) in zip(points, points[1:]):
        if x <= b:
            return u + (v - u) * (x - a) / (b - a)
    return points[-1][1]


@dataclass(frozen=True)
class Cost:
    prefill_rates: tuple
    decode_rates: tuple  # per stream
    mixed_short_rate: float
    accept: float

    def seconds(self, prefill, reservations, mean_accept):
        # A verify costs time even when all its proposals are rejected. Scale
        # by width (rows), not the fortuitous token count revealed afterwards.
        seconds = prefill / interp(self.prefill_rates, prefill)
        if reservations:
            rate = (self.mixed_short_rate if prefill else
                    interp(self.decode_rates, len(reservations)))
            seconds += mean_accept / rate
        return seconds


def recorded(bench=BENCH):
    timing = rows(bench / "2026-11-25-prefill-mixed.jsonl")
    fixture = rows(bench / "2026-11-26-fsched-arrivals.jsonl")
    quality = rows(bench / "2026-11-24-qsa-native-q8-quality.jsonl")
    costs, traces = {}, []
    for arm in sorted({r["arm"] for r in timing}):
        data = [dict(r["data"], pass_number=r["pass_number"]) for r in timing
                if r["arm"] == arm and r["kind"] == "timing"]
        gate, = [r["data"] for r in timing if r["arm"] == arm and r["kind"] == "gates"]
        chunk = gate["candidate"]["chunked_prefill_size"]
        train = [d for d in data if d["pass_number"] < 3]
        singles = [d for d in train if d["scenario"] == "single"]
        conc = [d for d in train if d["scenario"] == "concurrent"]
        mixed = [d for d in train if d["scenario"] == "mixed"]
        # Measured long-prompt throughput at the two chunk sizes. Avoid
        # accidentally fitting a 32k rate to the 4k arm's *short* prompts.
        rates = ((1, median(d["prefill_tps"] for d in singles if d["target_tokens"] == 2048)),
                 (chunk, median(d["prefill_tps"] for d in singles if d["target_tokens"] == 32000)))
        dr = [(1, median(gate["candidate_overlap"]["decode_tps"]))]
        dr += [(k, median(d["per_stream_decode_tps"] for d in conc if d["streams"] == k))
               for k in sorted({d["streams"] for d in conc})]
        costs[chunk] = Cost(rates, tuple(dr), median(x for d in mixed for x in d["short_decode_tps"]),
                            median(gate["candidate_overlap"]["accepted_per_verify"]))
        for trace in (r for r in fixture if r["arm"] == arm):
            name = trace["name"]
            obs = trace["requests"]
            if name == "quality":
                if [x[1] for x in obs] != [q["input_tokens"] for q in quality]:
                    raise ValueError("quality fixture does not match the gauntlet")
                specs = [(q["name"], 64 if q["kind"] == "task" else 0) for q in quality]
            else:
                ps, j, scenario = name.split("-")
                d = [d for d in data if d["pass_number"] == int(ps[1:])][int(j)-1]
                assert d["scenario"] == scenario
                if scenario == "single":
                    assert len(obs) == 1 and obs[0][1:3] == [d["prompt"], d["completion"]]
                    specs = [("single", 128)]
                elif scenario == "concurrent":
                    assert len(obs) == d["streams"] and sum(x[2] for x in obs) == d["completion_total"]
                    specs = [(f"s{i}", 200) for i in range(len(obs))]
                else:
                    assert len(obs) == d["long_count"] + d["short_streams"]
                    assert sorted((x[1], x[2]) for x in obs if x[1] == d["long_prompt"]) == sorted(
                        (x["prompt"], x["completion"]) for x in d["longs"])
                    specs = [(f"r{i}", 64 if x[1] == d["long_prompt"] else 200)
                             for i, x in enumerate(obs)]
            # Cache hits do not identify a radix path. Model only the first
            # shared 64-token chat page in bursts with measured hits. Do not
            # infer deeper shared paths from equal lengths or cached counts.
            shared = name != "quality" and any(x[3] >= 64 for x in obs)
            requests = [Request(key, arrival, prompt, limit, out,
                                prefix=(("chat", 64),) if shared else (),
                                cached=cached if name == "quality" else 0,
                                after_previous=trace["think_s"][i] if i and name == "quality" else None)
                        for i, ((key, limit), (arrival, prompt, out, cached))
                        in enumerate(zip(specs, obs, strict=True))]
            traces.append((chunk, name, requests))
    # Choose prefill rate by *dispatched shape*, not by the source of the
    # arrival trace. In particular, a 2k batch is not a slow 4k batch.
    rates = (costs[2048].prefill_rates[0], costs[2048].prefill_rates[1], costs[4096].prefill_rates[1])
    for chunk, c in costs.items():
        costs[chunk] = Cost(rates, c.decode_rates, c.mixed_short_rate, c.accept)
    return costs, traces


def synthetic(seed, count, arrival_rate, priorities, fractional=False):
    rng = random.Random(seed)
    now = 0.
    requests = []
    for i in range(count):
        if i and arrival_rate:
            now += rng.expovariate(arrival_rate)
        prefix = (("chat", 64), (f"topic{i % 3}", 256))
        prompt = 320 + rng.choice((32, 512, 2048, 8192))
        limit = rng.choice((64, 200))
        out = rng.randint(1, limit) if i % 3 == 0 else limit
        w = priority({"X-SGLang-Priority": priorities[i % len(priorities)]}, fractional)
        requests.append(Request(f"s{i}", now, prompt, limit, out, w, prefix))
    return requests
