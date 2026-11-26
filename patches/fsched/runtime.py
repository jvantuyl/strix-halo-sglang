"""Time-share scheduler for the pinned gfx1151 Qwen NEXTN configuration.

This module never mutates KV, a ScheduleBatch, or another policy's queue.
Durations are seconds. Only the worker's stream events constitute measurements.
"""
import hashlib
import logging
import math
import os
from collections import Counter, deque
from dataclasses import dataclass


CAP, POSITIONS, SECONDS = 12, 2048, 2.7
GEN, BAND = 1 / 3, .5
log = logging.getLogger(__name__)


def header_weight(values):
    if not values:
        return 1.
    if len(values) == 1:
        try:
            w = float(values[0])
            if math.isfinite(w) and w > 0:
                return w
        except ValueError:
            pass
    raise ValueError("X-SGLang-Priority must be one finite, positive weight")


def shares(weights):
    # Scale first: two individually finite header weights can sum to infinity.
    scale = max(weights.values(), default=1.)
    scaled = {k: v / scale for k, v in weights.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


def cost(p, g):
    # Recorded pass 1/2 calibration, not measured per-request GPU time.
    pt = p / (671.58245 + (758.48711 - 671.58245) * max(0, p-1) / 2047)
    if not g:
        return pt, 0.
    if p:
        return pt, 1 / 6.0068035
    grid = ((1, 25.98013), (4, 12.45013), (8, 8.563835), (12, 6.93829))
    for (lo, a), (hi, b) in zip(grid, grid[1:]):
        if g <= hi:
            return 0., 3.1219512 / (a + (b-a) * (g-lo) / (hi-lo))
    raise ValueError("fsched calibration ends at 12 generation rows")


def position_limit(g):
    lo, hi = 0, POSITIONS - g
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(cost(mid, g)) <= SECONDS:
            lo = mid
        else:
            hi = mid - 1
    return lo


@dataclass(eq=False)
class Account:
    weight: float
    lag: float = 0.
    since: float = 0.
    skips: int = 0
    last_paid: int = -1


def account(req):
    if not hasattr(req, "_fsched_account"):
        req._fsched_account = Account(getattr(req, "fsched_weight", 1.))
    return req._fsched_account


def simple(req):
    return not (req.return_logprob or req.beam_group is not None or
                req.input_embeds is not None or req.session_id is not None or
                req.multimodal_inputs is not None or req.positional_embed_overrides is not None)


def front(req, start, grid):
    """Identify an exact pending cache page; don't equate two cache-hit lengths."""
    ids = req.origin_input_ids
    end = (start // grid + 1) * grid
    if (req.output_ids or end >= len(ids) or
            getattr(req, "skip_radix_cache_insert", False)):
        return req  # private edge
    saved = getattr(req, "_fsched_front", None)
    if not saved or saved[0] != (start, end):
        # TokenizedGenerateReqInput uses array('q'); only hash a frontier once.
        key = hashlib.blake2b(memoryview(ids)[:end], digest_size=24).digest()
        req._fsched_front = (start, end), key
    return req.extra_key, req.cache_salt, start, req._fsched_front[1]


class Edge:
    def __init__(self, owner, members, start, grid):
        self.owner = owner
        self.start = start
        self.work = max(1, len(owner.origin_input_ids) + len(owner.output_ids) - start)
        # Bound pair comparisons to one possible forward. Never credit a
        # private suffix, an already-cached token, or the last prompt token.
        self.ends = {}
        ids = owner.origin_input_ids
        for r in members:
            if r is owner:
                continue
            other = r.origin_input_ids
            first = (start // grid + 1) * grid
            hi = min(len(ids)-1, len(other)-1, start+POSITIONS) // grid
            lo = first // grid
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if ids[first:mid*grid] == other[first:mid*grid]:
                    lo = mid
                else:
                    hi = mid-1
            self.ends[account(r)] = lo*grid - start
        self.leader = account(owner)
        self.members = (self.leader, *self.ends)

    def split(self, begin, end):
        bill = Counter()
        for stop in sorted({v for v in self.ends.values() if begin < v < end} | {end}):
            members = [self.leader, *(a for a, n in self.ends.items() if n >= stop)]
            for a, f in shares({a: a.weight for a in members}).items():
                bill[a] += (stop-begin)*f
            begin = stop
        return bill


class Round:
    def __init__(self, waiting, chunk, gen, slots, page, grid):
        self.gen = [account(r) for r in gen if not r.finished()]
        self.page = page
        self.slots = min(slots, CAP - len(self.gen))
        self.chunk = chunk
        # Admission window is FIFO. Beyond it only followers of eligible
        # physical work earn credit, without consuming an additional row.
        eligible = ([chunk] if chunk else []) + waiting[:max(0, self.slots-bool(chunk))]
        groups = {}
        starts = {}
        for r in eligible:
            start = len(r.prefix_indices) if r is chunk else r.num_matched_prefix_tokens
            starts[r] = start
            groups.setdefault(front(r, start, grid), []).append(r)
        for r in waiting[max(0, self.slots-bool(chunk)):]:
            key = front(r, r.num_matched_prefix_tokens, grid)
            if key in groups:
                groups[key].append(r)
        self.edges = {}
        for members in groups.values():
            owner = chunk if chunk in members else max(
                (r for r in members if r in starts),
                key=lambda r: (account(r).skips >= CAP, account(r).lag, account(r).weight))
            self.edges[owner] = Edge(owner, members, starts[owner], grid)
        self.both = bool(self.edges and self.gen)
        self.rate = {}
        for accounts, fraction in (([a for e in self.edges.values() for a in e.members], 1-GEN),
                                   (self.gen, GEN)):
            self.rate.update({a: f * (fraction if self.both else 1.)
                              for a, f in shares({a: a.weight for a in accounts}).items()})
        self.guard = max((a for a in self.rate if a.since >= .5 or a.skips >= CAP),
                         key=lambda a: (a.skips, a.since), default=None)

    def pack(self, g):
        """Lag-ranked quanta, constrained to the pin's one unfinished row."""
        left = position_limit(g)
        alloc = Counter()
        paid = Counter()
        partial = self.chunk
        while left:
            best = None
            for r, e in self.edges.items():
                old = alloc[r]
                if not old and len(alloc) >= max(1, self.slots):
                    continue
                # Other rows must fit completely while a partial is open.
                n = min(e.work-old, self.page if partial in (None, r) else e.work)
                size = min(e.work, ((old+n+self.page-1)//self.page)*self.page)
                n = size-old
                used = (size+self.page-1)//self.page*self.page - (old+self.page-1)//self.page*self.page
                if n <= 0 or used > left or (size < e.work and partial not in (None, r)):
                    continue
                parts = e.split(old, size)
                dt = sum(cost(sum(alloc.values())+n, g))
                rank = (r is self.chunk and not old, self.guard in parts and not paid[self.guard],
                        sum(a.lag + self.rate[a]*dt - paid[a] for a in parts), -e.work)
                if best is None or rank > best[0]:
                    best = rank, r, size, used, parts
            if best is None:
                break
            _, r, size, used, parts = best
            alloc[r] = size
            left -= used
            for a, v in parts.items():
                paid[a] += v / 758.48711
            if size < self.edges[r].work:
                partial = r
            elif partial is r:
                partial = None
        return alloc

    def choose(self, debt):
        plans = {}
        for name, pre, gen in (("prefill", True, False), ("generation", False, True),
                               ("mixed", True, True)):
            if (pre and not self.edges) or (gen and not self.gen):
                continue
            g = len(self.gen) if gen else 0
            alloc = self.pack(g) if pre else {}
            if pre and not alloc:
                continue
            served = set(self.gen if gen else ())
            for r, n in alloc.items():
                served.update(self.edges[r].split(0, n))
            pt, gt = cost(sum(alloc.values()), g)
            plans[name] = (alloc, debt + GEN*(pt+gt) - gt, self.guard in served)
        if any(v[2] for v in plans.values()):
            plans = {k:v for k,v in plans.items() if v[2]}
        band = BAND * (1-GEN) * cost(0, len(self.gen))[1] if self.gen else 0.
        if "mixed" in plans and max(abs(debt), abs(plans["mixed"][1])) <= band:
            name = "mixed"
        else:
            name = min(plans, key=lambda k: (abs(plans[k][1]), k != "mixed"))
        alloc = plans[name][0]
        # Native adder admits a new partial last; the existing chunk always
        # uses its separate continuation path.
        queue = sorted((r for r in alloc if r is not self.chunk),
                       key=lambda r: alloc[r] < self.edges[r].work)
        chunk_limit = alloc.get(self.chunk)
        return name, queue, chunk_limit


class FairScheduler:
    def __init__(self, page, grid):
        self.page, self.grid = page, grid
        self.debt = 0.
        self.epoch = self.sequence = 0
        self.pending = deque()
        self.trace = os.environ.get("SGLANG_FSCHED_TRACE") == "1"
        self.round = None
        self.name = "native"

    def begin(self):
        while self.pending and self.pending[0][1].query():
            start, end, entry = self.pending.popleft()
            measured = start.elapsed_time(end) / 1000
            self.apply(entry, measured-entry[0], correction=True)
            if self.trace:
                log.info("FSCHED round=%d plan=%s actual=%s pre=%d gen=%d predicted=%.6f gpu=%.6f "
                         "gen_fraction=%.6f contention=%d weights=%s", entry[1], *entry[-1],
                         entry[0], measured, entry[5], entry[6], sorted(a.weight for a in entry[3]))
        self.round = None
        self.name = "native"

    def plan(self, waiting, chunk, running, slots):
        self.round = Round(waiting, chunk, running, slots, self.page, self.grid)
        self.name, queue, limit = self.round.choose(self.debt)
        return self.name, queue, limit

    def snapshot(self, batch):
        extend = batch.forward_mode.is_extend()
        count = len(batch.decoding_reqs or []) if extend else len(batch.reqs)
        npre = len(batch.reqs)-count
        gen = [account(r) for r in batch.reqs[npre:]]
        work = Counter()
        for r, n, start in zip(batch.reqs[:npre], (batch.extend_lens or [])[:npre],
                               (batch.prefix_lens or [])[:npre], strict=True):
            edge = self.round.edges.get(r) if self.round else None
            if edge and edge.start == start:
                work.update(edge.split(0, n))
            else:
                work[account(r)] += n
        n = sum((batch.extend_lens or [])[:npre])
        pt, gt = cost(n, count)
        dt = pt+gt
        rate = dict(self.round.rate) if self.round else {}
        # Noncontended or native fallback: use the actual runnable shape.
        if not rate:
            for cohort, mass in ((work, 1-GEN if gen else 1.), (gen, GEN if work else 1.)):
                rate.update({a: f*mass for a,f in shares({a:a.weight for a in cohort}).items()})
        bill = {a: v/n*pt/dt for a,v in work.items()} if n else {}
        bill.update({a: gt/count/dt for a in gen})
        both = self.round.both if self.round else bool(n and count)
        if not both:
            self.debt = 0.
            self.epoch += 1
        self.sequence += 1
        actual = "mixed" if n and count else ("prefill" if n else "generation")
        entry = (dt, self.sequence, self.epoch, rate, bill, gt/dt, both,
                 (self.name, actual, n, count))
        self.apply(entry, dt, correction=False)
        return entry

    def apply(self, entry, dt, correction):
        _, seq, epoch, rate, bill, gen_share, both, _ = entry
        if both and epoch == self.epoch:
            self.debt += dt*(GEN-gen_share)
        for a in rate.keys() | bill.keys():
            due, paid = rate.get(a, 0.), bill.get(a, 0.)
            a.lag += dt*(due-paid)
            if paid and not correction:
                a.last_paid, a.skips, a.since = seq, 0, 0.
            elif not paid and seq > a.last_paid:
                a.since += dt*due
                a.skips += not correction
