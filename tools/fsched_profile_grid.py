#!/usr/bin/env python3
"""Sweep the fixed-cap profile scheduler on the six recorded 18+2 bursts."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
from statistics import median

from fsched_lag import TimeBudget
from fsched_replay import recorded
from fsched_sim import Config, simulate


DISCOUNTS = range(1, 9)
TIME_SPLITS = tuple(Fraction(2)**k for k in range(4, -5, -1))
BANDS = (.25, .5, 1., 2., 3., 4.)
PRIMARY = (2048, "p1-8-mixed")


def grid(discounts=DISCOUNTS):
    costs, traces = recorded()
    bursts = [(chunk, name, qs) for chunk, name, qs in traces if name.endswith("-mixed")]
    assert len(bursts) == 6
    for chunk, name, qs in bursts:
        # Short and long refer to prompt length. The recording also identifies
        # these groups by the client's 200- and 64-token generation limits.
        shorts = {q.name for q in qs if q.requested == 200}
        assert len(shorts) == 18 and len(qs) == 20
        for band in BANDS:
            for discount in discounts:
                # Do not round the share to an integer percentage: 2:1 is
                # two thirds of contended service time, not 66% or 67%.
                cfg = Config(percent=float(100 * discount / (discount + 1)), mixed_width=1)
                r = simulate(qs, "profile", costs[chunk], cfg, seed=7, save_rounds=True,
                             timing=TimeBudget(), band_factor=band)
                t = r["time"]
                assert all(q.eos == x["tokens_produced"] for q, x in zip(qs, r["requests"], strict=True))
                assert all(h["predicted_s"] <= h["budget_s"] == 2.7 for h in r["history"])
                assert r["totals"]["prefill"] == 52964
                assert sum(x["tokens_produced"] for x in r["requests"]) == 3728
                completion = [x["completion"] - x["arrival"] for x in r["requests"] if x["name"] in shorts]
                lag = max(h["since_service_lag_s"] for h in r["history"])
                assert lag == t["worst_since_service_s"]
                row = dict(calibration=chunk, trace=name, prefill_discount=discount, band=band,
                           median_short_s=median(completion), worst_since_s=lag, wall_s=r["wall"],
                           contended_gen_share=t["contended_gen_s"] / t["contended_s"])
                if discounts == TIME_SPLITS:
                    last = {x["name"]: x["arrival"] + x["ttft"] for x in r["requests"] if x["name"] in shorts}
                    gap = 0.
                    for h in r["history"]:
                        end = h["start"] + h["seconds"]
                        for key in h["generate"].keys() & shorts:
                            gap = max(gap, end - last[key])
                            last[key] = end
                    row.update(time_split=label(discount), target_gen_share=float(1/(1+discount)),
                               max_short_interdelivery_s=gap,
                               max_long_ttft_s=max(x["ttft"] for x in r["requests"] if x["name"] not in shorts))
                yield row


def label(d):
    return f"{d.numerator}:{d.denominator}"


def print_table(rows, title, key, precision, discounts):
    print(f"\n### {title}\n")
    print("| Band / " + ("time split" if discounts == TIME_SPLITS else "discount") + " | " +
          " | ".join(map(label, discounts)) + " |")
    print("|---:" * (len(discounts) + 1) + "|")
    for b in BANDS:
        cells = [f"{rows[b, d][key]:.{precision}f}" for d in discounts]
        print(f"| {b:g}× | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-splits", action="store_true",
                        help="sweep 16:1 through 1:16 prefill:generation time shares")
    parser.add_argument("--csv", type=Path, help="create a new per-replay CSV (no overwrite)")
    args = parser.parse_args()
    discounts = TIME_SPLITS if args.time_splits else DISCOUNTS
    results = list(grid(discounts))
    if args.csv:
        with args.csv.open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=results[0], lineterminator="\n")
            w.writeheader()
            w.writerows(results)
    primary = {(r["band"], r["prefill_discount"]): r for r in results
               if (r["calibration"], r["trace"]) == PRIMARY}
    print("# Profile scheduler grid: 2k arm, pass 1 (18 short + 2 long)")
    print_table(primary, "Median short-request completion (wall seconds)", "median_short_s", 2, discounts)
    print_table(primary, "Worst lag since service (owed service seconds)", "worst_since_s", 3, discounts)


if __name__ == "__main__":
    main()
