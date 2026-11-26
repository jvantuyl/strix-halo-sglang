"""Paired, out-of-sample test of coupled feedback on entitlement and switch band.

CPU model only. Use client clocks (including TTFT and inter-delivery gaps),
not a moving entitlement, to select gains. Never fit on evaluation rows.
"""
import argparse
import csv
import random
from math import log
from statistics import mean, median

from fsched_lag import TimeBudget
from fsched_replay import Request, recorded
from fsched_sim import Config, simulate


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Feedback:
    """Bounded two-input PI. No derivative on sparse, quantized observations."""
    def __init__(self, gain, band_gain):
        self.gain, self.band_gain = gain, band_gain
        self.last = {}
        self.i = [0., 0.]
        self.at = None
        self.peak = [0., 0.]
        self.split, self.band = 2/3, .5
        self.updates = 0

    def decide(self, now, pre, gen):
        parts = [[i for r in rows for i in r.members] for rows in (pre, gen)]
        self.last = {i:self.last.get(i, now) for members in parts for i in members}
        if not all(parts):
            self.i = [0., 0.]
            self.peak = [0., 0.]
            self.at = None
            self.split, self.band = 2/3, .5
        else:
            for j, ids in enumerate(parts):
                self.peak[j] = max(self.peak[j], max(now-self.last[i] for i in ids))
            if self.at is None:
                self.at = now
            if (dt := now-self.at) >= 2.7:
                # Differential pressure shifts time towards the longer gap.
                # Common pressure changes the incentive for mixing. Try both
                # signs on TRAIN: this actuator has no known monotone gain.
                p, g = (clamp(v/2.7, 0, 2) for v in self.peak)
                for j,(e,k,mid,lo,hi,slew) in enumerate((
                    (p-g,self.gain,2/3,.5,.8,.06),
                    ((p+g)/2-.5,self.band_gain,.5,.25,2.,.25))):
                    prev = (self.split,self.band)[j]
                    lower,upper = max(lo, prev-slew),min(hi,prev+slew)
                    di = k*e*min(dt,5.4)/12
                    target = mid+k*e+self.i[j]+di
                    if not (target>upper and di>0 or target<lower and di<0):
                        self.i[j] += di
                    out = clamp(mid+k*e+self.i[j],lower,upper)
                    if j == 0:
                        self.split = out
                    else:
                        self.band = out
                self.at, self.peak = now,[0.,0.]
                self.updates += 1
        return self.split*100, self.band

    def delivered(self, now, served):
        self.last.update((i,now) for i in served)


def workload(kind, seed):
    rng = random.Random(seed)
    qs = []
    now = 0.
    for i in range(24):
        if kind == "interactive":
            long, spacing, out = i%7==0, .8, 200
        elif kind == "documents":
            long, spacing, out = i%3!=0, .35, 32
        elif kind == "trickle":
            long, spacing, out = i%4==0, 1.6, 128
        elif kind == "shift":
            long, spacing, out = (i%6==0 if i<12 else i%4!=0), (.2 if i<12 else 1.), (192 if i<12 else 48)
        else:
            raise ValueError(kind)
        if i:
            now += rng.expovariate(1/spacing)
        prompt = rng.choice((4096,8192,16384) if long else (96,160,320,512))
        limit = 32 if long else out
        w = rng.choice((.03,1.,4.) if kind in ("shift","trickle") else (.1,1.,2.))
        qs.append(Request(str(i),now,prompt,limit,rng.randint(4,limit),w,(("shared",64),)))
    return qs


def metrics(qs, cost, seed, gains):
    feedback = Feedback(*gains) if gains else None
    cfg = Config(percent=100*2/3,mixed_width=1)
    r = simulate(qs,"profile",cost,cfg,seed,save_rounds=True,timing=TimeBudget(),
                 band_factor=.5,feedback=feedback)
    reqs = r["requests"]
    assert all(x["tokens_produced"]==q.eos for q,x in zip(qs,reqs,strict=True))
    assert all(h["predicted_s"]<=2.7 for h in r["history"])
    assert abs(r["time"]["billed_s"]-r["time"]["entitled_s"])<1e-6
    last = {x["name"]:x["arrival"]+x["ttft"] for x in reqs}
    gap = 0.
    for h in r["history"]:
        end = h["start"]+h["seconds"]
        for name in h["generate"]:
            gap = max(gap,end-last[name])
            last[name] = end
    t = r["time"]
    return dict(short_s=median(x["time_running"] for x,q in zip(reqs,qs) if q.prompt<1000),
                long_ttft_s=max(x["ttft"] for x,q in zip(reqs,qs) if q.prompt>=1000),
                gap_s=gap,wall_s=r["wall"],lag_s=t["worst_since_service_s"],
                jain_min=min(v for window in r["windows"] for v in window.values()),
                gen_share=t["contended_gen_s"]/t["contended_s"],
                updates=feedback.updates if feedback else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    args = ap.parse_args()
    costs,_ = recorded()
    rows = []
    gains = [(.06,-.6),(.06,.6),(.15,-1.5),(.15,1.5)]
    def run(phase,kinds,seeds,candidates):
        scores = {k:[] for k in candidates}
        for kind in kinds:
            for seed in seeds:
                qs = workload(kind,seed)
                for k in [None,*candidates]:
                    m = metrics(qs,costs[2048],seed,k)
                    if k is None:
                        base = m
                    # Client-centric utility; changing the entitlement promises
                    # cannot directly improve this objective.
                    loss = mean(log(m[v]/base[v]) for v in ("short_s","long_ttft_s","gap_s","wall_s"))
                    if k:
                        scores[k].append(loss)
                    rows.append(dict(phase=phase,kind=kind,seed=seed,gains=str(k),loss=loss,**m))
        return scores
    train = run("train",("interactive","documents"),(7,11,17),gains)
    chosen = min(gains,key=lambda g:mean(train[g]))
    print("training mean paired log loss:",{str(g):mean(v) for g,v in train.items()},"selected",chosen)
    run("heldout",("interactive","documents","trickle","shift"),(31,47,61),[chosen])
    with open(args.csv,"x",newline="") as out:
        w = csv.DictWriter(out,fieldnames=rows[0],lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__=="__main__":
    main()
