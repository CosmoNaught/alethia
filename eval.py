import json, csv, sys, math, numpy as np
from pathlib import Path

outdir = Path(sys.argv[1])           # e.g., out_audit
prompts_path = Path(sys.argv[2])     # /mnt/data/prompts.txt

VALID = {"analytical","creative","narrative"}

# 1) Rebuild TEST split to check no-leak
def load_prompts(fp):
    tagged, style_only, no_style = [], [], []
    for raw in fp.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"): continue
        if "||" in s:
            pr, st = [t.strip() for t in s.split("||", 1)]
            st = st.lower()
            if st in VALID: tagged.append((pr if pr else None, st))
        elif s.startswith("|"):
            st = s[1:].strip().lower()
            if st in VALID: style_only.append((None, st))
        else:
            no_style.append((s, None))
    all_items = tagged + style_only + no_style
    # Deterministic split with seed 42 (bench uses min(seeds))
    import random
    rng = random.Random(42)
    idx = list(range(len(all_items))); rng.shuffle(idx)
    n = len(idx); tr=int(n*0.7); va=int(n*0.1)
    test = [all_items[i] for i in idx[tr+va:]]
    return set([p for (p,_) in test if p])

# 2) Load logs
full_eval = list(Path(outdir/"logs").glob("full_eval_*.jsonl"))[0]
full_attempts = list(Path(outdir/"logs").glob("full_attempts_*.jsonl"))[0]
with full_eval.open("r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f]
with full_attempts.open("r", encoding="utf-8") as f:
    attempts = [json.loads(x) for x in f]

# 1) No-leak
test_prompts = load_prompts(prompts_path)
eval_prompts = [r["prompt"] for r in rows if r["prompt"]]
no_leak_ok = all(p in test_prompts for p in eval_prompts)

# 2) Style enforcement properties
acc = [r for r in rows if r["accepted"]==1 and r["target_style"]]
style_ok = all(
    (r["origin"] == r["target_style"]) and (r["style_prob"] >= 0.60) and (r["margin"] >= 0.15)
    for r in acc
)

# 3) Calibration helps — compute ECE with calibrated vs uncalibrated on the same eval set
def ece_brier(preds, labels, n_bins=20):
    preds=np.asarray(preds,float); labels=np.asarray(labels,int)
    bins=np.linspace(0.0,1.0,n_bins+1); N=len(preds); ece=0.0
    for i in range(n_bins):
        lo,hi=bins[i],bins[i+1]
        m=(preds>=lo)&(preds<(hi if i<n_bins-1 else hi+1e-9))
        if m.any(): ece += (m.sum()/N)*abs(labels[m].mean() - preds[m].mean())
    return float(ece)

labels   = [r["accepted"] for r in rows]
p_cal    = [r["confidence"] for r in rows]
p_uncal  = [r.get("confidence_uncal", r["confidence"]) for r in rows]
ece_cal  = ece_brier(p_cal, labels, 20)
ece_unc  = ece_brier(p_uncal, labels, 20)
cal_ok   = (ece_cal <= ece_unc + 1e-9)

# 4) Metacognition evidence
attempts_more_than_one = any(r.get("attempts_made",1) > 1 for r in attempts)
any_recovery = any(r.get("recovery_count",0) > 0 for r in attempts)
any_self_drift = any((r.get("self_drift",0.0) or 0.0) > 0 for r in attempts)
meta_ok = attempts_more_than_one or any_recovery
drift_ok = any_self_drift

print("NO-LEAK TEST PROMPTS:", "PASS" if no_leak_ok else "FAIL")
print("STYLE ENFORCEMENT   :", "PASS" if style_ok else "FAIL")
print(f"CALIBRATION (ECE)   : PASS (cal={ece_cal:.3f} ≤ uncal={ece_unc:.3f})" if cal_ok else
      f"CALIBRATION (ECE)   : FAIL (cal={ece_cal:.3f} > uncal={ece_unc:.3f})")
print("METACOGNITIVE LOOP  :", "PASS" if meta_ok else "FAIL", "(attempts/recovery)")
print("SELF-MODEL DRIFT    :", "PASS" if drift_ok else "FAIL")

# Non-zero coverage sanity
coverage = sum(labels)/max(1,len(labels))
print(f"COVERAGE: {coverage:.2%} over {len(labels)} samples")
