
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_dense.py — Denser benchmarks & extra plots for ALETHIA+KAIROS.

Adds:
- --prompt_file FILE   (# each line: "prompt || style" or "prompt" or "| style" for auto prompt)
- --seed_range A B     (inclusive range; appends to --seeds if both given)
- --n_bins K           (reliability bin count; default 20)
- Extra plots:
  * acceptance_vs_conf_threshold.png — sweep confidence gate and plot coverage & selective quality (full arm)
  * coherence_hist.png — histograms of coherence for accepted vs rejected (per arm)
Usage examples:
  python benchmark_dense.py --seeds 41 42 43 --n 40 --model gpt2 --out outbench
  python benchmark_dense.py --seed_range 41 55 --prompt_file big_prompts.txt --model gpt2 --out outbench_big
"""

import os, re, json, time, argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

# dynamic import of proto.py colocated with this file
import importlib.util
THIS_DIR = Path(__file__).resolve().parent
PROTO_PATH = (THIS_DIR / "proto.py").resolve()
spec = importlib.util.spec_from_file_location("proto", str(PROTO_PATH))
proto = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proto)

DEFAULT_TESTS = [
    (None, "analytical"),
    (None, "narrative"),
    ("The fundamental principles of machine learning include", "analytical"),
    ("In a world where dreams become reality,", "creative"),
    ("She opened the ancient book and discovered", "narrative"),
    ("The relationship between consciousness and", None),
    ("Beyond the kaleidoscope of imagination,", "creative"),
    ("The paradox of existence reveals", "analytical"),
]

def parse_prompt_file(p: Path) -> List[Tuple[Optional[str], Optional[str]]]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): 
                continue
            # split on '||' if present
            if "||" in s:
                prompt, style = [t.strip() or None for t in s.split("||", 1)]
                out.append((prompt if prompt else None, style if style else None))
            elif s.startswith("|"):
                out.append((None, s[1:].strip() or None))
            else:
                out.append((s, None))
    return out

def make_tests(n: int, prompt_file: Optional[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    if prompt_file:
        pf = Path(prompt_file)
        if pf.exists():
            tests = parse_prompt_file(pf)
            return tests[:n] if n and n > 0 else tests
    base = DEFAULT_TESTS.copy()
    if n <= len(base):
        return base[:n]
    styles = ["analytical", "creative", "narrative"]
    i = 0
    while len(base) < n:
        s = styles[i % 3]
        base.append((None, s))
        i += 1
    return base

def final_gate_pass(text: str, assessment: Dict[str, Any], target_style: Optional[str],
                    min_conf: float = 0.50, min_coh: float = 0.55, min_log: float = 0.50) -> bool:
    if re.search(r'\b(write|compose|generate|do not|don\'t|instructions?|topic)\b[: ]', text.strip(), flags=re.IGNORECASE):
        return False
    if len(re.findall(r'[^.!?]+[.!?]', text)) < 2:
        return False
    conf = float(assessment.get("confidence", 0.0))
    coh  = float(assessment.get("coherence", 0.0))
    logc = float(assessment.get("logic", 0.0))
    perp = float(assessment.get("perplexity", 1e9))
    origin = assessment.get("origin") or target_style or "analytical"
    style_caps = {
        "analytical": {"perp": 40.0, "coh": 0.45},
        "narrative":  {"perp": 85.0, "coh": 0.45},
        "creative":   {"perp": 110.0, "coh": 0.35},
    }
    caps = style_caps.get(origin, {"perp": 100.0, "coh": 0.40})
    meets_quality = (conf >= min_conf and coh >= max(min_coh, caps["coh"]) and perp < caps["perp"])
    if target_style == "analytical":
        meets_quality = meets_quality and (logc >= min_log)
    if target_style:
        probs = assessment.get("probs", {}) or {}
        tgt = float(probs.get(target_style, 0.0))
        others = sorted([v for k,v in probs.items() if k != target_style], reverse=True)
        margin = tgt - (others[0] if others else 0.0)
        style_ok = (origin == target_style) and (tgt >= 0.60) and (margin >= 0.15)
    else:
        style_ok = True
    return bool(meets_quality and style_ok)

@dataclass
class SampleLog:
    arm: str
    seed: int
    idx: int
    prompt: Optional[str]
    target_style: Optional[str]
    text: str
    accepted: int
    confidence: float
    coherence: float
    logic: float
    perplexity: float
    quality: float
    origin: str
    style_prob: float
    margin: float
    attempts: int
    recovery: int
    latency_s: float

def run_arm_baseline(model_name: str, seed: int, tests):
    proto.set_seeds(seed)
    agent = proto.AlethiaAgent(model_name=model_name, latent_dim=64)
    assessor = proto.KAIROSAssessor()
    logs = []
    for i, (prompt, style) in enumerate(tests):
        t0 = time.time()
        mem = agent.generate(prompt=prompt, style=style, max_length=50, temperature=0.9, attempt=0)
        ass = assessor.assess(mem)
        latency = time.time() - t0
        acc = 1 if final_gate_pass(mem.text, ass, style) else 0
        probs = ass.get("probs", {}) or {}
        tgtp = float(probs.get(style, 0.0)) if style else (max(probs.values()) if probs else 0.0)
        others = sorted([v for k,v in probs.items() if not style or k != style], reverse=True)
        margin = (tgtp - (others[0] if others else 0.0)) if style else 0.0
        logs.append(SampleLog("baseline", seed, i, prompt, style, mem.text, acc,
                              float(ass.get("confidence",0.0)), float(ass.get("coherence",0.0)),
                              float(ass.get("logic",0.0)), float(ass.get("perplexity",1e9)),
                              float(ass.get("quality_score",0.0)), str(ass.get("origin","unknown")),
                              float(tgtp), float(margin), 1, int(mem.recovery_count), float(latency)))
    return logs

def run_arm_assess_only(model_name: str, seed: int, tests):
    proto.set_seeds(seed)
    agent = proto.AlethiaAgent(model_name=model_name, latent_dim=64)
    assessor = proto.KAIROSAssessor()
    logs = []
    for i, (prompt, style) in enumerate(tests):
        t0 = time.time()
        mem = agent.generate(prompt=prompt, style=style, max_length=50, temperature=0.9, attempt=0)
        ass = assessor.assess(mem)
        latency = time.time() - t0
        acc = 1 if final_gate_pass(mem.text, ass, style) else 0
        probs = ass.get("probs", {}) or {}
        tgtp = float(probs.get(style, 0.0)) if style else (max(probs.values()) if probs else 0.0)
        others = sorted([v for k,v in probs.items() if not style or k != style], reverse=True)
        margin = (tgtp - (others[0] if others else 0.0)) if style else 0.0
        logs.append(SampleLog("assess_only", seed, i, prompt, style, mem.text, acc,
                              float(ass.get("confidence",0.0)), float(ass.get("coherence",0.0)),
                              float(ass.get("logic",0.0)), float(ass.get("perplexity",1e9)),
                              float(ass.get("quality_score",0.0)), str(ass.get("origin","unknown")),
                              float(tgtp), float(margin), 1, int(mem.recovery_count), float(latency)))
    return logs

def run_arm_full(model_name: str, seed: int, tests):
    proto.set_seeds(seed)
    system = proto.MetacognitiveGPT2System(model_name=model_name, seed=seed)
    logs = []
    for i, (prompt, style) in enumerate(tests):
        t0 = time.time()
        res = system.generate_with_metacognition(prompt=prompt, target_style=style, max_length=50, verbose=False)
        latency = time.time() - t0
        ass = res["assessment"]; txt = res["text"]
        acc = 1 if final_gate_pass(txt, ass, style) else 0
        probs = ass.get("probs", {}) or {}
        tgtp = float(probs.get(style, 0.0)) if style else (max(probs.values()) if probs else 0.0)
        others = sorted([v for k,v in probs.items() if not style or k != style], reverse=True)
        margin = (tgtp - (others[0] if others else 0.0)) if style else 0.0
        logs.append(SampleLog("full", seed, i, prompt, style, txt, acc,
                              float(ass.get("confidence",0.0)), float(ass.get("coherence",0.0)),
                              float(ass.get("logic",0.0)), float(ass.get("perplexity",1e9)),
                              float(ass.get("quality_score",0.0)), str(ass.get("origin","unknown")),
                              float(tgtp), float(margin), int(res.get("attempts_made",1)),
                              int(res.get("recovery_count",0)), float(latency)))
    return logs

def ece_brier(preds, labels, n_bins: int):
    preds = np.asarray(preds, dtype=float)
    labels = np.asarray(labels, dtype=int)
    brier = float(np.mean((preds - labels) ** 2))
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0; N = len(preds)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
        if mask.any():
            bin_acc = labels[mask].mean()
            bin_conf = preds[mask].mean()
            ece += (mask.sum() / N) * abs(bin_acc - bin_conf)
    return float(ece), float(brier)

def summarize(arm: str, seed: int, logs: List[SampleLog], n_bins: int):
    N = len(logs)
    accepted = [s for s in logs if s.accepted == 1]
    coverage = len(accepted) / max(1, N)
    selective_quality = mean([s.quality for s in accepted]) if accepted else 0.0
    style_hit = mean([1.0 if (s.target_style and s.origin == s.target_style and s.accepted) else 0.0 for s in logs]) if logs else 0.0
    mean_latency = mean([s.latency_s for s in logs]) if logs else 0.0
    mean_attempts = mean([s.attempts for s in logs]) if logs else 1.0
    preds = [s.confidence for s in logs]; labels = [s.accepted for s in logs]
    ece, brier = ece_brier(preds, labels, n_bins=n_bins)
    return {
        "arm": arm, "seed": seed, "total": N, "accepted": len(accepted),
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective_quality, 4),
        "style_hit_rate": round(style_hit, 4),
        "ece": round(ece, 4), "brier": round(brier, 4),
        "mean_latency_s": round(mean_latency, 3),
        "mean_attempts": round(mean_attempts, 2),
    }

def save_jsonl(path: Path, logs: List[SampleLog]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in logs:
            f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

def plot_coverage_vs_quality(rows, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    xs = [r["coverage"] for r in rows]; ys = [r["selective_accuracy"] for r in rows]
    labels = [f'{r["arm"]}-s{r["seed"]}' for r in rows]
    plt.scatter(xs, ys)
    for x, y, lab in zip(xs, ys, labels):
        plt.text(x, y, lab, fontsize=8, ha='left', va='bottom')
    plt.xlabel("Coverage")
    plt.ylabel("Selective Quality (accepted only)")
    plt.title("Coverage vs Selective Quality")
    plt.tight_layout()
    plt.savefig(outdir / "coverage_vs_quality.png", dpi=160)

def plot_reliability(rows_logs, outdir: Path, n_bins: int):
    outdir.mkdir(parents=True, exist_ok=True)
    arms = sorted(set(arm for arm, _ in rows_logs.keys()))
    plt.figure()
    for arm in arms:
        preds = []; labels = []
        for (a, seed), logs in rows_logs.items():
            if a != arm: continue
            preds += [s.confidence for s in logs]
            labels += [s.accepted for s in logs]
        if not preds: continue
        preds = np.asarray(preds, float); labels = np.asarray(labels, int)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        xs, ys = [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
            if mask.any():
                xs.append((lo+hi)/2); ys.append(labels[mask].mean())
        plt.plot(xs, ys, marker='o', label=arm)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical acceptance rate")
    plt.title(f"Reliability (ECE visual, bins={n_bins})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "reliability.png", dpi=160)

def plot_style_hit(rows, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    arms = [f'{r["arm"]}-s{r["seed"]}' for r in rows]
    vals = [r["style_hit_rate"] for r in rows]
    x = np.arange(len(arms))
    plt.bar(x, vals)
    plt.xticks(x, arms, rotation=45, ha='right')
    plt.ylabel("Style Hit-Rate (accepted only)")
    plt.title("Style Enforcement Effectiveness")
    plt.tight_layout()
    plt.savefig(outdir / "style_hit_rate.png", dpi=160)

def plot_acceptance_vs_conf_threshold(rows_logs, outdir: Path, min_conf_range=(0.4, 0.8), steps=9):
    # Sweep confidence threshold for FULL arm only; keep other gates same
    full_pairs = [k for k in rows_logs.keys() if k[0] == "full"]
    if not full_pairs: return
    lo, hi = min_conf_range
    grid = np.linspace(lo, hi, steps)
    plt.figure()
    for (arm, seed) in full_pairs:
        logs = rows_logs[(arm, seed)]
        xs = []; covs = []; quals = []
        for th in grid:
            accepted = [s for s in logs if final_gate_pass(s.text, {
                "confidence": s.confidence, "coherence": s.coherence, "logic": s.logic,
                "perplexity": s.perplexity, "origin": s.origin, "probs": {"analytical":0,"creative":0,"narrative":0}
            }, s.target_style, min_conf=th)]
            xs.append(th)
            covs.append(len(accepted)/max(1,len(logs)))
            quals.append(mean([s.quality for s in accepted]) if accepted else 0.0)
        plt.plot(xs, covs, marker='o', label=f'coverage-s{seed}')
        plt.plot(xs, quals, marker='x', label=f'quality-s{seed}')
    plt.xlabel("Confidence threshold (MIN_CONF)")
    plt.ylabel("Metric value")
    plt.title("Effect of Confidence Threshold (full arm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "acceptance_vs_conf_threshold.png", dpi=160)

def plot_coherence_hist(rows_logs, outdir: Path, bins=20):
    outdir.mkdir(parents=True, exist_ok=True)
    arms = sorted(set(arm for arm,_ in rows_logs.keys()))
    for arm in arms:
        all_logs = []
        for (a, seed), logs in rows_logs.items():
            if a == arm: all_logs += logs
        if not all_logs: continue
        coh_acc = [s.coherence for s in all_logs if s.accepted == 1]
        coh_rej = [s.coherence for s in all_logs if s.accepted == 0]
        plt.figure()
        plt.hist(coh_acc, bins=bins, alpha=0.6, label="accepted")
        plt.hist(coh_rej, bins=bins, alpha=0.6, label="rejected")
        plt.xlabel("Coherence score")
        plt.ylabel("Count")
        plt.title(f"Coherence distribution — {arm}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"coherence_hist_{arm}.png", dpi=160)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seeds", nargs="+", type=int, default=[])
    ap.add_argument("--seed_range", nargs=2, type=int, default=None)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--prompt_file", type=str, default=None)
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--out", type=str, default="outbench")
    args = ap.parse_args()

    seeds = list(args.seeds)
    if args.seed_range and len(args.seed_range) == 2:
        a, b = args.seed_range
        seeds += list(range(min(a,b), max(a,b)+1))
    if not seeds:
        seeds = [43]

    outdir = Path(args.out)
    logs_dir = outdir / "logs"
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    rows_logs: Dict[Tuple[str,int], List[SampleLog]] = {}

    for seed in seeds:
        tests = make_tests(args.n, args.prompt_file)

        logs_a = run_arm_baseline(args.model, seed, tests)
        save_jsonl(logs_dir / f"baseline_{seed}.jsonl", logs_a)
        row_a = summarize("baseline", seed, logs_a, n_bins=args.n_bins)
        all_rows.append(row_a); rows_logs[("baseline", seed)] = logs_a

        logs_b = run_arm_assess_only(args.model, seed, tests)
        save_jsonl(logs_dir / f"assess_only_{seed}.jsonl", logs_b)
        row_b = summarize("assess_only", seed, logs_b, n_bins=args.n_bins)
        all_rows.append(row_b); rows_logs[("assess_only", seed)] = logs_b

        logs_c = run_arm_full(args.model, seed, tests)
        save_jsonl(logs_dir / f"full_{seed}.jsonl", logs_c)
        row_c = summarize("full", seed, logs_c, n_bins=args.n_bins)
        all_rows.append(row_c); rows_logs[("full", seed)] = logs_c

    # save summary.csv
    import csv
    csv_path = outdir / "summary.csv"
    with csv_path.open("w", newline='', encoding="utf-8") as f:
        fieldnames = list(all_rows[0].keys()) if all_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        for r in all_rows: w.writerow(r)

    # plots
    plot_coverage_vs_quality(all_rows, plots_dir)
    plot_reliability(rows_logs, plots_dir, n_bins=args.n_bins)
    plot_style_hit(all_rows, plots_dir)
    plot_acceptance_vs_conf_threshold(rows_logs, plots_dir, min_conf_range=(0.4, 0.8), steps=9)
    plot_coherence_hist(rows_logs, plots_dir, bins=20)

    print(f"\nWrote: {csv_path}")
    print(f"Logs:  {logs_dir}/*.jsonl")
    print(f"Plots: {plots_dir}/*.png")

if __name__ == "__main__":
    main()