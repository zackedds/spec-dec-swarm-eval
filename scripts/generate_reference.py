#!/usr/bin/env python3
"""Generate prompts + reference data for a tier.

Picks N prompts from Spec-Bench (fixed seed, balanced across categories), runs
vanilla autoregressive + canonical naive Leviathan on each, saves
prompts_<tier>.jsonl and reference_<tier>.jsonl in evaluator/.
"""
import argparse, json, os, random, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.path.expanduser(os.environ.get("SPEC_DEC_ROOT", "~/spec-dec-work")))
PKG = Path(__file__).resolve().parent.parent
EVAL_DIR = PKG / "evaluator"

# Default location for the Spec-Bench prompts is the vendored copy in the repo.
SPECBENCH_QFILE = Path(os.environ.get("SPEC_DEC_QUESTIONS",
                                      str(PKG / "data" / "spec_bench_questions.jsonl"))).expanduser()

LLAMA_DIR = Path(os.environ.get("LLAMA_DIR", str(ROOT / "llama.cpp"))).expanduser()
BIN_VAN = LLAMA_DIR / "build/bin/llama-vanilla-bench"
BIN_NAI = LLAMA_DIR / "build/bin/llama-naive-spec"

MODELS = Path(os.environ.get("SPEC_DEC_MODELS_DIR", str(ROOT / "models"))).expanduser()
TARGET = Path(os.environ.get("SPEC_DEC_TARGET",
                              str(MODELS / "llama3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf"))).expanduser()
DRAFT  = Path(os.environ.get("SPEC_DEC_DRAFT",
                              str(MODELS / "llama3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf"))).expanduser()

GAMMA = 8
MAX_NEW = 128
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RUN_TIMEOUT_SEC = 12000  # whole-batch timeout (covers full tier ~480 prompts)

PROMPT_DELIM = "\n<<END_PROMPT>>\n"
BLOCK_RE = re.compile(r"=== PROMPT (\d+) BEGIN ===(.*?)=== PROMPT \1 END ===", re.S)
TPS_RE = re.compile(r"=\s+([\d.]+)\s+tok/s")
TOKENS_RE = re.compile(r"^TOKENS:((?:\s+-?\d+)+)\s*$", re.M)
DRAFTED_RE = re.compile(r"n_drafted\s+(\d+)")
ACCEPT_RE = re.compile(r"n_accept\s+(\d+)")


def run_persistent(bin_path, extra_args, prompts, timeout=RUN_TIMEOUT_SEC):
    cmd = [str(bin_path), "-m", str(TARGET),
           "-ngl", "99", "-c", "4096", "-n", str(MAX_NEW), "--temp", "0"] + extra_args
    payload = PROMPT_DELIM.join(prompts) + PROMPT_DELIM
    r = subprocess.run(cmd, input=payload, capture_output=True, text=True,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": GPU},
                       timeout=timeout)
    return r.stdout, r.stderr, r.returncode


def parse_blocks(stdout, n_prompts):
    """Returns list of dicts (one per prompt, in order)."""
    blocks = {int(m.group(1)): m.group(2) for m in BLOCK_RE.finditer(stdout)}
    out = []
    for i in range(n_prompts):
        block = blocks.get(i)
        if block is None:
            out.append(None)
            continue
        tm = TOKENS_RE.search(block)
        sm = TPS_RE.search(block)
        if not (tm and sm):
            out.append(None); continue
        d = {"tokens": [int(x) for x in tm.group(1).split()],
             "tps": float(sm.group(1))}
        am = ACCEPT_RE.search(block); dm = DRAFTED_RE.search(block)
        if am: d["n_accept"]  = int(am.group(1))
        if dm: d["n_drafted"] = int(dm.group(1))
        out.append(d)
    return out


def vanilla_all(prompts):
    so, se, rc = run_persistent(BIN_VAN, [], prompts)
    if rc != 0 and "PROMPT 0 END" not in so:
        print(f"vanilla binary exit {rc}; stderr tail:\n{se[-1500:]}")
    return parse_blocks(so, len(prompts))


def naive_all(prompts):
    extra = ["-md", str(DRAFT), "-ngld", "99",
             "--draft-min", str(GAMMA), "--draft-max", str(GAMMA), "--draft-p-min", "0.0"]
    so, se, rc = run_persistent(BIN_NAI, extra, prompts)
    if rc != 0 and "PROMPT 0 END" not in so:
        print(f"naive binary exit {rc}; stderr tail:\n{se[-1500:]}")
    return parse_blocks(so, len(prompts))


def pick_prompts(n):
    items = [json.loads(l) for l in open(SPECBENCH_QFILE)]
    by_cat = {}
    for r in items:
        by_cat.setdefault(r["category"], []).append(r)
    rng = random.Random(42)
    sampled = []
    if n == 480:
        sampled = items
    elif n == 80:
        # balanced 80-prompt set: 16 from MT-bench mix + 16 from each of 4 subtasks
        mt_cats = {"writing","roleplay","reasoning","math","coding","extraction","stem","humanities"}
        mt_pool = [r for r in items if r["category"] in mt_cats]
        sampled = rng.sample(mt_pool, 16)
        for sub in ["translation", "summarization", "qa", "math_reasoning"]:
            sampled.extend(rng.sample(by_cat[sub], 16))
    elif n == 40:
        # fast tier: 32 big-5 + 8 small-8 = 40 prompts.
        # Mirrors full-tier proportions (full = 83% big / 17% small) at 80/20,
        # with floor-of-1 per category so all 13 cats get feedback.
        # CV on mean ~5% (sufficient for hill-climbing on Δ ≥ 10% relative).
        big_cats = ["translation", "summarization", "rag", "qa", "math_reasoning"]
        small_cats = ["writing", "roleplay", "reasoning", "math",
                      "coding", "extraction", "stem", "humanities"]
        big_counts = [7, 7, 6, 6, 6]   # 32 total across big-5
        rng.shuffle(big_counts)
        for cat, k in zip(big_cats, big_counts):
            sampled.extend(rng.sample(by_cat[cat], k))
        for cat in small_cats:
            sampled.extend(rng.sample(by_cat[cat], 1))
    else:
        # legacy fast tier (n==20): 5 MT-bench mix + 3 each of 5 big subtasks = 20
        mt_cats = {"writing","roleplay","reasoning","math","coding","extraction","stem","humanities"}
        mt_pool = [r for r in items if r["category"] in mt_cats]
        sampled = rng.sample(mt_pool, 5)
        for sub in ["translation", "summarization", "qa", "math_reasoning", "rag"]:
            sampled.extend(rng.sample(by_cat[sub], 3))
    return [{"question_id": r["question_id"], "category": r["category"],
             "prompt": r["turns"][0]} for r in sampled[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["fast", "medium", "full"])
    ap.add_argument("--n-prompts", type=int, required=True)
    args = ap.parse_args()

    PROMPTS_FILE = EVAL_DIR / f"prompts_{args.tier}.jsonl"
    REF_FILE = EVAL_DIR / f"reference_{args.tier}.jsonl"

    print(f"=== Generating {args.tier} tier reference ({args.n_prompts} prompts) ===")
    prompts = pick_prompts(args.n_prompts)
    print(f"sampled {len(prompts)} prompts")
    with open(PROMPTS_FILE, "w") as f:
        for p in prompts:
            f.write(json.dumps(p) + "\n")

    prompt_texts = [p["prompt"] for p in prompts]

    print(f"\n[1/2] vanilla pass on {len(prompts)} prompts...")
    t_v = time.time()
    v_blocks = vanilla_all(prompt_texts)
    print(f"  vanilla pass complete in {(time.time()-t_v)/60:.1f} min")

    print(f"[2/2] naive spec-dec pass on {len(prompts)} prompts...")
    t_n = time.time()
    n_blocks = naive_all(prompt_texts)
    print(f"  naive pass complete in {(time.time()-t_n)/60:.1f} min")

    t0 = t_v
    results = []
    for i, (p, vb, nb) in enumerate(zip(prompts, v_blocks, n_blocks)):
        if vb is None or nb is None:
            print(f"[{i:3d} {p['category']:14s}] FAIL  vb={vb is not None} nb={nb is not None}")
            continue
        v_tps = vb["tps"]; s_tps = nb["tps"]
        speedup = s_tps / v_tps if v_tps > 0 else 0.0
        results.append({
            "question_id": p["question_id"], "category": p["category"], "prompt": p["prompt"],
            "vanilla_tok_per_sec": v_tps, "vanilla_n_tokens": len(vb["tokens"]),
            "naive_tokens": nb["tokens"], "naive_tok_per_sec": s_tps,
            "naive_speedup": speedup,
            "naive_n_drafted": nb.get("n_drafted"),
            "naive_n_accept":  nb.get("n_accept"),
        })
        if i % max(1, len(prompts)//40) == 0 or i < 5:
            print(f"[{i:3d}/{len(prompts)} {p['category']:14s}] van {v_tps:5.1f}  naive {s_tps:5.1f} = {speedup:.2f}x")

    with open(REF_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    dt = time.time() - t0
    print(f"\nSaved {len(results)} entries to {REF_FILE} in {dt/60:.1f} min")
    if results:
        van = statistics.mean(r["vanilla_tok_per_sec"] for r in results)
        nai = statistics.mean(r["naive_tok_per_sec"] for r in results)
        print(f"  mean vanilla = {van:.2f}  mean naive = {nai:.2f}  speedup = {nai/van:.3f}x")


if __name__ == "__main__":
    main()
