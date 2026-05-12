# Methodology

## What this task is

A speculative-decoding optimization benchmark for an agent swarm. Agents fork a
canonical implementation of the Leviathan/Chen 2022 spec-dec algorithm (single
chain, fixed γ, strict greedy rejection) and try to improve wall-clock decoding
throughput on a fixed model pair, evaluated on a stratified subset of
Spec-Bench prompts.

## The model pair (and why this one)

- **Target:** Llama-3.3-70B-Instruct, Q4_K_M GGUF (~42 GB)
- **Draft:** Llama-3.2-1B-Instruct, Q4_K_M GGUF (~750 MB)
- Both load on a single NVIDIA RTX A6000 (48 GB VRAM, sm_86, CUDA 12.9).

**Why this pair specifically:**

1. *Same family* (Llama-3.x): shared tokenizer is required for vanilla
   speculative sampling. Cross-family pairs (e.g., Qwen target + Llama draft)
   silently produce wrong outputs.
2. *Roughly 1.4% size ratio* (1B / 70B): puts us in a regime where draft cost
   is sub-target-cost, which is the regime where algorithmic improvements
   actually pay off. We tested Qwen3-32B + Qwen3-1.7B (5% ratio) earlier; in
   that regime, draft cost dominates and even production-grade tree drafting
   underperforms naive single-chain. The Llama pair gives the swarm room.
3. *Q4_K_M* matches realistic deployment regimes (vLLM, llama.cpp) and fits one
   A6000 with KV cache headroom. FP16 70B would need ~140 GB → 3+ GPUs.
4. *No trained head*: rules out EAGLE-style methods (which require multi-day GPU
   training of a draft head — infeasible per swarm iteration). Constraint by
   design — keeps the design space inference-time-algorithmic.

## The starting code

`workspace/initial_program.cpp` is ~250 lines of self-contained C++:

- Loads target + draft via llama.cpp's `common_init_from_params`.
- Tokenizes prompt, feeds prompt[:-1] to both models.
- Loop: draft autoregressively generates γ candidate tokens; target verifies
  the (id_last + γ drafts) batch in one parallel forward pass; walks accept/
  reject left-to-right (strict greedy: accept iff target's argmax matches the
  draft's choice; on first mismatch, replace with target's argmax); commits
  accepted prefix; trims both KV caches; emits TOKENS line + tok/s.
- γ defaults to 8.

The file links against `libllama` and `libllama-common` from llama.cpp's build
tree. Editing this file triggers a 2.5-second incremental build (only this one
.cpp recompiles + relinks).

## Scoring

Score is computed in `evaluator/eval.py` after running the agent's binary on
the configured prompt subset:

```
score = mean(agent_tok_per_sec) / mean(vanilla_tok_per_sec)
```

`vanilla_tok_per_sec` is a per-prompt cached number stored in
`reference_<tier>.jsonl`, generated once per tier by running
`llama-vanilla-bench` (target only, no spec-dec) on the same prompt set.

`agent_tok_per_sec` is the per-prompt mean of the agent's decoded throughput,
parsed from the binary's output (`decoded N tokens in T seconds, speed: X t/s`
or naive-spec's `=  X tok/s`).

If anti-cheat gates fail, `score = 0`.

### Anti-cheat gates

Without gates, several trivial cheats produce inflated scores:

| Cheat | What it looks like | Caught by |
|---|---|---|
| Just call draft-only, claim target's output | Draft generates 200+ tok/s, output is gibberish | LCS overlap < 50% |
| Always-accept (skip verify) | acceptance rate = 100%, output = draft's | Acceptance rate > 90% |
| No drafting at all (run vanilla, claim spec-dec) | Score is 1.0× anyway | Not technically cheating — gets ~1.0× score |

We use **two gates** (an earlier version had three; the third — `n_drafted > 0`
— turned out to be redundant because no-drafting = no speedup anyway):

1. **Acceptance rate** (`n_accept / n_drafted`) **must be in [5%, 90%].**
   - Below 5%: drafts never accepted = wasted work, agent has bug or trivial
     non-spec-dec implementation.
   - Above 90%: agent is bypassing the verify step. 90% is a safe upper
     threshold; legitimate algorithms typically land 30-70%.
2. **LCS overlap with reference ≥ 50%.** Compares the agent's output token
   sequence to the canonical reference's output token sequence
   (computed at reference generation time) via longest common subsequence.
   - Threshold of 50% is permissive enough that legitimate algorithmic variants
     pass (e.g., changing γ from 8 to 4 gets ~58% LCS due to FP non-associativity
     effects on argmax ties — see below) but tight enough that lossy methods
     (e.g., always-accept) get caught (they typically get ~30% LCS).

### "Lossless" is WRT FP precision, not bit-equal

A subtle but important point. Strict-rejection-sampling spec-dec at temperature
0 is *mathematically* equivalent to the target's greedy decoding — the proof
is in Leviathan & Chen 2022. **However**, the *exact* token sequence depends on
batch shape: cuBLAS/CUDA matmul is not associative in floating-point, so
different verify-batch shapes (γ=8 → 9 tokens per verify pass; γ=4 → 5 tokens)
produce slightly different logits. When two candidate tokens have logits
within FP precision of each other (which happens often in Q4-quantized models),
the argmax flips. Subsequent generation diverges from there.

This means **we cannot enforce bit-for-bit equality with vanilla**, even on
implementations that are provably correct. The `LCS overlap ≥ 50%` gate
captures the practical correctness notion — output content matches.

This is the same standard production frameworks (vLLM, llama.cpp,
TensorRT-LLM) use; none of them claim or enforce bit-equality across
implementations.

## Why three tiers

Per-eval wall time on the Llama-3.3-70B + Llama-3.2-1B pair on one A6000 is
~10 min/eval at fast tier. To enable both fast inner-loop iteration for the
swarm and high-confidence final validation, we provide three tiers with
different statistical properties:

| Tier | N | CV (~) | Distinguishes Δscore ≥ |
|---|---|---|---|
| fast | 40 | 5% | ~10% relative |
| medium | 80 | 3.5% | ~7% relative |
| full | 480 | 1.5% | ~3% relative |

Per-prompt CV on this pair is ~30% (high — different prompt categories produce
speedups ranging 1.8× to 5.7×). CV of the *mean* is per-prompt CV / √N. So
N=40 → CV ~5%, N=80 → CV ~3.5%, N=480 → CV ~1.5%.

### Proportional stratification on the fast tier

The fast tier samples 32 prompts from the "big-5" categories (translation,
summarization, rag, qa, math_reasoning — 80 prompts each in full Spec-Bench)
and 1 prompt from each of the 8 "small" categories (writing, roleplay,
reasoning, math, coding, extraction, stem, humanities — 10 prompts each in
full). This gives 80% big / 20% small, mirroring full's 83/17 split, with a
floor of 1 per category so the agent gets feedback on every category.

The earlier N=20 fast tier was un-proportional (75/25, with 4 small categories
entirely missing). For agents with category-specific recipes the
un-proportional fast tier under-estimated full-tier mean by ~10-15%,
weakening the inner-loop signal.

## Reference data lifecycle

For each tier we generate ONCE:
- `prompts_<tier>.jsonl` — the N selected prompts (with fixed seed, balanced
  across Spec-Bench subtasks: writing, roleplay, reasoning, math, coding,
  extraction, stem, humanities, translation, summarization, qa, math_reasoning,
  rag).
- `reference_<tier>.jsonl` — for each prompt: vanilla tok/s, canonical
  naive-spec tok/s, canonical naive-spec output tokens, mean accepted tokens
  per cycle.

These are generated by `scripts/generate_reference.py` and never need to be
regenerated unless you change the model pair, max_new_tokens, or the seed.

## Why max_new_tokens=128 — and what that means for the numbers

The harness runs at `max_new_tokens=128` to keep the swarm inner loop cheap
(fast-tier eval ~3 min). **This choice has a real, measurable effect on the
absolute speedup numbers** that anyone interpreting the results should
understand.

### The amortization effect

Spec-dec has fixed per-cycle overhead (draft autoregressive loop, verify-batch
construction, accept/reject walk, KV trim). That overhead is amortized over
the tokens the cycle accepts. Vanilla decoding has very little per-cycle
overhead. So:

- **Vanilla tok/s is essentially flat across `max_new`** (overhead amortizes
  after a few tokens and stays amortized).
- **Spec-dec tok/s grows with `max_new`** as overhead amortizes over more
  generated tokens.
- **The ratio (spec-dec / vanilla) therefore grows with `max_new`.**

### Measured sweep on this pair

The full sweep on Llama-3.3-70B + Llama-3.2-1B Q4_K_M, A6000, 40-prompt
data-driven subset:

| max_new | Vanilla tok/s | Canonical Leviathan γ=8 | Δ vs vanilla |
|---|---|---|---|
| 128 | 15.42 | 41.46 | **2.69×** |
| 256 | 15.50 | 46.00 | **2.97×** |
| 512 | 15.50 | 50.24 | **3.24×** |
| 1024 | 15.52 | 55.46 | **3.57×** |

The canonical baseline reads as **2.69×** on the current `max_new=128`
harness but **3.57×** at the Spec-Bench leaderboard convention
(`max_new=1024`). Same algorithm, same hardware, same prompts, different
operating point.

### What this means in practice

- **For swarm inner-loop iteration:** the choice of `max_new` doesn't
  change the optimization gradient (an algorithm that's better at `mn=128`
  is also better at `mn=1024`, just by a different absolute amount). The
  swarm reaches the same converged recipe either way.
- **For absolute reported numbers:** the choice does matter. If you report
  "the agent achieves N× on this benchmark", state the `max_new` setting,
  because the same recipe at `max_new=1024` will read substantially higher
  than the same recipe at `max_new=128`.
- **For comparison to published spec-dec papers:** most use `max_new=1024`
  (or 256). Match their convention when comparing.

### Why we picked 128 anyway

Wall-time arithmetic for swarm iteration:

| max_new | Fast-tier wall | Per swarm-iter inner loop (e.g., 20 iter) |
|---|---|---|
| 128 (current) | ~3 min | ~1 hour |
| 256 | ~5 min | ~1.7 hr |
| 1024 | ~24 min | ~8 hr |

For inner-loop hill-climbing the 128 setting is a deliberate compromise:
the *gradient* is preserved, only the *absolute numbers* shift.

### Recommended convention if you're publishing

Re-run the swarm's final selected recipe at `max_new=256` (most spec-dec
papers) or `max_new=1024` (Spec-Bench leaderboard) and report that as the
headline number, with a note about the operating-point dependence. The
fast-tier number is *internal* to the swarm — it's the cost function the
agent optimizes, not the number you'd put in an abstract.

## Persistent-process binary

As of the May 2026 refactor, `llama-naive-spec` and `llama-vanilla-bench`
load their models once and loop over prompts read from stdin (delimited by
`\n<<END_PROMPT>>\n`). The harness invokes the binary once per tier instead
of once per prompt. This avoids re-mmapping the 42 GB target between prompts
and roughly halves wall time. The agent's `initial_program.cpp` `main()`
implements the same outer loop; if the agent fully rewrites `main()`, they
must keep the stdin-delimited contract or the harness will fail to parse.
