# Results So Far

Summary of agent-swarm runs on this task in April 2026, before the package
was cleaned up for handoff.

## Run 1 — Qwen3-32B + Qwen3-1.7B Q4_K_M (24 hr, 18 experiments)

**Outcome: marginal improvement, structural ceiling.**

- Baseline (canonical Leviathan): 1.91×
- Agent's best: 1.99× (+4% relative)
- What the agent found: micro-optimizations (AVX-512 argmax, pre-allocation,
  skip-last-decode) that nibble at the ~2% range. Tried various γ values,
  PLD-style n-gram lookahead — none of these moved the score meaningfully on
  this pair.

**Why so weak:** for the Qwen pair, draft cost is 5% of target (1.7B / 32B);
draft work dominates the cycle. Any algorithmic change that *adds* draft work
(tree drafts, dynamic γ with longer chains) makes things worse. The structural
ceiling on this pair is ~2.0×.

We later confirmed via direct ceiling probe that llama.cpp's production
`llama-speculative` binary (with KV-cache reuse, tree drafting capability, etc.)
also gets only 1.86× on this pair — essentially identical to our naive impl.
The pair is over-balanced, and that's why it makes a poor benchmark.

## Run 2 — Llama-3.3-70B + Llama-3.2-1B Q4_K_M (4 hr, 30 + 10 experiments)

**Outcome: meaningful improvement, structural ceiling much higher.**

- Baseline (canonical Leviathan, fast tier 20-prompt): **2.79×**
- Agent's best (fast tier): **3.18×** (+14% relative)
- Agent's best (full Spec-Bench, 417/480 prompts): **3.91×**

> **Note on numbers:** the 2.79× / 3.18× / 3.91× above were measured at
> `max_new_tokens=256` and (for fast tier) on the legacy un-proportional
> N=20 stratification. After the May 2026 refactor (persistent process,
> `max_new_tokens=128`, N=40 proportional fast tier) the canonical baseline
> reads ~2.57× on fast and the agent recipe should read ~10–12% lower in
> absolute terms while preserving its relative win over canonical. For
> paper-grade headline numbers, re-run the agent's best at `max_new=256`.

The agent converged on a single recipe and reproduced it 3× independently
(σ ≈ 0.01 across reruns). The recipe:

> γ=16 chain-form draft from the 1B model, then search the generated context
> for a 4-gram match of the draft's last 4 tokens (last-match wins on ties),
> append up to 18 PLD-style "extension" tokens to the same chain, single
> verify pass with strict greedy rejection.

### Per-category breakdown (full Spec-Bench, 417/480 prompts)

| Category | n | Vanilla t/s | Agent t/s | Speedup |
|---|---|---|---|---|
| translation | 80 | 15.59 | 88.82 | **5.70×** |
| math | 10 | 15.58 | 70.28 | 4.51× |
| math_reasoning | 57 | 15.57 | 67.23 | 4.32× |
| rag | 80 | 15.34 | 63.76 | 4.16× |
| extraction | 10 | 15.29 | 54.30 | 3.55× |
| summarization | 72 | 15.35 | 53.49 | 3.48× |
| coding | 10 | 15.60 | 49.53 | 3.18× |
| stem | 10 | 15.59 | 43.21 | 2.77× |
| reasoning | 10 | 15.47 | 42.06 | 2.72× |
| qa | 48 | 15.60 | 37.44 | 2.40× |
| humanities | 10 | 15.61 | 35.75 | 2.29× |
| writing | 10 | 15.61 | 30.46 | 1.95× |
| roleplay | 10 | 15.58 | 28.53 | 1.83× |
| **OVERALL** | 417 | 15.49 | 60.50 | **3.91×** |

### What the agent rejected

In iterations 31–40 we suggested four specific directions; the agent
empirically rejected three of them and ran into an architectural blocker on
the fourth:

| Direction | Result | Agent's diagnosis |
|---|---|---|
| PLD cross-validation w/ draft agreement | 3.16 | "Rag prompts benefit from PLD precisely when the draft model is wrong (document quoting). Cross-validation filters out the wins." Counter-intuitive but correct. |
| Fixed-batch-shape padding for FP determinism | 2.74 | "Overhead of 18 padding tokens completely offsets FP consistency benefit. Not viable." |
| Adaptive γ (probe-and-commit) | 3.02 | "Roleplay gains +7 t/s but rag drops badly from FP divergence." |
| Two-chain tree (draft + PLD parallel) | 0.0 (build error) | `llama_memory_seq_cp` requires full KV buffer — would need different implementation (separate context instances) |

These rejections are interesting on their own merit — they're empirical
findings about the design space.

## Novelty audit (sub-agent adversarial review)

We commissioned an adversarial novelty audit on the agent's recipe. The
auditor found:

**Verdict: NOT NOVEL.**

The agent's "model draft → n-gram extension on same chain" is *materially
equivalent* to:

- **Blazedit** (UIUC ISE, HuggingFace blog, Oct 2025) — multi-layer speculation
  combining model draft + PLD on a single chain, single verify pass. Same two
  components, same single-chain topology, same single verify. The agent's
  variant is `draft → PLD-once`; Blazedit's is `PLD ↔ draft interleaved`.
  Permutation, not architectural difference.
- **Ouroboros** (Zhao et al., EMNLP 2024, K=1 case) — "phrase candidate pool"
  appended to model draft tokens. K=1 reduces to the audited algorithm exactly.

The auditor also noted the closest tree-merging approach (RASD, ACL Findings
2025) is *architecturally distinct* from the agent's chain-form. So the agent
discovered a chain-form analogue of RASD — but Blazedit and Ouroboros already
cover the chain-form case. The agent did not invent a new algorithm.

What's defensibly novel-or-near-novel:

1. **The FP-precision boundary observation** at extension cap = 18 (cap = 20
   crossed a tie-break boundary that hurts). This is an engineering finding
   about Q4 quantization at our specific batch shapes. I haven't found this
   documented elsewhere.
2. **First public benchmark on Llama-3.3-70B + Llama-3.2-1B Q4_K_M on a single
   A6000.** Modest empirical contribution — published numbers on this pair
   exist on H200 (3.55×, NVIDIA) and MI300X (2.31×, AMD), but not on consumer
   A6000.

## What this means for the paper

The spec-dec task here is **not a "swarm invents new SOTA algorithm" story**.
It's a **"swarm autonomously converges to known-good algorithms"** story.

For a paper framing swarms as exploring well-charted optimization spaces and
converging to high-performing solutions, this is positive evidence. Agents
independently rediscovering Blazedit / Ouroboros-class recipes — with
mechanistically-grounded ablations rejecting alternatives — is exactly the
behavior such frameworks are meant to demonstrate.

What the paper **shouldn't** claim:
- "Swarm-discovered novel speculative decoding algorithm"
- "First to combine model drafts with retrieval extensions"
- "New SOTA for speculative decoding on this pair"

What the paper **can** claim:
- "Swarm autonomously navigates the spec-dec design space and converges,
  across multiple seeds with low variance, on a recipe that matches published
  state-of-the-art (Blazedit / Ouroboros K=1) for chain-form spec-dec on
  consumer-GPU Q4_K_M deployment of a Llama-3.3-70B + Llama-3.2-1B pair."
- "Per-category breakdown of swarm-discovered methods identifies translation/
  math (5.7× / 4.5×) as primary beneficiaries and creative-writing/roleplay
  (1.95× / 1.83×) as the structural bottleneck — a finding consistent with
  the input-grounded-task hypothesis underlying PLD."
- "Empirical evidence (40 experiments, 4 plausible-but-rejected
  alternatives) that the swarm correctly diagnoses *why* candidate
  improvements fail, including the FP-non-associativity issue at variable
  verify-batch shapes."

## Notes on these numbers

The 3.18× / 3.91× speedups above were measured at `max_new_tokens=256` on the
legacy N=20 fast tier and on full Spec-Bench respectively. The recipe itself
is not shipped with this repo — agents start from the canonical Leviathan
baseline in `workspace/initial_program.cpp`. The audit here describes the
search space and where prior agents converged.
