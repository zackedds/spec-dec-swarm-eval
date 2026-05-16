# Fast-tier prompt selection

This benchmark's `fast` tier ships 40 prompts selected so that the mean of
the canonical-method speedup on the subset closely tracks the mean on the
full 480-prompt Spec-Bench. This document records how those 40 prompts were
chosen and what tracking error to expect.

## Why this matters

The fast tier exists for the swarm's inner loop — it has to be cheap enough
to run hundreds of times during a swarm session, but accurate enough that
candidate recipes can be ranked. Per-prompt speedup variance on this
benchmark is high (per-prompt CV ≈ 32% — translation prompts can score very
differently from creative-writing prompts). So a poorly-chosen 40-prompt
subset can systematically misrepresent the full-tier mean by ~5–8%, which is
larger than the within-subset sampling noise.

We measured this empirically: an a-priori proportional stratification
under-estimated the full-tier canonical speedup by **~4.5%**, with the bias
growing for recipes that disproportionately benefit input-grounded
categories. The data-driven subset shipped here reduces that bias to
**~0.5%**.

## How the subset was chosen

1. **Ran the canonical Leviathan baseline + two γ variants** (γ=8, γ=12,
   γ=16) on the full 480 prompts. Three methods spanning the chain-length
   axis gives per-prompt speedup vectors that are correlated but not
   identical — enough signal to pick a subset that's robust across the
   spec-dec design space, not just one particular operating point.
2. **Allocated 40 prompts across the 13 categories** proportionally to
   their counts in full Spec-Bench (80 each for the "big-5" — translation,
   summarization, rag, qa, math_reasoning — and 10 each for the "small-8"),
   with a floor of 1 per category. Result: 32 prompts from the big-5 (~80%
   of fast tier, matching full's 83%) plus 1 from each of the 8 small
   categories.
3. **Within each category**, ran random-restart sampling (4,000 trials) +
   local swap-improvement to find the subset of prompts whose **maximum
   per-method gap** between subset mean and category-full mean is minimized.
   Optimizing for the max across methods (rather than for a single method's
   bias) gives a subset that's robust to whichever recipe the swarm
   converges on.

## Measured tracking error

Subset-mean vs full-tier-mean for the canonical Leviathan baseline (γ=8,
fixed-chain, strict greedy):

| | Subset mean | Full mean | Gap | Relative |
|---|---|---|---|---|
| `fast` (40 prompts, data-driven) | 2.677× | 2.666× | +0.012 | **+0.43%** |

Same canonical baseline run on the *previous* a-priori-proportional
40-prompt subset (the legacy fast tier before this change):

| | Subset mean | Full mean | Gap | Relative |
|---|---|---|---|---|
| `fast` (40 prompts, a-priori) | 2.545× | 2.666× | -0.120 | **-4.52%** |

The bias direction matters: the previous subset *under*-reported full-tier
mean, which is the worst direction for a swarm whose job is to discover
recipes that look good on the fast tier. The new subset is within
sampling noise of the full-tier mean.

## Constraints and caveats

- **All 13 categories are represented** by at least one prompt — agents
  see feedback signal on every category, not just the high-prompt-count
  ones.
- **The selection used three canonical-method variants** (γ=8, γ=12, γ=16)
  during optimization. It's possible that a qualitatively different
  algorithm (e.g., a hypothetical method that exploits a pattern none of
  these three reward) could be tracked less well by this subset. The
  practical risk is small because most spec-dec recipes the swarm would
  reasonably explore are in the family the optimization covered.
- **The subset depends on the model pair.** If you change the target +
  draft models, the per-prompt speedup vectors change and the subset
  selection should be re-run from `data/spec_bench_questions.jsonl`. See
  `scripts/generate_reference.py` for the harness used to collect per-prompt
  data.

## Reproducing the selection

The per-method JSONL data from the original selection run isn't shipped in
this repo (it's tied to specific binary builds), but the methodology is
generic. To regenerate:

1. Pick 3 or more spec-dec methods that span the design space (varying γ
   on canonical, or one canonical + one with tree drafting, etc.).
2. Run each method on the full 480-prompt Spec-Bench at `max_new_tokens=128`
   (matching the harness default).
3. For each candidate 40-prompt subset (with the floor-of-1-per-category
   constraint), compute the maximum-across-methods gap between subset mean
   and category-full mean. Use random restarts + local swap to converge.
4. Save the resulting `prompts_fast.jsonl` and regenerate `reference_fast.jsonl`
   via `./setup.sh --regen-refs fast`.

A reasonable check is that the canonical baseline mean on the new subset
should fall within ~1% of the canonical baseline mean on the full 480.
