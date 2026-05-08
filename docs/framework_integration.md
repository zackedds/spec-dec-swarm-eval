# Framework integration notes

Notes for plugging this benchmark into an agent-swarm framework.

## Directory layout

The package follows the conventional `task.yaml` + `workspace/` + `evaluator/`
structure. Drop the directory anywhere your framework expects a task; the
`task.yaml` uses the `{{ eval_timeout_seconds }}` template variable for prompt
templating.

## The eval invokes a system-installed llama.cpp build

`evaluator/eval.py` reads `LLAMA_DIR` from the environment (default
`$SPEC_DEC_ROOT/llama.cpp`). The harness:

1. `cp workspace/initial_program.cpp $LLAMA_DIR/examples/naive-spec/naive-spec.cpp`
2. `cmake --build build --target llama-naive-spec` (~2.5 sec incremental rebuild)
3. Runs the resulting binary **once for the whole tier**, piping all prompts to
   it on stdin (delimited by `\n<<END_PROMPT>>\n`). The binary loads target +
   draft once and loops internally — no model reload between prompts.

The GGML-CUDA shared libs (~1 GB) make per-task duplication of the llama.cpp
tree expensive. If your framework expects fully-isolated workspaces per agent:

- **Serialize evals** through one shared `LLAMA_DIR` (simplest), OR
- Give each concurrent agent its own `LLAMA_DIR` (set per-agent env var). This
  costs ~12 GB disk × N agents but avoids races.

## The model files are also system-wide

The 42 GB target weights aren't worth duplicating. The harness reads them from
`SPEC_DEC_MODELS_DIR` (default `$SPEC_DEC_ROOT/models`). If your container
isolation requires model files inside the task workspace, mount the model
directory read-only.

## GPU allocation

`evaluator/eval.py` reads `CUDA_VISIBLE_DEVICES` from the environment. The
harness does NOT set it itself. Your framework controls which GPU each agent
runs on by setting that env var before invoking `task-eval`.

## Wall time budget per eval

| Tier | Wall time | Concurrent agents per 8-GPU box |
|---|---|---|
| fast (default) | ~3 min | up to 8 |
| medium | ~10 min | up to 8 |
| full | ~60–90 min | typically 1, possibly 2 with care |

Inner-loop swarm iteration should use `fast`. End-of-round / candidate
promotion uses `medium`. `full` is for paper-grade final validation only.

## Agent contract

`workspace/initial_program.cpp` `main()`:

1. Loads target+draft once
2. Reads prompts from stdin (split on `\n<<END_PROMPT>>\n`)
3. For each prompt: clears KV caches, runs the algorithm, emits a block
   between `=== PROMPT i BEGIN ===` and `=== PROMPT i END ===` markers
   containing per-prompt `decoded N tokens in T seconds = X tok/s`,
   `n_drafted N`, `n_accept N`, `TOKENS: id1 id2 ...` lines.

If the agent rewrites `main()` (full freedom — they can), they must keep this
outer contract or the harness will report "no block emitted" / "could not
parse". The canonical baseline they fork already implements this contract.

`max_new_tokens=128`. Speedup ratios at 128 are ~10–12% lower than at 256
(per-cycle setup amortizes over fewer generated tokens), but the optimization
landscape is preserved — agents see the same gradient.

## Anti-cheat gates can be tightened

Current gates are deliberately permissive (acceptance ∈ [5%, 90%], LCS ≥ 50%)
to allow legitimate algorithmic variants. Common cheats and the gate that
catches each:

- **"Just use the draft model"**: LCS gate catches this (draft output diverges
  from target output; LCS typically ~30%).
- **"Always-accept"**: acceptance-rate gate catches this (rate → 100%).
- **"Prefix-match a few tokens then garbage"**: tighten LCS threshold if you
  see this pattern.

Gate logic lives in `evaluator/eval.py` `main()` near the bottom. Adjust and
re-test. The unit-of-test is `./task-eval --tier fast` against known cheats.

## Recommended swarm config

Given the structural ceiling (~3.2× as a near-Pareto on this pair without
trained heads, per the prior-run audit in `docs/prior_runs.md`):

- **Iterations per agent:** 20–30 is sufficient. Beyond that, scores plateau
  and variance dominates.
- **Number of agents per round:** 8 fits on a typical 8-GPU box. Diversity
  comes from random seeds in agent reasoning, not parallel runs.
- **Use `fast` tier for inner loop**, `medium` for promoting candidates,
  `full` once at the end on the swarm's final selected recipe.

## Things that don't easily generalize

1. **The C++ build requires CUDA 12.x, gcc-11, and a CUDA arch flag matching
   your GPU.** `setup.sh` defaults to `sm_86` (A6000); pass `CUDA_ARCH=89`
   (L40), `90` (H100), etc. to override.
2. **The 70B target consumes ≥40 GB VRAM**, leaving ~5 GB for KV cache. A
   24 GB GPU would need different quantization (Q3 or smaller target).
3. **Reference data is model-specific.** If you change the model pair,
   max_new_tokens, or the prompt set, you must regenerate reference for that
   tier (`./setup.sh --regen-refs`) or scoring breaks.

## Open questions

1. **CV calibration on this exact pair.** The fast/medium/full CV estimates in
   `methodology.md` are extrapolated. Worth running 5 seeds on the canonical
   baseline at each tier to nail down actual variance.
2. **Direct comparison to Blazedit / Ouroboros K=1 / RASD on the same harness.**
   The 3.18× / 3.91× speedup comparison in `prior_runs.md` is to canonical
   Leviathan only. Same-harness numbers for those published methods would
   strengthen any "matches state-of-the-art" claim.
3. **Multi-GPU reference generation.** `generate_reference.py` runs on one GPU
   at a time. Splitting prompts across GPUs would cut full-tier ref-gen wall
   time from ~90 min to ~15 min.
