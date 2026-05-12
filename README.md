# spec-dec-swarm-eval

A speculative-decoding optimization benchmark for agent swarms. Agents fork a
single C++ file implementing canonical Leviathan/Chen 2022 speculative decoding,
modify the algorithm, and are scored on wall-clock decoding speedup over a
vanilla autoregressive baseline on a stratified subset of Spec-Bench prompts.

The benchmark target is **Llama-3.3-70B-Instruct** (Q4_K_M GGUF, ~42 GB) with
**Llama-3.2-1B-Instruct** (Q4_K_M GGUF, ~750 MB) as the draft. The 1B/70B size
ratio (~1.4%) puts agents in a regime where draft cost is well below target
cost — the regime where algorithmic improvements actually pay off.

## Hardware requirements

- 1× NVIDIA GPU with **≥48 GB VRAM** (verified on RTX A6000, sm_86; should also
  work on A100 80 GB, H100, RTX 6000 Ada — anything that fits a 42 GB Q4_K_M
  weight + KV cache).
- **~60 GB free disk** for the 42 GB target + 750 MB draft + 12 GB llama.cpp
  build tree + headroom. `setup.sh` pre-flights this.
- CUDA 12.x with `nvcc` and matching driver. The default config pins
  `sm_86`; override via `CUDA_ARCH=89` (L40), `90` (H100/H200), etc.
- gcc-11 (gcc-12+ is rejected by NVCC for some kernels), cmake 3.22+, git, python3.10+.

If your gcc-11 isn't at `/usr/bin/gcc-11` (e.g., devtoolset on RHEL,
`/usr/local/bin/gcc-11` on macOS-built CUDA), set:

```bash
GCC_HOST=/path/to/gcc-11 GXX_HOST=/path/to/g++-11 ./setup.sh
```

If you swap `SPEC_DEC_TARGET` / `SPEC_DEC_DRAFT` to a **gated** HuggingFace
repo (e.g., the Meta originals), run `huggingface-cli login` before invoking
`setup.sh` so the download has a token.

## Quick start

```bash
git clone https://github.com/zackedds/spec-dec-swarm-eval.git
cd spec-dec-swarm-eval
./setup.sh                 # 10–30 min depending on network (clone llama.cpp, build CUDA, download 43 GB models)
cd evaluator
./task-eval --tier fast    # ~3 min on canonical baseline → SCORE: ~2.57x
```

That's it. `task-eval` compiles `workspace/initial_program.cpp` (the agent's
editable file), runs it on the configured prompt subset, scores it, prints
`SCORE: <float>` as the last line.

## Three evaluation tiers

Same scoring methodology in all three; trade wall-clock for statistical
confidence.

Wall-time numbers below are measured on the canonical baseline (Llama-3.3-70B
target + Llama-3.2-1B draft, A6000 sm_86, ~4.0 s/prompt). Agent recipes that
generate longer / more drafted tokens may run somewhat slower per prompt.

| Tier | Prompts | Wall time | CV | Use case |
|---|---|---|---|---|
| `fast` | 40 (stratified, all 13 cats) | ~2.8 min | ~5% | **Swarm inner loop.** Distinguishes recipes that differ by ≥10%. |
| `medium` | 80 (stratified) | ~5.5 min | ~3.5% | **Round-end candidate validation.** |
| `full` | 480 (full Spec-Bench) | ~32 min | ~1.5% | **Paper-grade final number.** |

Reference data for all three tiers is shipped pre-generated in
`evaluator/{prompts,reference}_*.jsonl`. Reference token IDs are tied to the
specific GGUF files setup.sh downloads — if you swap models, regenerate via
`./setup.sh --regen-refs`.

The fast tier is **proportionally stratified** to mirror full's category
distribution (32 prompts from the 5 high-count categories, 1 from each of the
8 low-count categories — 80% / 20% split, matching full's 83% / 17%). This
keeps fast-tier mean an unbiased estimator of full-tier mean within sampling
noise.

## Persistent-process model

The `naive-spec` and `vanilla-bench` binaries load their target+draft models
**once** and loop over prompts read from stdin (delimited by
`\n<<END_PROMPT>>\n`). One binary invocation per tier, not one per prompt.
This avoids re-mmapping the 42 GB target between prompts and is roughly 5×
faster on full-tier evals than the naive per-prompt-spawn pattern.

`max_new_tokens=128`. **This is a deliberate inner-loop choice that affects
the absolute speedup numbers** — spec-dec's per-cycle overhead amortizes
over generated tokens, so the same algorithm reads higher at higher
`max_new`. The canonical baseline measures 2.69× at `mn=128` and 3.57× at
`mn=1024` (Spec-Bench leaderboard standard); the optimization gradient is
preserved across the range but the absolute numbers shift. **For any
externally reported number, re-run at `max_new=256` (most spec-dec papers)
or `1024` (leaderboard parity)** before quoting it. Full sweep + reasoning
in `docs/methodology.md`.

## What an agent does

1. Reads `workspace/initial_program.cpp` (~250 lines, all algorithm logic
   visible — model loading + draft loop + verify + accept/reject + KV trim).
2. Modifies the algorithm. Full freedom: tree drafting, dynamic γ, retrieval-
   augmented drafts, novel acceptance rules, anything.
3. Runs `./task-eval` (in `evaluator/`).
4. Harness compiles the modified file (~2.5 sec rebuild — only this single
   `.cpp` recompiles), runs it on N prompts, parses tok/s and tokens, computes
   score.

The agent must keep the per-prompt stdin/stdout contract (read prompts split
on `\n<<END_PROMPT>>\n`, emit `=== PROMPT i BEGIN === ... === PROMPT i END ===`
blocks containing `tok/s`, `n_drafted`, `n_accept`, `TOKENS:` lines). The
canonical baseline they fork already implements this contract.

## Score formula

```
score = mean(agent_tok_per_sec) / mean(vanilla_tok_per_sec)
        IF anti-cheat gates pass, ELSE 0
```

Anti-cheat gates:

1. **Acceptance rate ∈ [5%, 90%].**
   - <5% means the draft model is producing nothing useful (no real speculation).
   - >90% suggests the agent bypassed the verify step (lossy / always-accept).
2. **LCS overlap with reference ≥ 50%.**
   - Catches lossy methods whose output content fundamentally diverges from
     what the target would produce. Permissive enough that legitimate
     algorithmic variants pass — small FP non-associativity drift is expected
     and tolerated.

See `docs/methodology.md` for the full discussion of why these thresholds.

## Expected baseline

The canonical baseline scores (your numbers will vary by ~5% per run on fast
tier, less on larger tiers):

- **Vanilla** (target only, no spec-dec): ~15.5 tok/s (1.0×, by definition)
- **Canonical Leviathan** (γ=8, single chain, strict greedy):
  - fast tier (40 prompts): ~2.57×
  - medium tier (80 prompts): ~2.52×
  - full tier (480 prompts): ~2.67×

These are what an agent's recipe needs to beat. Prior agent runs on this
benchmark have produced higher numbers; we deliberately don't publish the
specific recipes or per-category breakdowns here, so each new run is an
independent search of the design space.

## Directory layout

```
spec-dec-swarm-eval/
├── README.md
├── LICENSE
├── setup.sh                     # one-shot installer (clone llama.cpp, build, download models, gen refs)
├── task.yaml                    # for swarm framework integration
├── workspace/
│   └── initial_program.cpp      # canonical Leviathan baseline (the agent edits this)
├── evaluator/
│   ├── task-eval                # entry script (wraps eval.py)
│   ├── eval.py                  # scoring with anti-cheat gates
│   ├── prompts_{fast,medium,full}.jsonl    # pre-shipped prompt subsets
│   └── reference_{fast,medium,full}.jsonl  # pre-shipped vanilla tok/s + canonical naive-spec output tokens
├── scripts/
│   ├── generate_reference.py    # idempotent reference regen
│   ├── reset_workspace.sh       # restore initial_program.cpp to canonical baseline
│   └── extras/
│       ├── canonical_baseline.cpp           # cached canonical Leviathan source
│       ├── naive-spec/{naive-spec.cpp,CMakeLists.txt}    # added to llama.cpp/examples/
│       └── vanilla-bench/{vanilla-bench.cpp,CMakeLists.txt}
├── data/
│   └── spec_bench_questions.jsonl   # vendored from Spec-Bench (CC-BY)
└── docs/
    ├── methodology.md           # algorithm, baseline, scoring rationale, gate thresholds
    ├── framework_integration.md # notes for plugging this into a swarm framework
    └── build_llama_cpp.md       # manual fallback if setup.sh fails
```

## Common operations

**Run on a different GPU:**
```bash
CUDA_VISIBLE_DEVICES=1 ./task-eval --tier fast
```

**Use a different model pair:** the harness reads paths from environment
variables; defaults can be overridden without source edits.

| env var | default |
|---|---|
| `SPEC_DEC_ROOT` | `$HOME/spec-dec-work` |
| `LLAMA_DIR` | `$SPEC_DEC_ROOT/llama.cpp` |
| `SPEC_DEC_MODELS_DIR` | `$SPEC_DEC_ROOT/models` |
| `SPEC_DEC_TARGET` | `$SPEC_DEC_MODELS_DIR/llama3.3-70b/Llama-3.3-70B-Instruct-Q4_K_M.gguf` |
| `SPEC_DEC_DRAFT` | `$SPEC_DEC_MODELS_DIR/llama3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf` |
| `SPEC_DEC_QUESTIONS` | `<repo>/data/spec_bench_questions.jsonl` |
| `SPEC_DEC_TIER` | `fast` |

After swapping models you must regenerate references — token IDs depend on the
specific tokenizer/quantization. `./setup.sh --regen-refs fast` (or `medium`,
`full`).

**Reset the workspace to canonical baseline:**
```bash
./scripts/reset_workspace.sh
```

**Cleanup between agents:** the harness copies `workspace/initial_program.cpp`
into the llama.cpp source tree before each compile, so each eval starts from
whatever `initial_program.cpp` currently contains. No manual cleanup needed.

## Known caveats

1. **"Lossless" is WRT FP precision**, not bit-equal across implementations.
   `cuBLAS` matmul is non-associative; different verify-batch shapes (γ=4
   vs γ=8) produce slightly different logits, and on Q4-quantized models
   that crosses argmax tie boundaries often enough to diverge subsequent
   tokens. The LCS-overlap gate captures the practical correctness notion.
   See `docs/methodology.md`.
2. **Concurrent agents share one llama.cpp source tree** by default.
   The harness writes the agent's `.cpp` into `$LLAMA_DIR/examples/naive-spec/`
   before each build — running >1 agent against the same `LLAMA_DIR`
   simultaneously will race. Either serialize evals per `LLAMA_DIR` or give
   each concurrent agent its own `LLAMA_DIR` (override via the `LLAMA_DIR`
   env var). See `docs/framework_integration.md`.
3. **Full-tier eval (~32 min) fits comfortably in `task.yaml`'s default
   7200s timeout** — but you should still bump it externally if you've
   modified the algorithm to be slower than canonical, or your hardware
   is below the 48 GB / sm_86 reference target.
4. **Per-prompt failure rate ~10–15% on `full` tier** due to long-prompt
   timeouts. Tier `full` tolerates up to 10% failures by default.

## Citation

If you use this benchmark in academic work:

```bibtex
@software{spec_dec_swarm_eval,
  title  = {spec-dec-swarm-eval: Speculative Decoding Optimization Benchmark for Agent Swarms},
  author = {Edds, Zack},
  year   = {2026},
  url    = {https://github.com/zackedds/spec-dec-swarm-eval},
}
```

## License

MIT — see `LICENSE`. The vendored Spec-Bench prompts (`data/spec_bench_questions.jsonl`)
are from the Spec-Bench project and retain their original license.

## Acknowledgments

- Speculative decoding algorithm: Leviathan, Kalman, Matias (2022), arXiv:2211.17192.
- Prompt corpus: [Spec-Bench](https://github.com/hemingkx/Spec-Bench) (Xia et al.).
- Inference engine: [llama.cpp](https://github.com/ggml-org/llama.cpp).
