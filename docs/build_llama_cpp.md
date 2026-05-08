# Building llama.cpp from source (manual fallback)

The top-level `setup.sh` automates this whole process — clone, patch, build.
This doc is the manual fallback if `setup.sh` fails or you want to drive the
build yourself.

The harness expects a llama.cpp build at `$LLAMA_DIR/build/` (default
`$SPEC_DEC_ROOT/llama.cpp/build/`) with CUDA support and two custom example
targets (`llama-naive-spec`, `llama-vanilla-bench`).

## Prerequisites

- CUDA 12.x with `nvcc` and a matching driver. Verified on CUDA 12.9.
- gcc-11 host compiler (CUDA frontends reject gcc-12+).
- cmake 3.22+.
- An NVIDIA GPU with compute capability matching `CMAKE_CUDA_ARCHITECTURES`
  (default `86` for A6000 / RTX 30-series). For other GPUs:
  - A100: `80`
  - L4 / L40 / RTX 4090 / RTX 6000 Ada: `89`
  - H100 / H200: `90`

## Clone the pinned llama.cpp commit

```bash
LLAMA_SHA=<see-setup.sh-LLAMA_SHA>
mkdir -p "$SPEC_DEC_ROOT"
cd "$SPEC_DEC_ROOT"
git clone https://github.com/ggml-org/llama.cpp llama.cpp
cd llama.cpp
git checkout "$LLAMA_SHA"
```

## Add the two custom example dirs

Copy the persistent-process binaries from this repo into llama.cpp's example
tree:

```bash
cp -r <repo>/scripts/extras/naive-spec    examples/
cp -r <repo>/scripts/extras/vanilla-bench examples/
```

Append to `examples/CMakeLists.txt` (after the existing
`add_subdirectory(speculative)` line):

```
add_subdirectory(naive-spec)
add_subdirectory(vanilla-bench)
```

## (Optional) Patch for cross-tokenizer pairs

Only needed if you swap in a draft model from a different family than the
target (e.g., Qwen draft + Llama target). The default Llama-3.x pair shares
tokenizers and doesn't need this.

In `examples/speculative/speculative.cpp` (around the BOS/EOS check, ~line 119):

```cpp
// Replace:
LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
return 1;
// With:
LOG_WRN("%s: draft model special tokens differ from target — proceeding anyway\n", __func__);
// /* return 1; */
```

## Build

```bash
export PATH=/usr/local/cuda/bin:$PATH      # adjust to your CUDA install
export CUDACXX=$(which nvcc)
cmake -B build \
    -DGGML_CUDA=ON \
    -DLLAMA_CURL=OFF \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-11 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DGGML_NATIVE=ON
cmake --build build --config Release -j 8 \
    --target llama-cli llama-speculative llama-bench llama-naive-spec llama-vanilla-bench
```

Initial build takes ~5–7 minutes (CUDA-heavy). Subsequent rebuilds during
agent iteration touch only the changed `.cpp` and finish in ~2.5 seconds.

## Verify

```bash
ls -la build/bin/{llama-cli,llama-speculative,llama-naive-spec,llama-vanilla-bench}
```

All four binaries should be present.

Smoke test (replace `$TARGET_GGUF` with your downloaded target weights):

```bash
export CUDA_VISIBLE_DEVICES=0
printf 'Hello world\n<<END_PROMPT>>\n' | ./build/bin/llama-vanilla-bench \
    -m "$TARGET_GGUF" -ngl 99 -c 2048 -n 32 --temp 0 \
    2>&1 | grep -E "PROMPT 0|TOKENS:|tok/s"
```

Should output a `=== PROMPT 0 BEGIN ===` block, a `TOKENS:` line, and ~15 tok/s.

## Disk usage

The build tree is ~12 GB total (object files + kernels for one CUDA arch).
Don't clean it between agent iterations — only the modified example `.cpp`
recompiles each time.

## Multi-arch builds

Pass multiple values: `-DCMAKE_CUDA_ARCHITECTURES="86;90"`. Compile time
scales linearly with arch count, so only do this if you actually need to run
on multiple GPU generations from one build.
