#!/usr/bin/env bash
# spec-dec-swarm-eval — one-shot installer.
#
# Idempotent: safe to re-run; only does work that's missing.
#
# Steps:
#   1. Verify host prerequisites (CUDA, gcc, cmake, python, huggingface-cli)
#   2. Clone llama.cpp at the pinned SHA (if not already present)
#   3. Copy our extra example dirs (naive-spec, vanilla-bench) into the tree
#      and patch examples/CMakeLists.txt to include them
#   4. Build llama.cpp + the two custom targets with CUDA
#   5. Download Llama-3.3-70B-Instruct-Q4_K_M and Llama-3.2-1B-Instruct-Q4_K_M
#      from HuggingFace; verify SHA-256 against the values these references
#      were generated against
#   6. (Optional) Regenerate reference data for the requested tier(s)
#
# Usage:
#   ./setup.sh                          # install + build + download
#   ./setup.sh --regen-refs fast        # also regenerate fast-tier references
#   ./setup.sh --regen-refs all         # regenerate all three tiers
#
# Environment variables (override defaults):
#   SPEC_DEC_ROOT          default $HOME/spec-dec-work
#   LLAMA_DIR              default $SPEC_DEC_ROOT/llama.cpp
#   SPEC_DEC_MODELS_DIR    default $SPEC_DEC_ROOT/models
#   CUDA_ARCH              default 86 (A6000); pass 80/89/90 for A100/L40/H100
#   CUDA_HOME              default /usr/local/cuda
#   GCC_HOST, GXX_HOST     default /usr/bin/gcc-11, /usr/bin/g++-11
#   FORCE_REBUILD=1        force re-cmake + rebuild even if binaries exist

set -euo pipefail

PKG="$(cd "$(dirname "$0")" && pwd)"
SPEC_DEC_ROOT="${SPEC_DEC_ROOT:-$HOME/spec-dec-work}"
LLAMA_DIR="${LLAMA_DIR:-$SPEC_DEC_ROOT/llama.cpp}"
MODELS_DIR="${SPEC_DEC_MODELS_DIR:-$SPEC_DEC_ROOT/models}"
CUDA_ARCH="${CUDA_ARCH:-86}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
GCC_HOST="${GCC_HOST:-/usr/bin/gcc-11}"
GXX_HOST="${GXX_HOST:-/usr/bin/g++-11}"

# Pinned llama.cpp commit. Verified-working SHA at time of release.
LLAMA_SHA="f53577432541bb9edc1588c4ef45c66bf07e4468"

# HuggingFace model IDs (both Q4_K_M GGUFs from bartowski's mirrors).
TGT_HF_REPO="bartowski/Llama-3.3-70B-Instruct-GGUF"
TGT_HF_FILE="Llama-3.3-70B-Instruct-Q4_K_M.gguf"
TGT_LOCAL_DIR="$MODELS_DIR/llama3.3-70b"
TGT_LOCAL_PATH="$TGT_LOCAL_DIR/$TGT_HF_FILE"
TGT_SHA256="32df3baccb556f9840059b2528b2dee4d3d516b24afdfb9d0c56ff5f63e3a664"

DFT_HF_REPO="bartowski/Llama-3.2-1B-Instruct-GGUF"
DFT_HF_FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DFT_LOCAL_DIR="$MODELS_DIR/llama3.2-1b"
DFT_LOCAL_PATH="$DFT_LOCAL_DIR/$DFT_HF_FILE"
DFT_SHA256="6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83"

# Parse args.
REGEN_TIERS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --regen-refs) REGEN_TIERS="$2"; shift 2 ;;
        --regen-refs=*) REGEN_TIERS="${1#*=}"; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [ "$REGEN_TIERS" = "all" ]; then REGEN_TIERS="fast medium full"; fi

echo "=========================================="
echo "  spec-dec-swarm-eval setup"
echo "=========================================="
echo "  package:    $PKG"
echo "  root:       $SPEC_DEC_ROOT"
echo "  llama.cpp:  $LLAMA_DIR (pinned to ${LLAMA_SHA:0:10})"
echo "  models:     $MODELS_DIR"
echo "  cuda arch:  sm_$CUDA_ARCH"
echo

mkdir -p "$SPEC_DEC_ROOT" "$MODELS_DIR"

# ---------------------------------------------------------------------------
# [1/6] Prerequisites
# ---------------------------------------------------------------------------
echo "[1/6] Checking prerequisites..."

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "  MISSING: command '$1' not found"
        echo "  install hint: $2"
        exit 1
    fi
    echo "  OK: $1"
}

require_cmd git "apt install git || dnf install git"
require_cmd cmake "apt install cmake (>= 3.22)"
require_cmd python3 "apt install python3.10"

if [ ! -x "$GCC_HOST" ]; then
    echo "  MISSING: $GCC_HOST"
    echo "  install: apt install gcc-11 g++-11   (CUDA frontends reject gcc-12+)"
    exit 1
fi
echo "  OK: $GCC_HOST"
[ -x "$GXX_HOST" ] || { echo "  MISSING: $GXX_HOST"; exit 1; }
echo "  OK: $GXX_HOST"

if [ ! -x "$CUDA_HOME/bin/nvcc" ]; then
    echo "  MISSING: nvcc at $CUDA_HOME/bin/nvcc"
    echo "  set CUDA_HOME=/path/to/cuda before re-running, or install CUDA 12.x"
    exit 1
fi
echo "  OK: nvcc ($("$CUDA_HOME/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p'))"

# Disk pre-flight: ~60 GB needed (42 GB target + 1 GB draft + 12 GB build + headroom).
NEED_GB=60
AVAIL_GB="$(df -BG "$SPEC_DEC_ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}' 2>/dev/null || echo 0)"
if [ "$AVAIL_GB" -lt "$NEED_GB" ] 2>/dev/null; then
    echo "  WARNING: only ${AVAIL_GB} GB free at $SPEC_DEC_ROOT (need ~${NEED_GB} GB for models + build)"
    echo "  Free up disk or set SPEC_DEC_ROOT/SPEC_DEC_MODELS_DIR/LLAMA_DIR to a larger volume."
    if [ "$AVAIL_GB" -lt 50 ] 2>/dev/null; then
        echo "  Aborting: <50 GB is too tight to even partial-download the target model." >&2
        exit 1
    fi
    echo "  Proceeding anyway, but the 42 GB model download may fail mid-stream."
else
    echo "  OK: ${AVAIL_GB} GB free at $SPEC_DEC_ROOT (need ~${NEED_GB} GB)"
fi

# huggingface-cli is the recommended downloader; the newer 'hf' alias also works.
HF=""
if command -v huggingface-cli >/dev/null 2>&1; then
    HF="huggingface-cli"
elif command -v hf >/dev/null 2>&1; then
    HF="hf"
else
    echo "  MISSING: huggingface-cli (or 'hf') for model download"
    echo "  install: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi
echo "  OK: $HF"

# ---------------------------------------------------------------------------
# [2/6] Clone llama.cpp at pinned SHA
# ---------------------------------------------------------------------------
echo
echo "[2/6] Setting up llama.cpp..."
if [ -d "$LLAMA_DIR/.git" ]; then
    cur_sha="$(git -C "$LLAMA_DIR" rev-parse HEAD)"
    if [ "$cur_sha" = "$LLAMA_SHA" ]; then
        echo "  OK: $LLAMA_DIR already at pinned SHA"
    else
        echo "  llama.cpp present but at SHA ${cur_sha:0:10} (want ${LLAMA_SHA:0:10})"
        echo "  fetching + checking out pinned SHA..."
        git -C "$LLAMA_DIR" fetch --quiet origin "$LLAMA_SHA" || \
            git -C "$LLAMA_DIR" fetch --quiet origin
        git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_SHA"
        echo "  OK: checked out ${LLAMA_SHA:0:10}"
    fi
else
    echo "  cloning llama.cpp..."
    git clone --quiet https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    git -C "$LLAMA_DIR" checkout --quiet "$LLAMA_SHA"
    echo "  OK: cloned + checked out ${LLAMA_SHA:0:10}"
fi

# ---------------------------------------------------------------------------
# [3/6] Copy custom example dirs + patch examples/CMakeLists.txt (idempotent)
# ---------------------------------------------------------------------------
echo
echo "[3/6] Adding custom example targets..."
for ex in naive-spec vanilla-bench; do
    src="$PKG/scripts/extras/$ex"
    dst="$LLAMA_DIR/examples/$ex"
    if [ ! -d "$src" ]; then
        echo "  MISSING source: $src" >&2; exit 1
    fi
    mkdir -p "$dst"
    cp -f "$src"/CMakeLists.txt "$dst/"
    cp -f "$src"/*.cpp "$dst/"
    echo "  OK: $dst"
done

CMK="$LLAMA_DIR/examples/CMakeLists.txt"
for ex in naive-spec vanilla-bench; do
    if grep -q "add_subdirectory($ex)" "$CMK"; then
        echo "  CMakeLists.txt already includes $ex"
    else
        if grep -q "add_subdirectory(speculative)" "$CMK"; then
            sed -i "/add_subdirectory(speculative)/a add_subdirectory($ex)" "$CMK"
        else
            printf '\nadd_subdirectory(%s)\n' "$ex" >> "$CMK"
        fi
        echo "  patched CMakeLists.txt: add_subdirectory($ex)"
    fi
done

# ---------------------------------------------------------------------------
# [4/6] Build with CUDA
# ---------------------------------------------------------------------------
echo
echo "[4/6] Building llama.cpp with CUDA (sm_$CUDA_ARCH)..."

NEEDED_BINS="llama-cli llama-naive-spec llama-vanilla-bench"
all_present=1
for b in $NEEDED_BINS; do
    [ -x "$LLAMA_DIR/build/bin/$b" ] || all_present=0
done

if [ "$all_present" = "1" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
    echo "  OK: all needed binaries already present (set FORCE_REBUILD=1 to rebuild)"
else
    export PATH="$CUDA_HOME/bin:$PATH"
    export CUDACXX="$CUDA_HOME/bin/nvcc"
    if [ ! -d "$LLAMA_DIR/build" ] || [ "${FORCE_REBUILD:-0}" = "1" ]; then
        echo "  configuring cmake..."
        cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
            -DGGML_CUDA=ON \
            -DLLAMA_CURL=OFF \
            -DCMAKE_CUDA_HOST_COMPILER="$GCC_HOST" \
            -DCMAKE_C_COMPILER="$GCC_HOST" \
            -DCMAKE_CXX_COMPILER="$GXX_HOST" \
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
            -DGGML_NATIVE=ON 2>&1 | tail -10
    fi
    echo "  building (this is the slow step — ~5-7 min cold)..."
    cmake --build "$LLAMA_DIR/build" --config Release -j 8 \
        --target llama-cli llama-naive-spec llama-vanilla-bench 2>&1 | tail -15
    for b in $NEEDED_BINS; do
        [ -x "$LLAMA_DIR/build/bin/$b" ] || { echo "  FAILED: $b not produced" >&2; exit 1; }
    done
    echo "  OK: built $NEEDED_BINS"
fi

# ---------------------------------------------------------------------------
# [5/6] Download + verify model files
# ---------------------------------------------------------------------------
echo
echo "[5/6] Setting up model files..."

verify_sha() {
    # $1 file path, $2 expected sha256
    local got
    got="$(sha256sum "$1" | awk '{print $1}')"
    if [ "$got" != "$2" ]; then
        echo "  SHA mismatch on $(basename "$1")" >&2
        echo "    expected: $2" >&2
        echo "    got:      $got" >&2
        return 1
    fi
    return 0
}

download_one() {
    local repo="$1" file="$2" dir="$3" path="$4" sha="$5"
    mkdir -p "$dir"
    if [ -f "$path" ]; then
        echo "  $file already present, verifying SHA..."
        if verify_sha "$path" "$sha"; then
            echo "  OK: $file"
            return 0
        else
            echo "  re-downloading $file..."
            rm -f "$path"
        fi
    fi
    echo "  downloading $repo / $file → $dir ..."
    "$HF" download "$repo" "$file" --local-dir "$dir" >/dev/null
    [ -f "$path" ] || { echo "  download failed: $path missing" >&2; exit 1; }
    if verify_sha "$path" "$sha"; then
        echo "  OK: $file ($(du -sh "$path" | cut -f1))"
    else
        echo "  WARNING: downloaded copy has a different SHA than the one our" >&2
        echo "  shipped reference data was generated against. You should" >&2
        echo "  regenerate references before scoring:" >&2
        echo "    ./setup.sh --regen-refs all" >&2
    fi
}

download_one "$TGT_HF_REPO" "$TGT_HF_FILE" "$TGT_LOCAL_DIR" "$TGT_LOCAL_PATH" "$TGT_SHA256"
download_one "$DFT_HF_REPO" "$DFT_HF_FILE" "$DFT_LOCAL_DIR" "$DFT_LOCAL_PATH" "$DFT_SHA256"

# ---------------------------------------------------------------------------
# [6/6] Reference data
# ---------------------------------------------------------------------------
echo
echo "[6/6] Reference data..."
for tier in fast medium full; do
    p="$PKG/evaluator/prompts_$tier.jsonl"
    r="$PKG/evaluator/reference_$tier.jsonl"
    if [ -f "$p" ] && [ -f "$r" ]; then
        echo "  OK: $tier (shipped: $(wc -l < "$r") rows)"
    else
        echo "  MISSING: $tier — run ./setup.sh --regen-refs $tier"
    fi
done

if [ -n "$REGEN_TIERS" ]; then
    echo
    echo "Regenerating reference data for: $REGEN_TIERS"
    for tier in $REGEN_TIERS; do
        case "$tier" in
            fast)   N=40  ;;
            medium) N=80  ;;
            full)   N=480 ;;
            *) echo "  unknown tier: $tier" >&2; continue ;;
        esac
        echo "  --regen $tier (N=$N)..."
        SPEC_DEC_ROOT="$SPEC_DEC_ROOT" LLAMA_DIR="$LLAMA_DIR" \
        SPEC_DEC_MODELS_DIR="$MODELS_DIR" \
        SPEC_DEC_TARGET="$TGT_LOCAL_PATH" SPEC_DEC_DRAFT="$DFT_LOCAL_PATH" \
        SPEC_DEC_QUESTIONS="$PKG/data/spec_bench_questions.jsonl" \
            python3 "$PKG/scripts/generate_reference.py" --tier "$tier" --n-prompts "$N"
    done
fi

echo
echo "=========================================="
echo "  setup complete"
echo "=========================================="
echo "Try it:"
echo "  cd $PKG/evaluator"
echo "  CUDA_VISIBLE_DEVICES=0 SPEC_DEC_ROOT=$SPEC_DEC_ROOT ./task-eval --tier fast"
echo
