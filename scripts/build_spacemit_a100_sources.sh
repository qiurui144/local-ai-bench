#!/usr/bin/env bash
set -euo pipefail

# Build SpacemiT A100-targeted llama.cpp and ONNX Runtime from source.
# Run this on an x86_64 build host with the SpacemiT RISC-V cross toolchain.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOOLCHAIN_ROOT="${SPACEMIT_TOOLCHAIN_ROOT:-}"
if [[ -z "${TOOLCHAIN_ROOT}" ]]; then
  if [[ -d "${REPO_ROOT}/drivers/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.2.4" ]]; then
    TOOLCHAIN_ROOT="${REPO_ROOT}/drivers/toolchains/spacemit-toolchain-linux-glibc-x86_64-v1.2.4"
  else
    TOOLCHAIN_ROOT="/data/RV/rv-spacemit-toolchain/spacemit-toolchain-linux-glibc-x86_64-v1.2.2"
  fi
fi

SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/drivers/spacemit-source}"
LLAMA_SRC="${LLAMA_SRC:-${SOURCE_ROOT}/llama.cpp}"
ORT_SRC="${ORT_SRC:-${SOURCE_ROOT}/onnxruntime}"
BUILD_ROOT="${BUILD_ROOT:-${REPO_ROOT}/builds/spacemit-a100}"
BUILD_LLAMA="${BUILD_LLAMA:-1}"
BUILD_ORT="${BUILD_ORT:-0}"
ORT_DEPS_MIRROR="${ORT_DEPS_MIRROR:-${REPO_ROOT}/drivers/ort-cmake-deps-mirror}"
PARALLEL="${PARALLEL:-$(nproc)}"
ORT_PARALLEL="${ORT_PARALLEL:-20}"
ORT_CLEAN="${ORT_CLEAN:-1}"
ORT_A100_FLAGS="${ORT_A100_FLAGS:--march=rv64gcv_zfh_zvfh_zba_zicbop_zihintpause_xsmtvdotii -mabi=lp64d -fno-tree-vectorize -fno-tree-loop-vectorize}"
ORT_TARGETS="${ORT_TARGETS:-onnxruntime onnxruntime_perf_test onnxruntime_mlas_benchmark}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_toolchain() {
  test -x "${TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-gcc" || {
    echo "missing SpacemiT RISC-V GCC under TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT}" >&2
    exit 2
  }
}

clone_sources_if_needed() {
  mkdir -p "${SOURCE_ROOT}"
  if [[ ! -d "${LLAMA_SRC}/.git" ]]; then
    git clone https://github.com/spacemit-com/llama.cpp "${LLAMA_SRC}"
  fi
  if [[ ! -d "${ORT_SRC}/.git" ]]; then
    git clone https://github.com/spacemit-com/onnxruntime "${ORT_SRC}"
  fi
}

build_llama() {
  local build_dir="${BUILD_ROOT}/llama-src"
  local install_dir="${BUILD_ROOT}/llama-install"
  log "configure llama.cpp A100 build"
  RISCV_ROOT_PATH="${TOOLCHAIN_ROOT}" cmake -S "${LLAMA_SRC}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CPU_RISCV64_SPACEMIT=ON \
    -DGGML_CPU_REPACK=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_APP=OFF \
    -DGGML_RVV=ON \
    -DGGML_RV_ZVFH=ON \
    -DGGML_RV_ZFH=ON \
    -DGGML_RV_ZICBOP=ON \
    -DGGML_RV_ZIHINTPAUSE=ON \
    -DGGML_RV_ZBA=ON \
    -DCMAKE_TOOLCHAIN_FILE="${LLAMA_SRC}/cmake/riscv64-spacemit-linux-gnu-gcc.cmake" \
    -DCMAKE_INSTALL_PREFIX="${install_dir}"
  log "build llama.cpp"
  cmake --build "${build_dir}" --parallel "${PARALLEL}" --config Release
  cmake --install "${build_dir}"
  tar -C "${install_dir}" -czf "${BUILD_ROOT}/llama-install.tar.gz" .
  log "llama.cpp install: ${install_dir}"
  log "llama.cpp tar: ${BUILD_ROOT}/llama-install.tar.gz"
}

build_ort() {
  local build_dir="${BUILD_ROOT}/onnxruntime-src"
  local install_dir="${BUILD_ROOT}/onnxruntime-install"
  local -a target_args=()
  local -a ort_targets=()
  if [[ -n "${ORT_TARGETS}" ]]; then
    read -r -a ort_targets <<< "${ORT_TARGETS}"
    target_args=(--targets "${ort_targets[@]}")
  fi
  log "cache ORT CMake deps mirror"
  python3 "${REPO_ROOT}/scripts/cache_spacemit_ort_deps.py" --mirror "${ORT_DEPS_MIRROR}" --download-missing
  if [[ "${ORT_CLEAN}" == "1" ]]; then
    rm -rf "${build_dir}/Release" "${install_dir}"
  fi
  log "build ONNX Runtime"
  (
    cd "${ORT_SRC}"
    RISCV_ROOT_PATH="${TOOLCHAIN_ROOT}" python3 tools/ci_build/build.py \
      --build_dir "${build_dir}" \
      --config Release \
      --update \
      --build \
      --build_shared_lib \
      --parallel "${ORT_PARALLEL}" \
      --compile_no_warning_as_error \
      --allow_running_as_root \
      --build_micro_benchmarks \
      --rv64 \
      --riscv_toolchain_root="${TOOLCHAIN_ROOT}" \
      --skip_submodule_sync \
      --use_mimalloc \
      --skip_tests \
      "${target_args[@]}" \
      --cmake_deps_mirror_dir "${ORT_DEPS_MIRROR}" \
      --cmake_extra_defines \
        onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON \
        CMAKE_C_FLAGS="${ORT_A100_FLAGS}" \
        CMAKE_CXX_FLAGS="${ORT_A100_FLAGS}" \
        CMAKE_ASM_FLAGS="${ORT_A100_FLAGS}" \
        SPACEMIT_RISCV_MLAS_FLAGS="${ORT_A100_FLAGS}" \
        CMAKE_INSTALL_PREFIX="${install_dir}"
    cmake --install "${build_dir}/Release"
  )
  tar -C "${install_dir}" -czf "${BUILD_ROOT}/onnxruntime-install.tar.gz" .
  log "ONNX Runtime install: ${install_dir}"
  log "ONNX Runtime tar: ${BUILD_ROOT}/onnxruntime-install.tar.gz"
}

require_toolchain
clone_sources_if_needed

mkdir -p "${BUILD_ROOT}"
log "toolchain: ${TOOLCHAIN_ROOT}"
if [[ "${BUILD_LLAMA}" == "1" ]]; then
  build_llama
fi
if [[ "${BUILD_ORT}" == "1" ]]; then
  build_ort
fi
