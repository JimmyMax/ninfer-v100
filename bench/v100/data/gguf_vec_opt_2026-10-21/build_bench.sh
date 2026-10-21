#!/bin/bash
# Fast rebuild of the GGUF GEMV harness: only the harness + stubs recompile; bridge/vec objects are cached.
set -e
cd /build
FLAGS="-O3 -arch=sm_70 -std=c++20 -I/src/src -I/src -I/src/src/ops/linear/gguf -I/src/third_party/ggml-quants/include -I/src/third_party/ggml-quants/src"
nvcc $FLAGS -c bench_gguf_gemv.cpp -o benchobjs/harness.o
nvcc $FLAGS -c bench_gguf_stubs.cu -o benchobjs/stubs.o
nvcc -o bench_gguf_gemv benchobjs/*.o -lcublas
echo built
