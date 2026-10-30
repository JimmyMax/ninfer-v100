# NInfer ggml block-quant sources

These files are copied from [llama.cpp](https://github.com/ggml-org/llama.cpp),
commit [`a25c9865fe03c954c93fd755b5d79ae86ba99750`](https://github.com/ggml-org/llama.cpp/commit/a25c9865fe03c954c93fd755b5d79ae86ba99750),
`ggml/include/` and `ggml/src/`, under the MIT license; see [LICENSE](LICENSE). Changes are
marked `NInfer:` in the source. Q4_0's DP4A matrix product (`mmq-vec-dot.cuh`, the route of GPUs
without integer tensor cores such as Tesla V100) takes its -8 offset term from the quantized
activation instead of the block's FP16 sum, which overflows once 32 activations sum past 65504.

NInfer compiles ggml's integer tensor-core matrix multiplication (`ggml-cuda/mmq.cuh`) and its
dequantizers through the bridge in `src/ops/linear/gguf/`, as a separate archive, and keeps only the
headers that code includes. Products of up to eight tokens use NInfer's own kernel
(`ggml_bridge_vec.cuh`), which reads the grids and block structs from `ggml-common.h`.

The flash-attention kernel (`ggml-cuda/fattn-mma-f16.cuh` with `fattn-common.cuh` and
`cp-async.cuh`, same base commit) runs Gemma 4's prompt attention (head widths 256 and 512)
through `src/ops/linear/gguf/ggml_fattn.cu`, which launches it without ggml's backend context.
The combine-loop barrier in `fattn-mma-f16.cuh` includes the exact condition from upstream
[`1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3`](https://github.com/ggml-org/llama.cpp/commit/1bb2b9fcbe85c35c4e3b787e1c868ef05ca702f3)
(PR #30103). It synchronizes before reusing the shared combine tile when
`nbatch_combine != DKQ/2`, including the Turing 512-wide 16/32-column configurations.

The CPU experts' fast arithmetic (`--expert-cpu-arithmetic ggml`, Flash-Next host or disk
experts computed on the CPU) compiles ggml's `ggml-quants.c` (row quantizers and dequantizers),
ggml-cpu's scalar `quants.c` (generic vecdot fallbacks) and, on x86-64, its AVX2
`arch/x86/quants.c` (`ggml_vec_dot_iq*_q8_K` and `_q8_0`, the kernels llama.cpp itself runs for
these block types), as C11. `src/ggml-cpu/quants_host.c` fills the FP16/E8M0/UE4M3 lookup tables
upstream keeps in ggml-cpu.c, and `src/ops/moe_experts/gguf_expert_cpu_ggml.cpp` drives the
kernels: it quantizes the BF16 activations to q8_K (the i-quant gate and up projections) or
q8_0 (the IQ4_NL and Q8_0 projections) with ggml's reference quantizers, and runs one vecdot
per weight row and token. The default (`exact`) CPU arithmetic stays NInfer's own bit-for-bit
q8_1 products.
