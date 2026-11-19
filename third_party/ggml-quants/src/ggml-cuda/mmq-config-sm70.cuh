// sm_70 (Volta / Tesla V100) MMQ tile configuration.
//
// The V100 has 80 SMs and 96 KiB opt-in shared memory, like Turing, but its INT8 tensor
// throughput is ~1/8 of an A100's, so the ampere table's (nthreads=256, I=128, stream_k)
// defaults — tuned on Ampere/Hopper — are not necessarily optimal here. This table overrides
// the block formats the deployed GGUF artifacts use (gguf_q4_k, gguf_q6_k, gguf_iq3_s);
// every other type falls through to the ampere table.
//
// Fields: CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback).
// Tuned with tools/mmq_sm70_tune_probe.cu on the V100; record results in the commit message.

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_sm70(ggml_type type, int J, bool fallback) {
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q4_K, 512, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_Q6_K, 512, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_IQ3_S, 512, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, true, false);

    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}
