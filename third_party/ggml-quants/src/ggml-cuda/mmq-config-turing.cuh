// Turing (sm_75) MMQ launch configs.
//
// Turing carries 1024 threads and 64 KiB of shared memory per SM (half of Ampere's
// threads, same shared memory). The Ampere table's I=128 tile therefore fits exactly
// one block per SM: 255 registers/thread (at the allocation cap) and ~58 KiB of
// dynamic shared memory leave 25% occupancy and a spill-heavy register file
// (measured on an RTX 2080 Ti with ncu: L1/TEX 70%, SM 42%).
//
// The mma vec-dot path hard-codes the warp-to-row mapping
//     I == (nwarps/ntx) * rows_per_warp,  ntx = rows_per_warp/16,  rows_per_warp = J >= 48 ? 32 : 16
// so with 256 threads (8 warps) I must stay 128. Halving I to 64 therefore goes
// together with halving the block to 128 threads (4 warps): ntx = 2, nty = 2,
// I = 2*32 = 64. That halves both the accumulators per thread and the shared
// memory tile, so two blocks fit per SM (16 warps, 50% occupancy). J = 128 tiles
// cannot satisfy the invariant at any block size that also halves the footprint
// (nthreads=256 forces I=128), so they are disabled and the launcher falls back
// to J = 64.

static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_turing(ggml_type type, int J, bool fallback) {
    ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    if (config.type == GGML_TYPE_COUNT) {
        return config;
    }
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_M:
            break;
        default:
            return config;
    }
    if (J > 64) {
        config.type = GGML_TYPE_COUNT;
        return config;
    }
    config.I        = 64;
    config.nthreads = 128;
    return config;
}
