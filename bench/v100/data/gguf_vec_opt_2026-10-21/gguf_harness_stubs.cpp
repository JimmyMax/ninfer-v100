// Link stubs for the harness: the MoE mmq entry points are never called from vector_product.
#include "ggml_bridge_internal.cuh"

namespace ninfer::ops::gguf::detail {

template <ggml_type type>
std::size_t matrix_fixup_bytes_impl(int, int) {
    return 0;
}

template <ggml_type type>
bool moe_matrix_fits_impl(int, int, bool) {
    return false;
}

template <ggml_type type>
void moe_matrix_product_impl(const void* const*, std::int64_t, int, int, const MoeRouting&, int,
                             int, bool, const void*, float*, cudaStream_t) {
    std::abort();
}

template <ggml_type type>
void matrix_product_impl(const void*, std::int64_t, int, int, const void*, int, float*,
                         std::int64_t, void*, cudaStream_t, bool) {
    std::abort();
}

#define NINFER_HARNESS_INST(t) \
    template std::size_t matrix_fixup_bytes_impl<t>(int, int); \
    template bool moe_matrix_fits_impl<t>(int, int, bool); \
    template void moe_matrix_product_impl<t>(const void* const*, std::int64_t, int, int, \
                                             const MoeRouting&, int, int, bool, const void*, \
                                             float*, cudaStream_t); \
    template void matrix_product_impl<t>(const void*, std::int64_t, int, int, const void*, \
                                         int, float*, std::int64_t, void*, cudaStream_t, bool);

NINFER_HARNESS_INST(GGML_TYPE_Q4_0)
NINFER_HARNESS_INST(GGML_TYPE_Q5_0)
NINFER_HARNESS_INST(GGML_TYPE_Q8_0)
NINFER_HARNESS_INST(GGML_TYPE_Q2_K)
NINFER_HARNESS_INST(GGML_TYPE_Q3_K)
NINFER_HARNESS_INST(GGML_TYPE_Q4_K)
NINFER_HARNESS_INST(GGML_TYPE_Q5_K)
NINFER_HARNESS_INST(GGML_TYPE_Q6_K)
NINFER_HARNESS_INST(GGML_TYPE_IQ2_XXS)
NINFER_HARNESS_INST(GGML_TYPE_IQ2_XS)
NINFER_HARNESS_INST(GGML_TYPE_IQ3_XXS)
NINFER_HARNESS_INST(GGML_TYPE_IQ1_S)
NINFER_HARNESS_INST(GGML_TYPE_IQ4_NL)
NINFER_HARNESS_INST(GGML_TYPE_IQ3_S)
NINFER_HARNESS_INST(GGML_TYPE_IQ2_S)
NINFER_HARNESS_INST(GGML_TYPE_IQ4_XS)
NINFER_HARNESS_INST(GGML_TYPE_IQ1_M)
NINFER_HARNESS_INST(GGML_TYPE_Q2_0)

}  // namespace ninfer::ops::gguf::detail
