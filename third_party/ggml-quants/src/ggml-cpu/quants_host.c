// NInfer: host-side support for the vendored ggml quantization translation units
// (llama.cpp a25c9865, MIT).
//
// The vendored C sources reference five helpers that upstream defines in ggml.c and ggml-cpu.c,
// which NInfer does not vendor. None of the call sites are reachable from NInfer (the enum
// dispatch and row-data validation of ggml-quants.c), but the symbols must exist:
//   - ggml_type_size / ggml_blck_size / ggml_type_name / ggml_row_size: type metadata,
//     implemented here straight from the vendored block types and QK constants;
//   - ggml_table_f32_f16 / ggml_table_f32_e8m0_half / ggml_table_f32_ue4m3: the FP16/E8M0/UE4M3
//     bit pattern to FP32 value lookup tables upstream fills in ggml_cpu_init(), filled here at
//     load, before any vecdot can read them (the AVX2 Q8_K and Q8_0 kernels convert their
//     scales with SIMD, but the generic fallbacks and the other vecdot types look them up).
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

float ggml_table_f32_f16[1 << 16];
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_ue4m3[1 << 8];

#if defined(_MSC_VER)
#pragma section(".CRT$X9", read)
__declspec(allocate(".CRT$X9"))
#else
__attribute__((constructor))
#endif
static void ninfer_ggml_init_quants_tables(void) {
    for (int i = 0; i < (1 << 16); ++i) { ggml_table_f32_f16[i] = GGML_FP16_TO_FP32((uint16_t) i); }
    for (int i = 0; i < (1 << 8); ++i) {
        ggml_table_f32_e8m0_half[i] = ggml_e8m0_to_fp32((uint8_t) i);
        ggml_table_f32_ue4m3[i]     = ggml_ue4m3_to_fp32((uint8_t) i);
    }
}

int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_I8: case GGML_TYPE_I16:
    case GGML_TYPE_I32: case GGML_TYPE_I64: case GGML_TYPE_F64: case GGML_TYPE_BF16:
        return 1;
    case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0: case GGML_TYPE_Q8_1: case GGML_TYPE_IQ4_NL:
        return QK8_0;
    case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_K: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
    case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ1_S: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ2_S:
    case GGML_TYPE_IQ4_XS: case GGML_TYPE_IQ1_M: case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0:
        return QK_K;
    case GGML_TYPE_MXFP4:
        return QK_MXFP4;
    case GGML_TYPE_NVFP4:
        return QK_NVFP4;
    case GGML_TYPE_Q1_0:
        return QK1_0;
    case GGML_TYPE_Q2_0:
        return QK2_0;
    default:
        return 1;
    }
}

size_t ggml_type_size(enum ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:  return sizeof(float);
    case GGML_TYPE_F16:  return sizeof(ggml_half);
    case GGML_TYPE_Q4_0: return sizeof(block_q4_0);
    case GGML_TYPE_Q4_1: return sizeof(block_q4_1);
    case GGML_TYPE_Q5_0: return sizeof(block_q5_0);
    case GGML_TYPE_Q5_1: return sizeof(block_q5_1);
    case GGML_TYPE_Q8_0: return sizeof(block_q8_0);
    case GGML_TYPE_Q8_1: return sizeof(block_q8_1);
    case GGML_TYPE_Q2_K: return sizeof(block_q2_K);
    case GGML_TYPE_Q3_K: return sizeof(block_q3_K);
    case GGML_TYPE_Q4_K: return sizeof(block_q4_K);
    case GGML_TYPE_Q5_K: return sizeof(block_q5_K);
    case GGML_TYPE_Q6_K: return sizeof(block_q6_K);
    case GGML_TYPE_Q8_K: return sizeof(block_q8_K);
    case GGML_TYPE_IQ2_XXS: return sizeof(block_iq2_xxs);
    case GGML_TYPE_IQ2_XS:  return sizeof(block_iq2_xs);
    case GGML_TYPE_IQ3_XXS: return sizeof(block_iq3_xxs);
    case GGML_TYPE_IQ1_S:   return sizeof(block_iq1_s);
    case GGML_TYPE_IQ4_NL:  return sizeof(block_iq4_nl);
    case GGML_TYPE_IQ3_S:   return sizeof(block_iq3_s);
    case GGML_TYPE_IQ2_S:   return sizeof(block_iq2_s);
    case GGML_TYPE_IQ4_XS:  return sizeof(block_iq4_xs);
    case GGML_TYPE_I8:      return sizeof(int8_t);
    case GGML_TYPE_I16:     return sizeof(int16_t);
    case GGML_TYPE_I32:     return sizeof(int32_t);
    case GGML_TYPE_I64:     return sizeof(int64_t);
    case GGML_TYPE_F64:     return sizeof(double);
    case GGML_TYPE_IQ1_M:   return sizeof(block_iq1_m);
    case GGML_TYPE_BF16:    return sizeof(ggml_bf16_t);
    case GGML_TYPE_TQ1_0:   return sizeof(block_tq1_0);
    case GGML_TYPE_TQ2_0:   return sizeof(block_tq2_0);
    case GGML_TYPE_MXFP4:   return sizeof(block_mxfp4);
    case GGML_TYPE_NVFP4:   return sizeof(block_nvfp4);
    case GGML_TYPE_Q1_0:    return sizeof(block_q1_0);
    case GGML_TYPE_Q2_0:    return sizeof(block_q2_0);
    default:                return 0;
    }
}

const char* ggml_type_name(enum ggml_type type) {
    switch (type) {
    case GGML_TYPE_F32:  return "F32";
    case GGML_TYPE_F16:  return "F16";
    case GGML_TYPE_Q4_0: return "Q4_0";
    case GGML_TYPE_Q4_1: return "Q4_1";
    case GGML_TYPE_Q5_0: return "Q5_0";
    case GGML_TYPE_Q5_1: return "Q5_1";
    case GGML_TYPE_Q8_0: return "Q8_0";
    case GGML_TYPE_Q8_1: return "Q8_1";
    case GGML_TYPE_Q2_K: return "Q2_K";
    case GGML_TYPE_Q3_K: return "Q3_K";
    case GGML_TYPE_Q4_K: return "Q4_K";
    case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K";
    case GGML_TYPE_Q8_K: return "Q8_K";
    case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
    case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
    case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    case GGML_TYPE_IQ1_S:   return "IQ1_S";
    case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
    case GGML_TYPE_IQ3_S:   return "IQ3_S";
    case GGML_TYPE_IQ2_S:   return "IQ2_S";
    case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
    case GGML_TYPE_I8:      return "I8";
    case GGML_TYPE_I16:     return "I16";
    case GGML_TYPE_I32:     return "I32";
    case GGML_TYPE_I64:     return "I64";
    case GGML_TYPE_F64:     return "F64";
    case GGML_TYPE_IQ1_M:   return "IQ1_M";
    case GGML_TYPE_BF16:    return "BF16";
    case GGML_TYPE_TQ1_0:   return "TQ1_0";
    case GGML_TYPE_TQ2_0:   return "TQ2_0";
    case GGML_TYPE_MXFP4:   return "MXFP4";
    case GGML_TYPE_NVFP4:   return "NVFP4";
    case GGML_TYPE_Q1_0:    return "Q1_0";
    case GGML_TYPE_Q2_0:    return "Q2_0";
    default:                return "unknown";
    }
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return (ne / ggml_blck_size(type)) * ggml_type_size(type);
}
