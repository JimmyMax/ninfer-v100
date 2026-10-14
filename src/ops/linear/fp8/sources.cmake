target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_volta_qpn_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_prepack_sm70.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120_unified.cu"
)

# 本仓库 V100 增量：sm_70 的 FP8 宽 T（prefill）CUTLASS 路径，仅 Volta 构建编译。
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/fp8_cutlass_sm70.cu")
endif()
