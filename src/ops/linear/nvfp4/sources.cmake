target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch_unified.cpp"
  # 本仓库 V100 增量（docs/v100/PLAN.md）：Volta QPN 路径
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_sm70.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_qpn_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_multi_qpn.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_stage_sm70.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_volta_mma_gemm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_prepack_sm70.cu"
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
)

if(TARGET ninfer_nvfp4_non_rdc)
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_a4_tma.cu"
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4_tma.cu"
  )
endif()

# 本仓库 V100 增量：sm_70 的 NVFP4 宽 T（prefill）CUTLASS 路径，仅 Volta 构建编译。
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4_cutlass_sm70.cu")
endif()
