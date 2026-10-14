target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_decode_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_matrix.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_matrix_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_a8_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_plan_unified.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_fused.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_fused_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_conv_plan_unified.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_decode_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_small_t_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_w4a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_plan_unified.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_decode_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_small_t_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_post.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_post_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_snapshot_plan_unified.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_independent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_independent_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/q4_q5_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gdn_projected_conv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_decode.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_decode_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_gemm_mma_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_gemm_splitk.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_gemm_splitk_unified.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/q8_gdn_input_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/gdn_input_proj.cpp"
)

if(TARGET ninfer_nvfp4_non_rdc)
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/nvfp4/nvfp4_gdn_input_a4_tma.cu"
  )
endif()

# 本仓库 V100 增量（docs/v100/PLAN.md）：Volta QPN 路径
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_volta_qpn.cu")
endif()

# 本仓库 V100 增量：sm_70 的 FP8 宽 T（prefill）CUTLASS 路径，仅 Volta 构建编译。
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/fp8/fp8_gdn_input_cutlass_sm70.cu")
endif()
