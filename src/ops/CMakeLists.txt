# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(ninfer_ops STATIC)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)
# The opt-in cuBLAS prefill route hands its GEMM to cuBLAS; see
# ops/linear_swiglu/q4cublas/w4_cublas_prefill.h for why that route exists.
target_link_libraries(ninfer_ops PRIVATE CUDA::cublas)

# 本仓库 V100 增量：Volta 构建下 sm_70 的 NVFP4/FP8 宽 T 内核依赖 CUTLASS 头文件库
# （顶层 CMakeLists.txt 里 FetchContent，锁定 v4.4.2）。
# EXPERIMENT-11：Turing 也需要 CUTLASS（128×128 tile GEMM）
if(NINFER_ARCH_VOLTA OR NINFER_ARCH_TURING)
  target_link_libraries(ninfer_ops PRIVATE nvidia::cutlass::cutlass)
endif()

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers.
# The SM120 tensor-core qualifiers do not exist on sm_86/sm_89, so this fork creates the archive
# only in a build with 120a, and compiles it for 120a alone; only a card that runs the 120a code
# launches its kernels. Family manifests add to it only when the target exists.
if(NINFER_ARCH_BLACKWELL)
  add_library(ninfer_nvfp4_non_rdc STATIC)
  set_target_properties(ninfer_nvfp4_non_rdc PROPERTIES CUDA_ARCHITECTURES 120a)
  ninfer_internal_includes(ninfer_nvfp4_non_rdc)
  ninfer_cuda_non_rdc_archive(ninfer_nvfp4_non_rdc)
  target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE
    ninfer_core ${NINFER_CUDART_TARGET} CUDA::cuda_driver)
  target_link_libraries(ninfer_ops PRIVATE ninfer_nvfp4_non_rdc)
endif()

# The vendored llama.cpp flash-attention tree (third_party/llama_cpp_fattn) defines CUDA_CHECK,
# WARP_SIZE and a common.cuh that collide with src/ops, so its include directory is scoped to the
# single translation unit that drives it.
set_source_files_properties(
  "${CMAKE_CURRENT_LIST_DIR}/launcher/volta_flash_llama.cu"
  TARGET_DIRECTORY ninfer_ops
  PROPERTIES
    INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/third_party/llama_cpp_fattn"
    COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-diag-suppress=940>")
# 本仓库 V100 增量：Volta-only 单元自成一个静态库，**只按 sm_70 编译**。
# 它们的实现整个在 `#if defined(NINFER_TIER_VOLTA) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700)`
# 里，并依赖只在该通道里声明的类型（如 Fp8VoltaQpnSchedule）。多架构构建（"70;89"）若把它们
# 和别的架构编在同一个目标里，会多出一份 compute_89 的宿主通道，它引用该文件里只有 700 通道
# 定义的内部链接核函数 → 链接期 undefined reference。CMake 只有 target 级的 CUDA_ARCHITECTURES
# （没有源文件级的），所以用独立目标，和上面的 ninfer_nvfp4_non_rdc 同一思路。
if(NINFER_ARCH_VOLTA)
  add_library(ninfer_ops_volta STATIC)
  set_target_properties(ninfer_ops_volta PROPERTIES CUDA_ARCHITECTURES 70)
  ninfer_internal_includes(ninfer_ops_volta)
  ninfer_cuda_archive(ninfer_ops_volta)
  target_link_libraries(ninfer_ops_volta PUBLIC ninfer_core)
  target_link_libraries(ninfer_ops_volta PRIVATE nvidia::cutlass::cutlass CUDA::cublas)
  target_link_libraries(ninfer_ops PRIVATE ninfer_ops_volta)
endif()

# 加入 Volta-only 的源文件；非 Volta 构建里它们完全不存在，与改动前一致。
function(ninfer_ops_add_volta_sources)
  if(NOT NINFER_ARCH_VOLTA)
    return()
  endif()
  target_sources(ninfer_ops_volta PRIVATE ${ARGN})
endfunction()

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")

if(NINFER_COMPAT_PATH)
  # NVFP4 W4A4 uses mma.sync...kind::mxf4nvf4.block_scale; FP8 A8 uses
  # mma.sync...kind::f8f6f4. Both qualifiers are Blackwell-only, and sm_80/86/89 have no FP8 or FP4
  # tensor-core path to fall back to, so a build without 120a leaves the owning translation units
  # out. The A16 dequantizing routes for both formats are kept and remain fully functional. A build
  # with 120a on this path keeps both (NINFER_SM120_FP8, NINFER_SM120_NVFP4).
  if(NOT NINFER_ARCH_BLACKWELL)
    get_target_property(ninfer_ops_sources ninfer_ops SOURCES)
    list(FILTER ninfer_ops_sources EXCLUDE REGEX "nvfp4/.+_(w4a4|a4)\\.cu$")
    list(FILTER ninfer_ops_sources EXCLUDE REGEX "fp8/.+_a8(_unified)?\\.cu$")
    set_property(TARGET ninfer_ops PROPERTY SOURCES ${ninfer_ops_sources})
    target_sources(ninfer_ops PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/nvfp4_sm86_stubs.cpp"
      "${CMAKE_CURRENT_LIST_DIR}/fp8_sm86_stubs.cpp")
  endif()
  target_compile_definitions(ninfer_ops PUBLIC NINFER_SM8X_COMPAT=1)
endif()
if(NINFER_ARCH_BLACKWELL)
  target_compile_definitions(ninfer_ops PUBLIC NINFER_SM120_NVFP4=1 NINFER_SM120_FP8=1)
endif()
