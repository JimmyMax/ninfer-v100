add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  # 本仓库 V100 增量（docs/v100/v2-direct-read-2026-10-16.md）：v2 清单→v3 目录的归一化
  v2_directory.cpp
  binder.cpp
  materializer.cpp
  transcode.cpp
  permute_row_split.cu
  # 本仓库 V100 增量：NVFP4 的 QPN 片段序（加载时/恢复时原地施加）
  permute_nvfp4_qpn.cu
  permute_fp8_qpn.cu
  # 本仓库 V100 增量：Q6_K 的逐行密集 6-bit 码字流（Volta 向量内核 64-bit 读取）
  permute_q6k_dense.cu
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
