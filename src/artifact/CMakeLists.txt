add_library(ninfer_artifact STATIC
  schema.cpp
  formats.cpp
  layouts.cpp
  file_io.cpp
  reader.cpp
  binder.cpp
  materializer.cpp
  transcode.cpp
  permute_row_split.cu
  # 本仓库 V100 增量：NVFP4 的 QPN 片段序（加载时/恢复时原地施加）
  permute_nvfp4_qpn.cu
  permute_fp8_qpn.cu
  views.cpp
)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core ninfer::json)
