# NOTICE

This repository is a fork of [JimmyMax/ninfer-v100](https://github.com/JimmyMax/ninfer-v100)
(Apache-2.0). The fork chain of the engine is:

    Neroued/ninfer  (upstream, sm_120a / RTX 5090)
     └─ geoffwatts/ninfer-v100  (Tesla V100 port)
         └─ liujun-7788/ninfer-v3-v100  (v3 .ninfer artifacts)
             └─ JimmyMax/ninfer-v100  (v2+v3, nvfp4full)
                 └─ this branch: KVMem working set, native Windows build, vision, WebUI

Engine sources remain under the Apache License 2.0 (see `LICENSE`). This NOTICE records the
third-party sources integrated into this branch and the changes made, per Apache-2.0 §4.

## 1. KVMem integration

- `kvmem/` — portable KVMem host core (block store, residency contract, mean-K retrieval,
  tiers), copied verbatim from [kvmem/kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp),
  branch `feat/ninfer-all-migration`, tag `v0.18.0-ninfer-rc2` (`ae48ec0`).
  That repository ships no `LICENSE` file; its README states: "KVMem-qw3 source is Apache-2.0;
  this port should be treated the same unless a `LICENSE` file is added to this tree."
  The code is redistributed here as Apache-2.0 on that stated basis.
  Citation: Chai et al., *KVMem: Virtualizing Million-Token Agent Workspaces on a Consumer GPU*,
  arXiv:2609.04852.
- `src/targets/qwen3_6/impl/runtime/kvmem_window_impl.h` — port of the official ninfer backend
  integration (`backends/patches/ninfer-kvmem.patch`, 131 files, pinned to
  `iamwavecut/ninfer-all` @ `8319e8f`) onto this tree's runtime, adapted for sm_70 / Volta and
  Windows. Modified vs. the upstream patch; the modifications are stated in the commit history
  and in the file header.

## 2. GGUF / IQ3_S weight loading

GGUF block layouts and ggml MMQ/vec kernels adapted from
[1314521gjy/ninfer-fusion-kvmem](https://github.com/1314521gjy/ninfer-fusion-kvmem) (Apache-2.0)
and [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) (Apache-2.0).

## 3. Build-time assets (not committed)

- WebUI assets embedded by `-DNINFER_WEBUI_DIR` are fetched by `scripts/fetch-webui.sh` from
  [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) releases (default tag `b11165`,
  MIT). Binaries built with the WebUI must carry llama.cpp's MIT license text.
- Vendored libraries under `third_party/` (cpp-httplib, ggml-quants, llama_cpp_fattn, nlohmann,
  spdlog, utf8proc) keep their own license files in their directories.
- FFmpeg and libcurl are not part of this source tree; the Windows build expects them under
  `NINFER_FFMPEG_ROOT` / `NINFER_CURL_ROOT`.

## 4. Model artifacts

`.ninfer` model artifacts are not distributed with this source tree and are not covered by its
license. They are separate downloads from their own publishers.
