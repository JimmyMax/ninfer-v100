#pragma once

#include "artifact/schema.h"

#include <cstdint>
#include <string_view>

namespace ninfer::artifact {

// 本仓库 V100 增量（docs/v100/v2-direct-read-2026-10-16.md）：v2 制品的直读。
//
// v2 与 v3 的差别只在**清单**：对象（名称、偏移、字节数、形状、格式、布局）与权重字节完全
// 保留，升级只是补出 components/bindings/uses/files 并把词汇换成小写下划线。实测同一制品的
// 1184 个张量偏移在升级前后 1:1 相同，所以归一化可以在内存里做——把 {identity, objects}
// 变成 v3 形态的目录 JSON，再交给同一个 parse_directory，热路径与 v3 逐字不变。
//
// 规则与 tools/upgrade_ninfer_v2_to_v3.py 的 make_directory 一一对应（那个工具是这份实现的
// 规格与等价性 oracle：同一输入下两者的输出必须逐字段相同）。
[[nodiscard]] Json normalize_v2_directory(const Json& v2_root, std::uint64_t payload_start,
                                          std::uint64_t file_bytes,
                                          std::string_view entry_name);

} // namespace ninfer::artifact
