from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.formats import GGUF_FORMATS, GGUF_FORMATS_BY_TYPE
from tools.artifact.layouts import encoded_size, gguf_blocks_geometry
from tools.convert import gguf_blocks
from tools.convert.sources.gguf import GGUFFile, write_gguf

TYPE_Q8_0 = 8
TYPE_F32 = 0

# ggml-common.h: block values and bytes of every type the importer keeps.
GGML_BLOCKS = {
    2: (32, 18),
    6: (32, 22),
    8: (32, 34),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    16: (256, 66),
    17: (256, 74),
    18: (256, 98),
    19: (256, 50),
    20: (32, 18),
    21: (256, 110),
    22: (256, 82),
    23: (256, 136),
    29: (256, 56),
    42: (64, 18),
}


def _q8_0_file(tmp_path, rows: int, columns: int, seed: int, name: str = "w"):
    generator = np.random.default_rng(seed)
    blocks = generator.integers(0, 256, size=(rows, columns // 32, 34), dtype=np.uint8)
    scales = generator.uniform(0.01, 0.02, size=(rows, columns // 32)).astype(np.float16)
    blocks[:, :, :2] = scales.view(np.uint8).reshape(rows, columns // 32, 2)
    path = tmp_path / "blocks.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [(name, (rows, columns), TYPE_Q8_0, blocks.tobytes())],
    )
    return path, blocks


def _q8_0_values(blocks: np.ndarray) -> np.ndarray:
    """Independent ggml Q8_0 dequantization: binary16 d times each signed code byte."""

    scales = blocks[..., :2].copy().view(np.float16).astype(np.float32)
    codes = blocks[..., 2:].copy().view(np.int8).astype(np.float32)
    return (codes * scales).reshape(blocks.shape[0], -1)


def test_formats_are_ggml_blocks():
    assert {spec.ggml_type for spec in GGUF_FORMATS.values()} == set(GGML_BLOCKS)
    for type_id, (elements, size) in GGML_BLOCKS.items():
        spec = GGUF_FORMATS_BY_TYPE[type_id]
        assert (spec.block_elements, spec.block_bytes) == (elements, size)
        geometry = gguf_blocks_geometry(spec, (3, 4 * elements))
        assert geometry.row_bytes == 4 * size
        assert encoded_size("gguf_blocks_v1", spec, (3, 4 * elements)) == 12 * size
    with pytest.raises(ValueError):
        gguf_blocks_geometry("gguf_q4_k", (2, 128))


def test_rows_are_copied_byte_for_byte_through_the_row_map(tmp_path):
    path, blocks = _q8_0_file(tmp_path, 12, 64, 1)
    order = np.array([7, 2, 11, 2], dtype=np.int64)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (4, 64), lambda b, e: order[b:e])
        words = source.read_encoded(0, 4)
        assert words.format == "gguf_q8_0"
        assert torch.equal(words.codes, torch.from_numpy(blocks.reshape(12, -1)[order]))
        assert torch.equal(
            source.read_encoded(1, 3).codes, torch.from_numpy(blocks.reshape(12, -1)[order[1:3]])
        )


def test_row_values_are_ggml_dequantization(tmp_path):
    path, blocks = _q8_0_file(tmp_path, 6, 64, 2)
    with GGUFFile(path) as gguf:
        source = gguf_blocks.block_source(gguf, "w", (6, 64), lambda b, e: np.arange(b, e))
        assert torch.equal(source.rows(3, 5), torch.from_numpy(_q8_0_values(blocks[3:5])))


def test_output_projection_columns_read_the_grouped_value_heads():
    columns = gguf_blocks.tiled_input_columns()
    assert sorted(columns.tolist()) == list(range(6144))
    for tiled_head in (0, 1, 15, 16, 31, 47):
        key_head, repeat = tiled_head % 16, tiled_head // 16
        assert columns[tiled_head * 128 + 5] == (key_head * 3 + repeat) * 128 + 5


def test_q8_0_gdn_controls_round_to_bf16_in_grouped_head_order(tmp_path):
    # Underdog-Saluki-style exports store ssm_alpha/ssm_beta as Q8_0 blocks: their values reach the
    # Engine's BF16 operand, the exporter's tiled value heads back in the grouped order.
    heads = gguf_blocks.GDN_VALUE_HEADS
    path, blocks = _q8_0_file(tmp_path, heads, 64, 5, "blk.0.ssm_alpha.weight")
    with GGUFFile(path) as gguf:
        values = gguf_blocks.gdn_control_source(gguf, "blk.0.ssm_alpha.weight").rows(0, heads)
    tiled = _q8_0_values(blocks)
    # Grouped head k * 3 + r is the exporter's tiled head r * 16 + k.
    grouped = [r * 16 + k for k in range(16) for r in range(3)]
    assert values.dtype == torch.bfloat16
    assert torch.equal(values, torch.from_numpy(tiled[grouped]).to(torch.bfloat16))


def test_f32_gdn_controls_round_to_bf16_in_grouped_head_order(tmp_path):
    # Standard llama.cpp exports store ssm_alpha/ssm_beta as F32 words: their values reach the
    # Engine's BF16 operand rounded to BF16, the exporter's tiled value heads back in the
    # grouped order.
    heads = gguf_blocks.GDN_VALUE_HEADS
    columns = 64
    f32 = np.random.default_rng(11).uniform(-1.0, 1.0, size=(heads, columns)).astype(np.float32)
    path = tmp_path / "f32.gguf"
    write_gguf(
        path,
        {"general.architecture": "qwen35"},
        [("blk.0.ssm_alpha.weight", (heads, columns), TYPE_F32, f32.tobytes())],
    )
    with GGUFFile(path) as gguf:
        values = gguf_blocks.gdn_control_source(gguf, "blk.0.ssm_alpha.weight").rows(0, heads)
    grouped = [r * 16 + k for k in range(16) for r in range(3)]
    assert values.dtype == torch.bfloat16
    assert torch.equal(values, torch.from_numpy(f32[grouped]).to(torch.bfloat16))
