#!/usr/bin/env python3
"""Merge a standalone DSpine drafter .ninfer into a target .ninfer so the drafter becomes a
`dspine` component of the same artifact. The local V100 loader materialises a single artifact,
so the drafter must live in the target file for the existing companion path to work."""
import json
import struct
import sys

MODEL_DIR = "/mnt/data/dockerdata/llama-cpp/models"
TARGET = f"{MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
DRAFTER = f"{MODEL_DIR}/Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer"
OUT = f"{MODEL_DIR}/Qwen3.8-27B-GSQ-RCO-IQ3_S-dspine.ninfer"

K_HEADER = 32
K_ALIGN = 4096


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(K_HEADER)
    magic = head[:8]
    assert magic == b"NINFER\x00\x03", f"bad magic {magic!r} in {path}"
    json_bytes = struct.unpack("<Q", head[8:16])[0]
    uuid = head[16:32]
    with open(path, "rb") as f:
        f.seek(K_HEADER)
        text = f.read(json_bytes)
    directory = json.loads(text)
    return uuid, directory, json_bytes


def _payload_start(f):
    f.seek(8)
    json_len = struct.unpack("<Q", f.read(8))[0]
    return ((K_HEADER + json_len + K_ALIGN - 1) // K_ALIGN) * K_ALIGN


def main():
    t_uuid, t_dir, t_json_len = read_header(TARGET)
    _, d_dir, d_json_len = read_header(DRAFTER)
    t_payload_start = ((K_HEADER + t_json_len + K_ALIGN - 1) // K_ALIGN) * K_ALIGN
    d_payload_start = ((K_HEADER + d_json_len + K_ALIGN - 1) // K_ALIGN) * K_ALIGN

    t_objects = t_dir["objects"]
    d_objects = d_dir["objects"]

    # Highest weight index in the target; the drafter's weights continue after it.
    import re
    t_max = max((int(re.match(r"weight/(\d+)", o["id"]).group(1))
                 for o in t_objects if o["id"].startswith("weight/")), default=-1)
    base = t_max + 1
    rename = {}
    for i, o in enumerate(d_objects):
        m = re.match(r"weight/(\d+)", o["id"])
        assert m, f"unexpected drafter object id {o['id']}"
        old = int(m.group(1))
        rename[old] = base + i

    def new_weight_id(old_index):
        return f"weight/{rename[old_index]:06d}"

    # Target payload size = last object end.
    t_payload = max(o["offset"] + o["bytes"] for o in t_objects)

    # Renamed, offset-adjusted drafter objects.
    merged_objects = list(t_objects)
    for o in d_objects:
        m = re.match(r"weight/(\d+)", o["id"])
        old = int(m.group(1))
        no = dict(o)
        no["id"] = new_weight_id(old)
        no["offset"] = o["offset"] + t_payload
        merged_objects.append(no)

    def remap_ref(ref):
        m = re.match(r"weight/(\d+)", ref)
        if not m:
            return ref
        return new_weight_id(int(m.group(1)))

    merged_bindings = dict(t_dir["bindings"])
    for key, val in d_dir["bindings"].items():
        nv = {}
        if "object" in val:
            nv["object"] = remap_ref(val["object"])
        if "parts" in val:
            nv["parts"] = [
                {**p, "object": remap_ref(p["object"])} for p in val["parts"]
            ]
        merged_bindings[key] = nv

    merged_uses = list(t_dir["uses"]) + list(d_dir["uses"])
    merged_components = dict(t_dir["components"])
    dspine_component = dict(d_dir["components"]["dspine"])
    dspine_component["target"] = "text"
    merged_components["dspine"] = dspine_component

    total_payload = t_payload + max(o["offset"] + o["bytes"] for o in d_objects)
    merged_dir = {
        "components": merged_components,
        "objects": merged_objects,
        "bindings": merged_bindings,
        "uses": merged_uses,
        "metadata": t_dir.get("metadata", {}),
        "provenance": {
            **t_dir.get("provenance", {}),
            "merged_dspine": "Qwen3.8-27B-DSpine-b16-ninfer-v3.ninfer",
        },
        "files": [{"path": None, "payload_bytes": total_payload}],
    }

    payload = json.dumps(merged_dir, separators=(",", ":")).encode("utf-8")
    json_len = len(payload)
    payload_start = ((K_HEADER + json_len + K_ALIGN - 1) // K_ALIGN) * K_ALIGN
    pad = payload_start - (K_HEADER + json_len)
    assert pad >= 0

    with open(OUT, "wb") as f:
        f.write(b"NINFER\x00\x03")
        f.write(struct.pack("<Q", json_len))
        f.write(t_uuid)
        f.write(payload)
        f.write(b"\0" * pad)
        import shutil

        with open(TARGET, "rb") as ts:
            ts.seek(t_payload_start)
            shutil.copyfileobj(ts, f)
        with open(DRAFTER, "rb") as ds:
            ds.seek(d_payload_start)
            shutil.copyfileobj(ds, f)

    print(f"wrote {OUT}")
    print(f"  components: {list(merged_components.keys())}")
    print(f"  objects: {len(merged_objects)} (target {len(t_objects)} + drafter {len(d_objects)})")
    print(f"  weight rename base: {base} (target max {t_max})")
    print(f"  json_len={json_len} payload_start={payload_start} total_payload={total_payload}")


if __name__ == "__main__":
    main()
