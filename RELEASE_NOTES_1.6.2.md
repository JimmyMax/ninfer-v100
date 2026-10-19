# NInfer V100 v1.6.2

Prebuilt **Tesla V100 (sm_70)** binaries, compiled with CUDA 12.8 on Ubuntu 24.04 (glibc 2.39).
The runtime hard-requires compute capability 7.0 — other GPUs are rejected at startup.

A patch release on [v1.6.1](https://github.com/JimmyMax/ninfer-v100/releases/tag/v1.6.1) for the V100
container image only. The binaries and their numbers are unchanged.

## Fixed: the `sm70` image could not be started from an orchestration file

`Dockerfile.sm70` set `ENTRYPOINT ["ninfer-serve"]`, so a compose `command:` list — which names the
program itself — produced `ninfer-serve ninfer-serve /models/x.ninfer` and the server stopped with

```text
ninfer-serve: unknown argument: /workspace/models/x.ninfer
```

The image now sets no entrypoint, like every other ninfer image: the command names `ninfer-serve`
first, and a compose service works as written.

```yaml
services:
  llama-cpp-text:
    image: ghcr.io/jimmymax/ninfer-v100:1.6.2-sm70
    command: [ninfer-serve, /workspace/models/model.ninfer, --host, 0.0.0.0, --port, "8000"]
```

```bash
docker run --rm --gpus all -p 8000:8000 --ulimit memlock=-1 \
  -v "$PWD/models:/models" ghcr.io/jimmymax/ninfer-v100:1.6.2-sm70 \
  ninfer-serve /models/model.ninfer --host 0.0.0.0 --port 8000 --model-id model \
  --max-context 32768 --kv-dtype int8
```

**`1.6.1-sm70` and `sm70` as published with v1.6.1 keep the entrypoint** — they still work as
`docker run ... <image> /models/x.ninfer`, but not from a compose `command:` list. Use
`1.6.2-sm70` (and the `sm70` tag, which moves with this release).

## Unchanged

Everything from [v1.6.1](https://github.com/JimmyMax/ninfer-v100/releases/tag/v1.6.1): the
`undefined reference` fix from [#2](https://github.com/JimmyMax/ninfer-v100/issues/2), the dense
Gemma 4 31B, and the artifacts. The tarballs are rebuilt from the same source and carry the same
behaviour.

## Assets

| File | Contents |
|---|---|
| `ninfer-v100-v1.6.2-linux-x86_64-sm70.tar.gz` | `ninfer` CLI + `ninfer-serve` + `ninfer-perplexity` |
| `ninfer-serve-sm70.tar.gz` | server binary only |
| `SHA256SUMS.txt` | checksums for both |

Apache-2.0, as upstream; model weights keep their own licenses (see `LICENSE` and the README).
