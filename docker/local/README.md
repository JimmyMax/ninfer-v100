# Local image recipes

Personal, unpublished image recipes for this V100/4080S line. The published images are built
from [`Dockerfile`](../../Dockerfile) (sm_75/86/120a toolchain image) and
[`Dockerfile.sm70`](../../Dockerfile.sm70) (V100, `ghcr.io/<owner>/<repo>:<version>-sm70`); both
stay at the repository root because `.github/workflows/` and `.github/scripts/` invoke them
there.

| Recipe | Produces | Notes |
|---|---|---|
| `Dockerfile.sm70.local` | full V100 build (CUDA 12.8, `-DCMAKE_CUDA_ARCHITECTURES=70`) | slow (~1 h); see `docs/v100/PLAN.md` |
| `Dockerfile.sm70-runtime.local` | repackage an already-built `ninfer-serve` into the runtime base | seconds; context is `docker-runtime/` |
| `Dockerfile.1.7.1.local` | single image with both sm_70 and sm_89 binaries | historical |
| `Dockerfile.1.7.2.local` | the 1.7.2 dual-architecture image | historical |
| `Dockerfile.1.8.0-inkling2.local` | 1.7.2 layout with the `inkling2` sm_70 binary | current line |
| `Dockerfile.1.8.0-inkling2-sm89.local` | adds the `inkling2` sm_89 binary | current line |

Build them from the repository root, with the context unchanged:

```sh
docker build -f docker/local/Dockerfile.sm70.local         -t ninfer:v100-gemma4 .
docker build -f docker/local/Dockerfile.sm70-runtime.local -t ninfer:v100-gemma4 docker-runtime/
```

`docker-runtime/` holds the built binaries these recipes copy and is gitignored.
