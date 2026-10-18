# Cutting a release

This repository publishes its binaries as the Docker image, which CI builds from every `master`
commit that passes it and changes a file the image is built from
([README](../../README.md#docker)); the image is tagged `latest`, `sha-<commit>` and the contents of
`VERSION`. Release archives for Windows and Linux are an
additional, manual path that no release has used yet. This page is that path.

## The flow

1. **Prepare.** Branch from `master`, set `VERSION` to the full tag (for example `0.13.0-rtx3090`),
   write `RELEASE_NOTES_<X.Y.Z>.md` for it, and merge. The packagers derive everything
   version-specific from `VERSION` and refuse to run without the matching release notes.
2. **Build both platforms from the same commit.** The binaries do not embed `VERSION`, so they can
   be built before the release PR merges.
   - Windows: `scripts\build.ps1 -Package` builds `ninfer`, `ninfer-serve` and `ninfer_bench`
     ([Windows build](../build-windows.md)) and packages them.
   - Linux: a fresh `git archive` of the commit, `-DCMAKE_CUDA_ARCHITECTURES=86`,
     `-DBUILD_TESTING=OFF`, `-DNINFER_BUILD_BENCHMARKS=ON`, targets `ninfer ninfer-serve
     ninfer_bench`, then `scripts/package-release.sh` (or `scripts/build.sh --package`).
3. **Package.** `scripts/package-release.ps1` writes the Windows zip and `scripts/package-release.sh`
   the Linux tar.gz, each with `SHA256SUMS-v<X.Y.Z>-{windows,linux}.txt`, under `dist/`. The
   archive README is [`release-archive-windows.md`](../release-archive-windows.md) or
   [`release-archive-linux.md`](../release-archive-linux.md); check that it still describes the
   archive's requirements and launcher profiles.
4. **Smoke-test on a real card** (below), not just the unit tests.
5. **Publish.** Tag the merge commit (annotated), push the tag, then
   `gh release create vX.Y.Z-rtx3090 <zip> <tar.gz> <both SHA256SUMS files> -R JimmyMax/ninfer-v100
   --notes-file RELEASE_NOTES_X.Y.Z.md --title "..." --latest`.

## Gotchas

**Always pass `-R JimmyMax/ninfer-v100` to `gh`.** The checkout also has the `upstream` and
`ashalliants` remotes, and `gh` without `-R` may resolve to another fork.

**Windows executables import cuBLAS, and the archive must carry it.** The cuBLAS prefill route makes
every executable import `cublas64_12.dll` (which imports `cublasLt64_12.dll`), so a machine without
the CUDA Toolkit cannot start any of them (`0xC0000135`), even when `--prefill-cublas` is never used.
`package-release.ps1` copies both DLLs from `%CUDA_PATH%\bin` and refuses to produce an archive whose
executables do not start with only Windows on `PATH`. Do not weaken that check.

**Redistributing NVIDIA's DLLs has conditions.** The CUDA Toolkit EULA (Attachment A) lists the
cuBLAS DLLs as redistributable. The grant requires that the application has material functionality
of its own, that the DLLs are used only by it, and that the distribution terms are consistent with
the EULA and protect NVIDIA's rights. The packager therefore ships the toolkit's `EULA.txt` as
`NVIDIA-CUDA-EULA.txt`, and the archive README names the two DLLs as NVIDIA's, outside the
project's Apache-2.0 licence. Re-read the EULA of the toolkit that builds a release.

**Linux links cuBLAS, FFmpeg and glibc dynamically.** The Linux archive does not ship `libcublas`
(the Lt library alone is about 500 MB), so `release-archive-linux.md` states the requirements: the
glibc of the build host, the CUDA runtime libraries including cuBLAS, FFmpeg and libcurl. Check them
with `ldd` on the new binaries and update the archive README when they change.

**Asset sizes.** Every executable links every kernel and weighs about half a gigabyte; the Windows
zip with the cuBLAS DLLs is about 1.5 GB. GitHub rejects release assets over 2 GiB.

**Smoke-test the launcher default on the real card.** A desktop holding 2-3 GiB of a 24 GB card
refuses the default DFlash2 profile, both on the engine's runtime reservation and, on Windows, on
pinned host state. `run.{sh,bat}` step down by themselves when startup is refused for lack of GPU
memory ([launcher profiles](launcher-profiles.md)). Start the launcher with nothing overridden and
confirm it serves a request.

## Checks before publishing

- `scripts/check-linux-scripts.sh` passes.
- Both archives: every file matches its inner `SHA256SUMS.txt`, and the outer checksum file verifies.
- Windows: all three executables start with only Windows on `PATH`.
- Linux: a fresh unpack runs `./ninfer-serve --help` and `./run.sh --help`.
- The default launcher profile starts and serves a chat completion on a real 24 GB card.
