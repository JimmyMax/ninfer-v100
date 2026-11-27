# Release notes

One file per released version, named `RELEASE_NOTES_<X.Y.Z>.md`. They are the authoritative
release text: the `release` workflow publishes the tagged version's file as the GitHub release
body (`body_path`), and both packagers (`scripts/package-release.sh`, `.ps1`) copy it into the
archive they build, failing before they delete or build anything when it is missing.

To cut a release, bump `VERSION` and add `docs/release-notes/RELEASE_NOTES_<new-version>.md`;
see [Cutting a release](../maintainer/release-process.md).

| Version | Notes |
|---|---|
| 1.7.2 | [RELEASE_NOTES_1.7.2.md](RELEASE_NOTES_1.7.2.md) |
| 1.7.1 | [RELEASE_NOTES_1.7.1.md](RELEASE_NOTES_1.7.1.md) |
