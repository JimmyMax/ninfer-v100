#!/usr/bin/env bash
# Decides whether a CI run builds and tests the CUDA targets: prints `changed=true` when a file they
# are built, tested or shipped from differs from the base, `changed=false` when none does (for
# $GITHUB_OUTPUT). The changed files go to stderr.
#
# The base:
#   pull request   the branch it merges into: the first parent of the merge commit checked out
#                  (actions/checkout with fetch-depth 2).
#   main           the commit the published image was built from (its revision label). A push
#                  cancels the run of the push before it, so comparing with that push would lose
#                  its changes; comparing with the image cannot.
#   anything else  none: a ci/** branch runs the whole workflow.
# A missing base, or one that cannot be fetched, builds.
#
#   .github/scripts/build-inputs.sh >> "$GITHUB_OUTPUT"
set -euo pipefail

# The paths outside the build. Nothing the compiler, CTest (whose Python helpers read tools/) or the
# image reads is here, and a path this list does not name builds.
outside_build() {
  case "$1" in
    *.md | docs/* | eval/* | examples/* | LICENSE) ;;
    tests/convert/* | tests/pods/* | tests/reference/*) ;;   # pytest alone runs these
    scripts/pods/* | scripts/sweeps/* | scripts/multi-gpu-testing/* | scripts/sm80-probe/*) ;;
    *.bat | *.ps1) ;;                                         # Windows; CI builds Linux only
    .claude/* | .codex/* | .clang-format | .clangd | .gitignore | compose.yaml | flake.*) ;;
    .github/workflows/cancel-superseded.yml) ;;
    *) return 1 ;;
  esac
}

build() { printf '%s\n' "$1" >&2; echo changed=true; exit 0; }

base=""
case "$GITHUB_EVENT_NAME:$GITHUB_REF" in
  pull_request:*)
    base="$(git rev-parse --verify --quiet 'HEAD^1')" || base="" ;;
  push:refs/heads/main)
    base="$(docker buildx imagetools inspect "ghcr.io/${GITHUB_REPOSITORY,,}:latest" \
      --format '{{ index .Image.Config.Labels "org.opencontainers.image.revision" }}')" || base="" ;;
esac
[[ -n "$base" ]] || build "no base to compare with: building"
if ! git cat-file -e "$base^{commit}" 2>/dev/null \
    && ! git fetch --quiet --no-tags --depth=1 origin "$base"; then
  build "base $base cannot be fetched: building"
fi
paths="$(git diff --name-only --no-renames "$base" HEAD)" || build "no diff against $base: building"

inside=()
while IFS= read -r path; do
  [[ -z "$path" ]] || outside_build "$path" || inside+=("$path")
done <<< "$paths"
printf 'base %s; %d changed files, %d of them in the build\n' \
  "$base" "$(grep -c . <<< "$paths" || true)" "${#inside[@]}" >&2
(( ${#inside[@]} == 0 )) || build "$(printf '  %s\n' "${inside[@]:0:20}")"
echo changed=false
