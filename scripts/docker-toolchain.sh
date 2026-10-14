#!/usr/bin/env bash
# Forward commands to the existing CLion project container. Never run Linux ELF on Windows.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container="${STEREOTRACKER_DOCKER_CONTAINER:-st_copy_build}"
workdir="${STEREOTRACKER_DOCKER_WORKDIR:-/tmp/NEW_Steretracker-master_COPY}"
export MSYS_NO_PATHCONV=1
export MSYS2_ARG_CONV_EXCL='*'
if [ "$#" -eq 0 ]; then
    printf 'Usage: %s <command> [arguments...]\n' "$0" >&2
    exit 2
fi
# Verify the writable bind is this checkout before starting or executing anything.
if command -v cygpath >/dev/null 2>&1; then
    expected="$(cygpath -m "$project_root" | tr '[:upper:]' '[:lower:]')"
else
    expected="$project_root"
fi
mounted="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "'"$workdir"'"}}{{.Source}}{{end}}{{end}}' "$container")"
mounted="$(printf '%s' "$mounted" | tr '\\' '/' | tr '[:upper:]' '[:lower:]')"
if [ "$mounted" != "$expected" ]; then
    printf 'Container %s does not bind this checkout at %s; refusing execution.\n' "$container" "$workdir" >&2
    exit 2
fi
if [ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]; then
    docker start "$container" >/dev/null
fi
exec docker exec --workdir "$workdir" "$container" "$@"
