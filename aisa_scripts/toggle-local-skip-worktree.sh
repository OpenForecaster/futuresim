#!/usr/bin/env bash
set -euo pipefail

# Local-only files to keep untouched during git pulls.
FILES=(
  "pyproject.toml"
  "uv.lock"
  "aisa_scripts/toggle-local-skip-worktree.sh"
)

usage() {
  echo "Usage: $0 {on|off|status}"
  echo "  on     -> mark files as skip-worktree"
  echo "  off    -> unmark skip-worktree"
  echo "  status -> show skip-worktree status for managed files"
}

ensure_repo_root() {
  local root
  root="$(git rev-parse --show-toplevel)"
  cd "$root"
}

tracked_files() {
  local tracked=()
  local f
  for f in "${FILES[@]}"; do
    if git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
      tracked+=("$f")
    fi
  done
  printf '%s\n' "${tracked[@]}"
}

set_on() {
  mapfile -t tracked < <(tracked_files)
  if [[ ${#tracked[@]} -gt 0 ]]; then
    git update-index --skip-worktree "${tracked[@]}"
  fi
  echo "Enabled skip-worktree:"
  printf '  - %s\n' "${tracked[@]:-<none>}"
}

set_off() {
  mapfile -t tracked < <(tracked_files)
  if [[ ${#tracked[@]} -gt 0 ]]; then
    git update-index --no-skip-worktree "${tracked[@]}"
  fi
  echo "Disabled skip-worktree:"
  printf '  - %s\n' "${tracked[@]:-<none>}"
}

show_status() {
  echo "Managed files:"
  for f in "${FILES[@]}"; do
    local tag
    tag="$(git ls-files -v -- "$f" | awk '{print $1}' || true)"
    if [[ "$tag" == S ]]; then
      echo "  [skip]   $f"
    elif [[ -n "$tag" ]]; then
      echo "  [track]  $f"
    elif [[ -e "$f" ]]; then
      echo "  [untracked] $f"
    else
      echo "  [missing] $f"
    fi
  done
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  ensure_repo_root

  case "$1" in
    on) set_on ;;
    off) set_off ;;
    status) show_status ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
