#!/usr/bin/env bash
# Pre-push release check: no keys or absolute machine paths in tracked files, local-only files
# untracked, released data unchanged. Exits non-zero on any finding.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
DATA_REF=${DATA_REF:-2f14781}  # last commit with only the released data and website
status=0

# Tracked text files, without the released data and the generated website maps
files=$(git ls-files | grep -v -E '^(data|plots|static)/|^uv\.lock$|^index\.html$')

keys='sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|hf_[A-Za-z0-9]{30,}|(api_?key|token|secret)[^A-Za-z0-9]{1,5}["'\'']?[A-Za-z0-9_-]{20,}'
paths='(^|[^A-Za-z0-9_.-])/(mnt|storage|home|Users|nfs|scratch|work|hpcwork|gpfs|lustre|beegfs)/'
# Optional extra patterns (e.g. usernames, hostnames), kept out of the repo: RELEASE_CHECK_EXTRA='name1|host2'
extra=${RELEASE_CHECK_EXTRA:-}

if hits=$(echo "$files" | xargs grep -n -I -E "$keys" 2>/dev/null); then
  echo "Possible secrets:"; echo "$hits"; status=1
fi
if hits=$(echo "$files" | grep -v '^scripts/release_check.sh$' | xargs grep -n -I -E "$paths" 2>/dev/null); then
  echo "Absolute machine paths:"; echo "$hits"; status=1
fi
if [ -n "$extra" ] && hits=$(echo "$files" | xargs grep -n -I -i -E "$extra" 2>/dev/null); then
  echo "Matches of RELEASE_CHECK_EXTRA:"; echo "$hits"; status=1
fi
for f in .env configs/local.yaml; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then echo "Tracked local file: $f"; status=1; fi
done
for d in slurm data/processed results; do
  if [ -n "$(git ls-files "$d")" ]; then echo "Tracked local folder: $d/"; status=1; fi
done
if ! git diff --quiet "$DATA_REF" -- data/; then
  echo "data/ changed since $DATA_REF:"; git diff --stat "$DATA_REF" -- data/; status=1
fi

[ $status -eq 0 ] && echo "Release check passed."
exit $status
