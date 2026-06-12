#!/usr/bin/env bash
#
# Re-publish this repo with a CLEAN history. Run ON YOUR MAC:
#
#     cd ~/Desktop/projects/lob-market-manipulation-detection
#     bash setup_git.sh
#
# Why a fresh history: the clone currently tracks ~90MB of parquet data and ~120MB
# of experiment submission CSVs. A fresh init drops those blobs entirely and the new
# .gitignore keeps them out. This FORCE-PUSHES over the existing repo — intended,
# since you're replacing the old group version with your cleaned one.

set -euo pipefail
cd "$(dirname "$0")"

REPO="lob-market-manipulation-detection"
OWNER="6ixE11even"

# Clear the bloated history.
rm -rf .git
git init -b main
git config user.name  >/dev/null 2>&1 || git config user.name  "Tejas Pandya"
git config user.email >/dev/null 2>&1 || git config user.email "tbp8777@nyu.edu"

git add pyproject.toml uv.lock .gitignore configs src
git commit -m "Restructure into a documented src package (features, scorers, ensemble)"

git add notebooks
git commit -m "Keep experiment notebooks as the reproducible model trail"

git add -A
git commit -m "Add consolidated pipeline CLI, diagnostics, and rewrite README"

git remote add origin "https://github.com/$OWNER/$REPO.git"

echo
echo "Committed locally with a clean history. To publish (replaces old history):"
echo "  git push --force -u origin main"
echo
read -r -p "Push now with --force? [y/N] " ans
[[ "${ans:-N}" =~ ^[Yy]$ ]] && git push --force -u origin main && echo "Done — https://github.com/$OWNER/$REPO"
