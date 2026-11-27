#!/bin/bash
# One-time: turn this folder into a git repo and push it to GitHub. Run on your Mac, from the repo root:
#   bash scripts/first_push.sh https://github.com/<user>/ds5110-trench-play.git
set -e
URL="${1:?usage: bash scripts/first_push.sh <github-repo-url>}"
[ -f config.py ] || { echo "run this from the repo root"; exit 1; }
git init -b main
git add -A
git status --short | head -50
git commit -m "Initial repo: pipeline 01-11 (incl. team EDA and player-level get-off), Rivanna scripts"
git remote add origin "$URL"
git push -u origin main
echo "pushed to $URL"
