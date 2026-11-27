#!/bin/bash
# Copy the latest outputs (incl. videos, play viewer, executed notebooks) from the shared
# Rivanna clone to this laptop. Run from the repo root on your Mac:
#   bash scripts/pull_outputs_from_rivanna.sh <computing-id>
set -e
ID="${1:?usage: bash scripts/pull_outputs_from_rivanna.sh <computing-id>}"
REMOTE="$ID@login.hpc.virginia.edu:/standard/ds7200-apt4c/DS5110_Fall26_Group3/ds5110-trench-play"
[ -f config.py ] || { echo "run this from the repo root"; exit 1; }
rsync -av --progress "$REMOTE/outputs/" outputs/
rsync -av --progress --include='*.ipynb' --exclude='*' "$REMOTE/pipeline/" pipeline/
echo "done. 'git status' shows what changed; commit the figures/tables/notebooks you want to share."
