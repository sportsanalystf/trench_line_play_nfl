#!/bin/bash
# ---------------------------------------------------------------------------
# run_pipeline.slurm -- run the pipeline notebooks on UVA Rivanna (batch job).
#
# Run from the repo root:
#   sbatch slurm/run_pipeline.slurm               # all steps 01-11 (incl. 07b)
#   sbatch slurm/run_pipeline.slurm 08 09 10 11   # only these steps (by number)
#   sbatch slurm/run_pipeline.slurm 02 07b        # one or more steps
#
# Spark runs in local mode on every core of one node (UVA RC's advice for
# single-node Spark). Executed notebooks overwrite pipeline/*.ipynb with fresh
# outputs; figures and tables go to outputs/.
# ---------------------------------------------------------------------------
#SBATCH -J ds5110_trench
#SBATCH -p standard
#SBATCH -A ds7200-apt4c
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 05:00:00
#SBATCH -o slurm/logs/slurm-%j.out

module purge
module load spark        # provides pyspark 4.x -- never `pip install pyspark` on Rivanna
module load miniforge    # python + jupyter

cd "$SLURM_SUBMIT_DIR"
[ -f config.py ] || { echo "Run sbatch from the repo root (the folder with config.py)."; exit 1; }

GROUP=/standard/ds7200-apt4c/DS5110_Fall26_Group3
export BDB_DATA_DIR="${BDB_DATA_DIR:-$GROUP/data}"                 # raw CSVs (shared, read-only)
export BDB_PROCESSED_DIR="${BDB_PROCESSED_DIR:-$GROUP/processed}"   # Parquet tables (shared with the team)
export SPARK_DRIVER_MEMORY="48g"
export SPARK_SHUFFLE_PARTITIONS="64"
export CV_PARALLELISM="4"      # 06: models fitted in parallel during CV
export N_BOOT="300"            # 07: bootstrap resamples
export PLAYER_CV_REPEATS="20"  # 07b: repeats of the player-level 5-fold CV
export PERF_REPEATS="3"        # 10: timed runs per benchmark
export PERF_CORES="1,2,4,8,16" # 10: core counts for the scaling test
export PYSPARK_PYTHON=$(which python)
python -c "import pyspark, sys; print('pyspark', pyspark.__version__, sys.executable)"

ALL=(01_data_ingest_and_cleaning 02_eda_data_dictionary_and_quality 03_combine_trait_features
     04_ingame_movement_and_run_attribution 05_dataset_preparation_and_splitting
     06_model_training_cross_validation 07_model_evaluation_and_comparison 07b_player_level_getoff 08_tracking_animations
     09_spark_workflow_diagrams 10_spark_performance_evaluation 11_full_field_play_viewer)
if [ $# -eq 0 ]; then STEPS=("${ALL[@]}"); else
  STEPS=(); for n in "$@"; do for nb in "${ALL[@]}"; do [[ $nb == ${n}_* ]] && STEPS+=("$nb"); done; done
fi
echo "steps: ${STEPS[*]}"

cd pipeline
for nb in "${STEPS[@]}"; do
  echo "=== $(date '+%H:%M:%S') running $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$nb.ipynb" \
      || { echo "FAILED: $nb"; exit 1; }
done
echo "=== all steps finished $(date '+%H:%M:%S')"
seff "$SLURM_JOB_ID" 2>/dev/null || true
