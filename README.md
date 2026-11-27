# Trench Play: Do Combine First-Step Quickness, Lateral Agility and Burst Predict NFL Line Play?

**DS 5110 Big Data Systems · University of Virginia · Fall 2026 · Group 3**
Faizan Khan · Isaac Marr · Jarmarr Washington · Weston Williams

Data: [NFL Big Data Bowl 2027](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027) (CC BY-NC 4.0): 9 CSV files, ~2.3 GB, 510 rookies from the 2023–25 draft classes, 22.65M in-game tracking frames. The project studies the **239 offensive (121) and defensive (118) linemen**. Everything runs in PySpark on UVA Rivanna.

## The question

The 2027 release adds 10 Hz tracking of every Combine position drill. From it we measure three movement traits and ask whether they predict NFL line play:

| Trait | Measured from Combine tracking as | Stopwatch equivalent |
| --- | --- | --- |
| **First-step quickness** | time to cover 1 and 2 yards from the first move; speed 0.3 s and 0.5 s in | 10-yard split |
| **Lateral agility** | at each direction reversal: time slowed down, braking into it, re-acceleration out of it | short shuttle, 3-cone |
| **Burst** | peak acceleration in the first 1.5 s; speed at 1.0 s; ground covered in 1 s | broad jump, vertical, 40 |

| Group | Outcome (per snap, regular season) |
| --- | --- |
| DL | **pass rush**: pressure on the snap; get-off time |
| OL | **pass blocking**: pressure allowed; peak pressure probability allowed |
| OL | **run blocking**: run success (EPA > 0) when he is at the run's point of attack (gap credit from the play-by-play) |

## Repository layout

```
ds5110-trench-play/
├── README.md               this file
├── config.py               ALL paths, constants and the Spark session (every notebook imports it)
├── requirements.txt
├── pipeline/               the shared, numbered, step-by-step analysis (run in order)
│   ├── 01_…ipynb … 11_…ipynb
│   └── src/                the same notebooks as plain .py files (easy to read and diff)
├── slurm/                  run_pipeline.slurm (all steps or chosen steps), logs/
├── scripts/                first_push.sh, pull_outputs_from_rivanna.sh
├── docs/                   workflow.md (GitHub ⇄ Rivanna ⇄ laptop), deliverables
├── outputs/                figures/ + tables/ (in git); videos, play viewer, models (regenerated, not in git)
└── data/                   raw CSVs: NOT in git (see data/README.md)
```

## The pipeline, step by step

Each notebook starts with **what it does, its inputs and its outputs**, and saves every figure / table it makes to `outputs/` with its step number as the prefix.

**All computation runs in Spark** (DataFrames, window functions, Spark ML). The shared helpers in `config.py` (`describe`, `histogram`, `pearson`, `spearman`, `avg_rank`) compute statistics as Spark aggregations. pandas only receives finished, small results (a few thousand rows at most) to draw a figure or write a CSV, and the frames of the few plays drawn in 08 and 11. The one deliberate exception is the pandas baseline that notebook 10 benchmarks Spark against.

| Step | Notebook | What it does | Main outputs |
| --- | --- | --- | --- |
| 01 | `01_data_ingest_and_cleaning` | explicit schemas for all 9 CSVs, OL + DL cohort, CSV → Parquet | `line_*.parquet`, `combine_tracking.parquet` |
| 02 | `02_eda_data_dictionary_and_quality` | data dictionary, descriptive stats, data-quality checks, EDA figures; whole-release view (all positions, drill coverage, snap thresholds, get-off by alignment) | `02_*.csv`, `02_*.png` |
| 03 | `03_combine_trait_features` | first step / lateral agility / burst from 10 Hz Combine tracking | `trait_indices.parquet`, `03_*` |
| 04 | `04_ingame_movement_and_run_attribution` | in-game get-off and speed, OL side from the hash marks, run-gap credit, Combine → game translation | `ingame_snaps.parquet`, `04_*` |
| 05 | `05_dataset_preparation_and_splitting` | five modeling tables; split by player (test = 2025 class) | `task_*.parquet`, `feature_sets.json` |
| 06 | `06_model_training_cross_validation` | 9-model ladder × 5 tasks as Spark ML Pipelines, player-grouped 5-fold CV | `06_cv_*.csv`, `models/` |
| 07 | `07_model_evaluation_and_comparison` | test metrics, bootstrap CIs, trait effects vs benchmark, sensitivity | `07_*.csv`, `07_*.png` |
| 07b | `07b_player_level_getoff` | player-level view: Combine drills vs a DL's average get-off (correlations, linear regression, repeated CV, 2025-class test, link to the snap-level results) | `07b_*.csv`, `07b_*.png` |
| 08 | `08_tracking_animations` | MP4s of every trait metric on real drills and NFL snaps | `animations/` |
| 09 | `09_spark_workflow_diagrams` | flowchart, Spark on Rivanna, stages / shuffles, ML pipeline + CV | `09_*.png` |
| 10 | `10_spark_performance_evaluation` | 11 benchmarks of the Spark choices taught in DS 5110 | `10_*.csv`, `10_*.png` |
| 11 | `11_full_field_play_viewer` | whole plays on a full field with OL / DL stats live (HTML + MP4) | `play_viewer/`, `animations/field_*` |

Steps 07b and 08–11 only need 01–05 (07b and 09 also read 06–07 results when present), so they can be re-run on their own.

## Run it

**On Rivanna (the normal way).** From the repo root:
```bash
module load spark miniforge
pip install --user -r requirements.txt      # one time; never pip install pyspark on Rivanna
sbatch slurm/run_pipeline.slurm             # all steps (about 1.5 h on 16 cores)
sbatch slurm/run_pipeline.slurm 08 09 10 11 # or only some steps
tail -f slurm/logs/slurm-<jobid>.out
```
Raw data is read from `/standard/ds7200-apt4c/DS5110_Fall26_Group3/data` and the Parquet tables are written to `/standard/ds7200-apt4c/DS5110_Fall26_Group3/processed`, both shared by the team. Interactive work: Open OnDemand → JupyterLab, open any notebook in `pipeline/`.

**On a laptop.** `pip install "pyspark>=3.4" -r requirements.txt`, put the CSVs in `data/` (or `../data`), run the notebooks in order. Set `SPARK_DRIVER_MEMORY=6g` or so.

**Keeping laptop, GitHub and Rivanna in sync:** see [`docs/workflow.md`](docs/workflow.md).

## Results so far (full run on Rivanna)

* **Burst → DL get-off is the trait effect that holds up on the 2025 class:** +0.007 R² over the context benchmark, CI [+0.003, +0.012]. It was also significant with the earlier train/validation split (+0.006, CI [+0.001, +0.013]), so it does not depend on which players happen to be in training.
* **Lateral agility (+0.011) and all three traits together (+0.023) also beat the benchmark for get-off with the current split, but not with the earlier one.** With only ~59 DL in training, swapping 15 of them moves these effects, so treat them as suggestive. The stopwatch drills add +0.020, CI [−0.001, +0.044].
* No trait improves prediction of **pressure** (DL or OL) or **run success**. Players do differ: knowing each player's true average lifts AUROC by about 0.04–0.05 for pressure and 0.02 for run success. But Combine movement drills don't capture what differs.
* At the **player level** (07b), the stopwatch speed and jump drills correlate strongly with a DL's average get-off (r ≈ 0.6–0.7, 55 players). In repeated cross-validation they add about 0.06 R² over alignment alone (0.39 vs 0.33); the tracking traits correlate less (|r| ≈ 0.1–0.35) and add nothing. Only about 10% of get-off variance is between players, so the same effect is worth under 0.01 R² per snap. The 2025 class (11 players) is too small to confirm it out of time.
* **Spark** (notebook 10, 16 cores):
  * scanning 2 GB of tracking CSV is 12.7× faster on 16 cores than on 1;
  * Parquet is 6.5× faster than CSV;
  * DataFrames are 49× faster than RDDs;
  * broadcast joins and skew-aware keys are 2–3× faster;
  * small model fits gain nothing from more cores (Amdahl's law).

## Course requirements map

| Requirement | Where |
| --- | --- |
| Import and preprocessing (schemas, filtering, imputation, binning, outliers, feature engineering) | 01, 03, 04, 05 |
| Splitting: test = 2025 class, validation = 20% of 2023–24 players, 5 player-grouped folds | 05 |
| EDA (2+ figures), data dictionary, data-quality assessment | 02 (plus figures in 03, 04) |
| Models: benchmark + sophisticated models, as Spark ML Pipelines | 06; player-level models in 07b |
| K-fold CV of hyperparameters | 06: `CrossValidator(foldCol="fold")` |
| Evaluation: AUROC, accuracy, precision, recall, F1, confusion matrix, lift; R², RMSE, MAE; comparison | 07 |
| Sensitivity analysis | 07: tree hyperparameters; DL get-off trait effects over 10 re-drawn player splits |
| Why Spark: distributed processing, performance | 09, 10 |

## Team and contributions

| Who | Computing ID | Contributed |
| --- | --- | --- |
| Faizan Khan | xdy6sg | pipeline design and integration (01–11), Rivanna runs |
| Isaac Marr | iam8bu | data loading with explicit schemas; whole-release EDA and the get-off idea (02 §6, 07b start) |
| Jarmarr Washington | sek8rf | player-level get-off analysis: correlations, imputation, Spark ML regression (07b §2–3) |
| Weston Williams | cyw8tw | |

Everyone works in the same `pipeline/` notebooks. New analyses are added as a section of the step they belong to, or as a lettered step (like 07b), so the numbered order still runs end to end.
