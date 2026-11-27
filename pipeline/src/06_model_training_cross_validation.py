# %% [markdown]
# # 06 · Model training with player-grouped cross-validation (all five tasks)
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# For every task from notebook 05 it trains the same ladder of **Spark ML Pipelines**, each tuned with `CrossValidator(foldCol="fold")`. The folds are player-grouped, so a player's snaps never sit in both the fitting and the scoring fold.
#
# | Model | Features (on top of pre-snap context) | Question it answers |
# |---|---|---|
# | `benchmark` | none | how far does the game situation alone go? |
# | `first_step` | tracking first-step index | does **first-step quickness** add anything? |
# | `lateral` | tracking lateral-agility index | does **lateral agility** add anything? |
# | `burst` | tracking burst index | does **burst** add anything? |
# | `traits` | all three tracking indices | the three traits together |
# | `stopwatch` | the three stopwatch indices (+ missing flags) | the traditional equivalent |
# | `traits_size` | three tracking + three stopwatch indices + weight, height, arm length | everything, linear |
# | `rf` | same as `traits_size` | non-linear: Random Forest |
# | `gbt` | same as `traits_size` | non-linear: Gradient-Boosted Trees |
#
# Linear models are logistic regression (classification) or linear regression (regression). Trees are made **player-robust** with at least 500–2,000 snaps per leaf, so a leaf cannot single out one player. With only ~50 training players per group, a handful of compact trait indices is the main defence against memorising individuals.
#
# Selection metric: AUROC (classification) or R² (regression). The best setting is refitted on all training players and saved.
#
# **Inputs:** `processed/task_*.parquet`, `outputs/feature_sets.json`.  **Outputs:** `outputs/models/<task>/<model>/`, `outputs/tables/06_cv_results.csv`, `outputs/tables/06_cv_summary.csv`.
#
# Environment: `QUICK=1` = one setting per grid; `CV_PARALLELISM` = parallel fits (≈ cores / 2 on Rivanna); `TASKS=dl_pressure,ol_run` = run a subset.

# %%
import sys, json, time, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import config as C

spark = C.get_spark("06_train")
QUICK = os.environ.get("QUICK") == "1"
PAR = int(os.environ.get("CV_PARALLELISM", "2"))
FS = json.load(open(C.OUTPUT_DIR / "feature_sets.json"))
TASKS = os.environ.get("TASKS", ",".join(FS["tasks"])).split(",")
print("tasks:", TASKS, "| QUICK:", QUICK, "| parallelism:", PAR)

# %% [markdown]
# ## 1. Pipeline builder and model ladder

# %%
def pipeline(task, estimator, player_feats, scale):
    cats = FS["context_cat"] + (FS["run_cat"] if task == "ol_run" else [])
    nums = FS["context_num"] + (FS["run_num"] if task == "ol_run" else [])
    st = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cats]
    st.append(OneHotEncoder(inputCols=[f"{c}_idx" for c in cats], outputCols=[f"{c}_ohe" for c in cats], handleInvalid="keep"))
    impute = [c for c in player_feats if not c.startswith("miss_")]
    if impute:
        st.append(Imputer(inputCols=impute, outputCols=impute, strategy="median"))   # fitted on the fitting folds only
    out = "features_raw" if scale else "features"
    st.append(VectorAssembler(inputCols=[f"{c}_ohe" for c in cats] + nums + list(player_feats), outputCol=out, handleInvalid="keep"))
    if scale:
        st.append(StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True))
    st.append(estimator)
    return Pipeline(stages=st)

def ladder(task, kind):
    clf = kind == "classification"
    common = dict(featuresCol="features", labelCol="target", weightCol="weight")
    lin = lambda: (LogisticRegression(maxIter=200, **common) if clf else LinearRegression(maxIter=200, **common))
    rf = lambda: (RandomForestClassifier if clf else RandomForestRegressor)(seed=C.SEED, subsamplingRate=0.8,
                                                                             featureSubsetStrategy="sqrt", **common)
    gbt = lambda: (GBTClassifier if clf else GBTRegressor)(seed=C.SEED, subsamplingRate=0.8, **common)
    full = FS["trk"] + FS["sw"] + FS["size"]
    lin_specs = [("benchmark", []), ("first_step", ["trk_first_step"]), ("lateral", ["trk_lateral"]),
                 ("burst", ["trk_burst"]), ("traits", FS["trk"]), ("stopwatch", FS["sw"]), ("traits_size", full)]
    out = []
    for name, feats in lin_specs:
        e = lin()
        out.append((name, pipeline(task, e, feats, True), {e.regParam: [0.0, 0.01]}))
    e = rf();  out.append(("rf", pipeline(task, e, full, False), {e.numTrees: [100], e.maxDepth: [4, 6], e.minInstancesPerNode: [500, 2000]}))
    e = gbt(); out.append(("gbt", pipeline(task, e, full, False), {e.maxIter: [40], e.maxDepth: [2, 3], e.minInstancesPerNode: [500, 2000], e.stepSize: [0.1]}))
    return out

def grid(spec):
    g = ParamGridBuilder()
    for p, vals in spec.items():
        g = g.addGrid(p, vals[:1] if QUICK else vals)
    return g.build()

# %% [markdown]
# ## 2. Cross-validated training, task by task

# %%
cv_rows, summary = [], []
# when re-running only some tasks, keep the other tasks' earlier results
if "TASKS" in os.environ and (C.TABLE_DIR / "06_cv_summary.csv").exists():
    cv_rows = [r for r in pd.read_csv(C.TABLE_DIR / "06_cv_results.csv").to_dict("records") if r["task"] not in TASKS]
    summary = [r for r in pd.read_csv(C.TABLE_DIR / "06_cv_summary.csv").to_dict("records") if r["task"] not in TASKS]
for task in TASKS:
    kind = FS["tasks"][task]["kind"]
    train = spark.read.parquet(C.p(f"task_{task}.parquet")).filter("split = 'train'").cache()
    n, n_pl = train.count(), train.select("nfl_id").distinct().count()
    ev = (BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC") if kind == "classification"
          else RegressionEvaluator(labelCol="target", metricName="r2"))
    print(f"\n=== {task} ({kind}): {n:,} training snaps, {n_pl} players")
    for name, pipe, spec in ladder(task, kind):
        pg = grid(spec); t0 = time.time()
        cvm = CrossValidator(estimator=pipe, estimatorParamMaps=pg, evaluator=ev, numFolds=C.N_FOLDS,
                             foldCol="fold", parallelism=PAR, seed=C.SEED).fit(train)
        for pm, m, s in zip(pg, cvm.avgMetrics, cvm.stdMetrics):
            cv_rows.append({"task": task, "model": name, **{p.name: v for p, v in pm.items()}, "cv_mean": m, "cv_std": s})
        bi = max(range(len(pg)), key=lambda i: cvm.avgMetrics[i])
        best = {p.name: v for p, v in pg[bi].items()}
        summary.append({"task": task, "kind": kind, "model": name, "settings": len(pg), "best_params": json.dumps(best),
                        "cv_mean": round(cvm.avgMetrics[bi], 4), "cv_std": round(cvm.stdMetrics[bi], 4),
                        "seconds": round(time.time() - t0, 1)})
        cvm.bestModel.write().overwrite().save(str(C.MODEL_DIR / task / name))
        print(f"  {name:<12} CV {'AUROC' if kind == 'classification' else 'R²'} {cvm.avgMetrics[bi]:.4f} ± {cvm.stdMetrics[bi]:.4f}  {best}  ({time.time() - t0:,.0f} s)", flush=True)
    train.unpersist()
    # save after every task so a long run can be resumed with TASKS=...
    pd.DataFrame(cv_rows).to_csv(C.TABLE_DIR / "06_cv_results.csv", index=False)
    pd.DataFrame(summary).to_csv(C.TABLE_DIR / "06_cv_summary.csv", index=False)

# %% [markdown]
# ## 3. Cross-validation summary

# %%
sm = pd.DataFrame(summary)
sm.pivot_table(index="model", columns="task", values="cv_mean").reindex(
    ["benchmark", "first_step", "lateral", "burst", "traits", "stopwatch", "traits_size", "rf", "gbt"]).round(4)
