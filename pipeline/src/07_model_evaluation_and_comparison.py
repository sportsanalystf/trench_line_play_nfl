# %% [markdown]
# # 07 · Model evaluation: which trait matters for which outcome?
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# 1. Scores every tuned model from notebook 06 on **validation** (2023–24 players held out) and **test** (the whole 2025 class, never used for training or tuning).
# 2. **Metrics.** Classification: AUROC with a 95% **player-bootstrap** CI, AUPR, and accuracy / precision / recall / F1 at a threshold chosen on validation (max F1), plus top-decile lift. Regression: R², RMSE, MAE at the snap level, and R² of player averages (30+ snaps).
# 3. **Ceiling ("oracle").** Context plus each player's *actual* average outcome on the scored split. It leaks on purpose and shows the most any player trait could add.
# 4. **Trait effects (the research question).** For first-step quickness, lateral agility and burst, separately and together, plus the stopwatch equivalent: the change in test metric over the context-only benchmark, with a **paired player-bootstrap 95% CI**. Also the **direction** of each trait's standardized coefficient, oriented so that + means "better trait → better line play".
# 5. **Champion** per task, chosen on validation: confusion matrix (classification), hyperparameter **sensitivity**, and feature-importance shares by block.
# 6. **Player-level ranking**: do the models order the 2025 rookies correctly?
#
# **Inputs:** `processed/task_*.parquet`, `outputs/models/*`, `outputs/tables/06_*.csv`.  **Outputs:** `outputs/tables/07_*.csv`, `outputs/figures/07_*.png`.

# %%
import sys, json, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
import config as C

spark = C.get_spark("07_eval")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
FS = json.load(open(C.OUTPUT_DIR / "feature_sets.json"))
cv_sum = pd.read_csv(C.TABLE_DIR / "06_cv_summary.csv")
cv_res = pd.read_csv(C.TABLE_DIR / "06_cv_results.csv")
TASKS = [t for t in FS["tasks"] if t in set(cv_sum.task)]
MODELS = ["benchmark", "first_step", "lateral", "burst", "traits", "stopwatch", "traits_size", "rf", "gbt"]
LABEL = {"benchmark": "Benchmark (context only)", "first_step": "+ first-step quickness", "lateral": "+ lateral agility",
         "burst": "+ burst", "traits": "+ all three traits", "stopwatch": "+ stopwatch equivalents",
         "traits_size": "+ traits, stopwatch, size", "rf": "Random forest (all)", "gbt": "GBT (all)", "oracle": "Ceiling: + player's true average"}
TASK_LABEL = {"dl_pressure": "DL pass rush: pressure", "dl_getoff": "DL pass rush: get-off time",
              "ol_pressure": "OL pass block: pressure allowed", "ol_pressure_prob": "OL pass block: peak pressure prob.",
              "ol_run": "OL run block: run success at his gap"}
# + = better line play: DL want pressure (+) and a low get-off; OL want low pressure; run success (+)
GOOD_DIR = {"dl_pressure": 1, "dl_getoff": -1, "ol_pressure": -1, "ol_pressure_prob": -1, "ol_run": 1}
B = int(os.environ.get("N_BOOT", "300"))

# %% [markdown]
# ## 1. Score every model on validation and test
# All predictions go into one Spark table, `eval_predictions` (task, model, split, row, player, target, prediction), so every metric below is a Spark aggregation over it.

# %%
row_id = F.xxhash64("game_id", "play_id", "nfl_id")            # stable row key (tie-break for orderings)
def scored(model, df, kind, task, name, split):
    out = model.transform(df)
    pred = vector_to_array("probability")[1] if kind == "classification" else F.col("prediction")
    return out.select(F.lit(task).alias("task"), F.lit(kind).alias("kind"), F.lit(name).alias("model"), F.lit(split).alias("split"),
                      row_id.alias("row_id"), "nfl_id", F.col("target").cast("double").alias("target"), pred.alias("p"))

def oracle_pipe(task, kind):
    cats = FS["context_cat"] + (FS["run_cat"] if task == "ol_run" else [])
    nums = FS["context_num"] + (FS["run_num"] if task == "ol_run" else []) + ["player_mean"]
    est = (LogisticRegression(featuresCol="features", labelCol="target", weightCol="weight", regParam=0.01) if kind == "classification"
           else LinearRegression(featuresCol="features", labelCol="target", weightCol="weight", regParam=0.01))
    return Pipeline(stages=[StringIndexer(inputCol=c, outputCol=f"{c}_i", handleInvalid="keep") for c in cats] + [
        OneHotEncoder(inputCols=[f"{c}_i" for c in cats], outputCols=[f"{c}_o" for c in cats], handleInvalid="keep"),
        VectorAssembler(inputCols=[f"{c}_o" for c in cats] + nums, outputCol="fr", handleInvalid="keep"),
        StandardScaler(inputCol="fr", outputCol="features", withMean=True, withStd=True), est])
with_mean = lambda df: df.withColumn("player_mean", F.avg("target").over(Window.partitionBy("nfl_id")))

data, parts = {}, []
for task in TASKS:
    kind = FS["tasks"][task]["kind"]
    d = spark.read.parquet(C.p(f"task_{task}.parquet")); data[task] = d
    for split in ("validation", "test"):
        part = d.filter(F.col("split") == split)
        for m in MODELS:
            if (C.MODEL_DIR / task / m).exists():
                parts.append(scored(PipelineModel.load(str(C.MODEL_DIR / task / m)), part, kind, task, m, split))
    # ceiling ("oracle"): context + the player's actual average outcome on the split being scored (leaks on purpose)
    om = oracle_pipe(task, kind).fit(with_mean(d.filter("split = 'train'")))
    for split in ("validation", "test"):
        parts.append(scored(om, with_mean(d.filter(F.col("split") == split)), kind, task, "oracle", split))
    print(task, "scored")
P = parts[0]
for p_ in parts[1:]:
    P = P.unionByName(p_)
P.write.mode("overwrite").partitionBy("task").parquet(C.p("eval_predictions.parquet"))
P = spark.read.parquet(C.p("eval_predictions.parquet")).cache()
print(f"{P.count():,} predictions in eval_predictions")

# %% [markdown]
# ## 2. Metrics (Spark aggregations over `eval_predictions`)
# * **AUROC** = Mann–Whitney statistic on average ranks of the prediction; **AUPR** = average precision over the ranking.
# * **Threshold** for accuracy / precision / recall / F1 = the validation F1-maximizing one among the 30th–98th percentiles of the validation predictions; then applied to test.
# * **Lift** = positive rate in the top 10% of predictions ÷ base rate. **R², RMSE, MAE** per snap; **player R²** on player averages (30+ snaps).

# %%
G = ["task", "kind", "model", "split"]
def auroc(df, by):
    r = df.withColumn("_r", C.avg_rank("p", by))
    a = r.groupBy(*by).agg(F.sum(F.col("_r") * F.col("target")).alias("sr"), F.sum("target").alias("n1"), F.count(F.lit(1)).alias("n"))
    return a.select(*by, ((F.col("sr") - F.col("n1") * (F.col("n1") + 1) / 2) / (F.col("n1") * (F.col("n") - F.col("n1")))).alias("auroc"))
def r2(df, by, w=None):
    w = F.lit(1.0) if w is None else F.col(w)
    a = df.groupBy(*by).agg(F.sum(w * (F.col("target") - F.col("p")) ** 2).alias("sse"), F.sum(w * F.col("target")).alias("sy"),
                            F.sum(w * F.col("target") ** 2).alias("syy"), F.sum(w).alias("sw"))
    return a.select(*by, (1 - F.col("sse") / (F.col("syy") - F.col("sy") ** 2 / F.col("sw"))).alias("r2"))

clf, reg = P.filter(F.col("kind") == "classification"), P.filter(F.col("kind") == "regression")
desc_w = Window.partitionBy(*G).orderBy(F.desc("p"), "row_id")
ranked = (clf.withColumn("_k", F.row_number().over(desc_w))
             .withColumn("_tp", F.sum("target").over(desc_w.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
             .withColumn("_n", F.count(F.lit(1)).over(Window.partitionBy(*G))))
aupr = ranked.groupBy(*G).agg((F.sum(F.col("target") * F.col("_tp") / F.col("_k")) / F.sum("target")).alias("aupr"))
lift = ranked.groupBy(*G).agg((F.avg(F.when(F.col("_k") <= F.round(0.1 * F.col("_n")), F.col("target"))) / F.avg("target")).alias("lift_top10"),
                              F.avg("target").alias("base_rate"))
# validation threshold that maximizes F1, then the confusion counts on test
grid = [round(0.30 + 0.01 * i, 2) for i in range(69)]
thr = (clf.filter(F.col("split") == "validation").groupBy("task", "model")
          .agg(F.percentile("p", grid).alias("t")).select("task", "model", F.explode("t").alias("threshold")))
def confusion(df, t):
    yh = (F.col("p") >= F.col("threshold")).cast("int")
    return (df.join(t, ["task", "model"]).groupBy(*G, "threshold")
              .agg(F.sum(yh * F.col("target")).alias("tp"), F.sum(yh * (1 - F.col("target"))).alias("fp"),
                   F.sum((1 - yh) * F.col("target")).alias("fn"), F.sum((1 - yh) * (1 - F.col("target"))).alias("tn"))
              .withColumn("precision", F.coalesce(F.col("tp") / (F.col("tp") + F.col("fp")), F.lit(0.0)))
              .withColumn("recall", F.coalesce(F.col("tp") / (F.col("tp") + F.col("fn")), F.lit(0.0)))
              .withColumn("f1", F.coalesce(2 * F.col("precision") * F.col("recall") / (F.col("precision") + F.col("recall")), F.lit(0.0)))
              .withColumn("accuracy", (F.col("tp") + F.col("tn")) / (F.col("tp") + F.col("fp") + F.col("fn") + F.col("tn"))))
best = (confusion(clf.filter(F.col("split") == "validation"), thr)
          .withColumn("_o", F.row_number().over(Window.partitionBy("task", "model").orderBy(F.desc("f1"), "threshold")))
          .filter("_o = 1").select("task", "model", "threshold"))
at_test = confusion(clf.filter(F.col("split") == "test"), best).drop("split", "kind")
# regression extras
reg_extra = reg.groupBy(*G).agg(F.sqrt(F.avg((F.col("target") - F.col("p")) ** 2)).alias("rmse"), F.avg(F.abs(F.col("target") - F.col("p"))).alias("mae"))
pm = (reg.groupBy(*G, "nfl_id").agg(F.count(F.lit(1)).alias("n"), F.avg("target").alias("target"), F.avg("p").alias("p"))
         .filter("n >= 30"))
player_r2 = r2(pm, G).withColumnRenamed("r2", "player_r2")
main = (auroc(clf, G).withColumnRenamed("auroc", "metric")
        .unionByName(r2(reg, G).withColumnRenamed("r2", "metric")))

# %% [markdown]
# ### Player-bootstrap confidence intervals (Poisson bootstrap, in Spark)
# Each of `N_BOOT` draws gives every test player a weight ~ Poisson(1): how many times he is "resampled". (This is the standard way to bootstrap at scale: it needs no shuffle of the data, only a weight per player per draw, and it approximates resampling players with replacement.) The weight comes from a seeded hash, so draws are reproducible. Every model is scored on the **same** draws, so the differences against the benchmark (section 3) are paired.

# %%
POIS = [0.36787944, 0.73575888, 0.91969860, 0.98101184, 0.99634015, 0.99940582, 0.99991676, 0.99998975]   # Poisson(1) CDF
test_players = P.filter(F.col("split") == "test").select("task", "nfl_id").distinct()
u = F.abs(F.xxhash64("task", "nfl_id", "b", F.lit(C.SEED)) % 1_000_000_007).cast("double") / 1_000_000_007   # uniform in [0, 1)
k = sum(F.when(u > c, 1).otherwise(0) for c in POIS)          # inverse-CDF draw of a Poisson(1) count
draws = (spark.range(B).withColumnRenamed("id", "b").crossJoin(test_players)
              .withColumn("w", k.cast("double")).filter(F.col("w") > 0))
T = P.filter(F.col("split") == "test").join(draws, ["task", "nfl_id"])
# weighted AUROC per draw: for each prediction value, positives x (negatives below + half the ties)
Bg = ["task", "model", "b"]
byp = (T.filter(F.col("kind") == "classification").groupBy(*Bg, "p")
        .agg(F.sum(F.col("w") * F.col("target")).alias("wpos"), F.sum(F.col("w") * (1 - F.col("target"))).alias("wneg")))
below = Window.partitionBy(*Bg).orderBy("p").rowsBetween(Window.unboundedPreceding, -1)
boot_auc = (byp.withColumn("negbelow", F.coalesce(F.sum("wneg").over(below), F.lit(0.0)))
               .groupBy(*Bg).agg((F.sum(F.col("wpos") * (F.col("negbelow") + 0.5 * F.col("wneg"))) /
                                  (F.sum("wpos") * F.sum("wneg"))).alias("metric")))
boot_r2 = r2(T.filter(F.col("kind") == "regression"), Bg, w="w").withColumnRenamed("r2", "metric")
boot = boot_auc.unionByName(boot_r2)
boot.write.mode("overwrite").parquet(C.p("eval_bootstrap.parquet"))
boot = spark.read.parquet(C.p("eval_bootstrap.parquet")).cache()
ci = boot.groupBy("task", "model").agg(F.percentile("metric", [0.025, 0.975]).alias("ci"))
print(f"bootstrap: {B} draws, {boot.count():,} (task, model, draw) metrics")

# %%
wide = (main.groupBy("task", "kind", "model").pivot("split", ["validation", "test"]).agg(F.first("metric"))
            .withColumnRenamed("validation", "val"))
cv_sum_s = spark.read.csv(str(C.TABLE_DIR / "06_cv_summary.csv"), header=True, inferSchema=True).select("task", "model", F.col("cv_mean").alias("cv"))
test_only = lambda df: df.filter(F.col("split") == "test").drop("split", "kind")
res_s = (wide.join(cv_sum_s, ["task", "model"], "left")
             .join(ci.select("task", "model", F.col("ci")[0].alias("test_ci_lo"), F.col("ci")[1].alias("test_ci_hi")), ["task", "model"], "left")
             .join(test_only(aupr), ["task", "model"], "left").join(test_only(lift), ["task", "model"], "left")
             .join(at_test.select("task", "model", "threshold", "accuracy", "precision", "recall", "f1"), ["task", "model"], "left")
             .join(test_only(reg_extra), ["task", "model"], "left").join(test_only(player_r2), ["task", "model"], "left"))
res = res_s.toPandas()
res["label"] = res.model.map(LABEL)
res["_o"] = res.model.map({m: i for i, m in enumerate(MODELS + ["oracle"])})
res = res.sort_values(["task", "_o"]).drop(columns="_o")[
    ["task", "kind", "model", "label", "cv", "val", "test", "test_ci_lo", "test_ci_hi", "aupr", "threshold", "accuracy", "precision",
     "recall", "f1", "lift_top10", "base_rate", "rmse", "mae", "player_r2"]]
res.round(4).to_csv(C.TABLE_DIR / "07_model_metrics.csv", index=False)
print("test metric = AUROC (classification) or R² (regression)")
res.pivot_table(index="model", columns="task", values="test").reindex(MODELS + ["oracle"]).round(3)

# %% [markdown]
# ## 3. Trait effects: change over the benchmark, paired player bootstrap
# Both models are scored on the **same** bootstrap draws, so the interval reflects the difference itself.

# %%
EFFECT_MODELS = ["first_step", "lateral", "burst", "traits", "stopwatch"]
point = main.filter(F.col("split") == "test").select("task", "model", "metric")
bench_pt = point.filter(F.col("model") == "benchmark").select("task", F.col("metric").alias("bench"))
bench_b = boot.filter(F.col("model") == "benchmark").select("task", "b", F.col("metric").alias("bench"))
eff_s = (boot.filter(F.col("model").isin(EFFECT_MODELS)).join(bench_b, ["task", "b"])
             .groupBy("task", "model").agg(F.percentile(F.col("metric") - F.col("bench"), [0.025, 0.975]).alias("ci"))
             .join(point.join(bench_pt, "task").select("task", "model", (F.col("metric") - F.col("bench")).alias("delta")), ["task", "model"])
             .select("task", "model", "delta", F.col("ci")[0].alias("ci_lo"), F.col("ci")[1].alias("ci_hi"))
             .withColumn("significant", (F.col("ci_lo") > 0) | (F.col("ci_hi") < 0)))
eff = eff_s.toPandas()
eff["_o"] = eff.model.map({m: i for i, m in enumerate(EFFECT_MODELS)}); eff = eff.sort_values(["task", "_o"]).drop(columns="_o")
eff.round(4).to_csv(C.TABLE_DIR / "07_trait_effects.csv", index=False)
eff.pivot_table(index="model", columns="task", values="delta").reindex(EFFECT_MODELS).round(4)

# %% [markdown]
# ### Direction of each trait (standardized coefficients in the `traits` model)
# Oriented so that **+ = a better Combine trait goes with better line play** (more pressure for DL, a quicker get-off, less pressure allowed, more run success). Magnitudes are per standard deviation of the trait index (log-odds for classification, outcome units for regression). The coefficients are read from the fitted Spark models.

# %%
coef_rows = []
for task in TASKS:
    mdl = PipelineModel.load(str(C.MODEL_DIR / task / "traits"))
    tr = mdl.transform(data[task].limit(5))
    attrs = tr.schema["features_raw"].metadata["ml_attr"]["attrs"]
    names = [a["name"] for a in sorted([a for g in attrs.values() for a in g], key=lambda a: a["idx"])]
    coefs = mdl.stages[-1].coefficients.toArray()
    for tname in FS["trk"]:
        c_ = float(coefs[names.index(tname)])
        coef_rows.append((task, tname.replace("trk_", ""), c_, GOOD_DIR[task] * c_))
coef = pd.DataFrame(coef_rows, columns=["task", "trait", "coef", "oriented"])
coef.round(4).to_csv(C.TABLE_DIR / "07_trait_directions.csv", index=False)
coef.pivot_table(index="trait", columns="task", values="oriented").round(4)

# %%
# Headline figure: trait effects per task with paired-bootstrap CIs
fig, axes = plt.subplots(1, len(TASKS), figsize=(3.3 * len(TASKS), 4.2), sharey=True)
cols = {"first_step": "#2a6fdb", "lateral": "#2e9e6b", "burst": "#e07b39", "traits": "#14213D", "stopwatch": "#9aa5b1"}
for ax, task in zip(np.atleast_1d(axes), TASKS):
    d = eff[eff.task == task].set_index("model").reindex(EFFECT_MODELS)
    ypos = np.arange(len(d))[::-1]
    ax.errorbar(d.delta, ypos, xerr=[d.delta - d.ci_lo, d.ci_hi - d.delta], fmt="none", ecolor="#555", capsize=3, lw=1)
    ax.scatter(d.delta, ypos, color=[cols[m] for m in d.index], s=50, zorder=3)
    ax.axvline(0, color="#999", ls=":", lw=1)
    ax.set_title(TASK_LABEL[task], fontsize=9)
    ax.set_xlabel("Δ test " + ("AUROC" if FS["tasks"][task]["kind"] == "classification" else "R²") + " vs benchmark")
np.atleast_1d(axes)[0].set_yticks(np.arange(len(EFFECT_MODELS))[::-1], [LABEL[m] for m in EFFECT_MODELS])
fig.suptitle("What each Combine trait adds to game context, on the unseen 2025 class (95% paired player-bootstrap CI)", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "07_trait_effects.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4. Champion per task (chosen on validation), confusion matrices, sensitivity

# %%
cand = res_s.filter(~F.col("model").isin("benchmark", "oracle"))
champ_s = (cand.withColumn("_o", F.row_number().over(Window.partitionBy("task").orderBy(F.desc("val"))))
               .filter("_o = 1").select("task", "model", "val", "test", "test_ci_lo", "test_ci_hi")
               .join(res_s.filter(F.col("model") == "benchmark").select("task", F.col("test").alias("benchmark_test")), "task")
               .join(res_s.filter(F.col("model") == "oracle").select("task", F.col("test").alias("ceiling_test")), "task")
               .withColumn("share_of_possible_gain", (F.col("test") - F.col("benchmark_test")) / (F.col("ceiling_test") - F.col("benchmark_test"))))
champ = champ_s.orderBy("task").toPandas().set_index("task")
champ.round(3).to_csv(C.TABLE_DIR / "07_champions.csv")
champ.round(3)

# %%
clf_tasks = [t for t in TASKS if FS["tasks"][t]["kind"] == "classification"]
cm = (at_test.join(champ_s.select("task", "model"), ["task", "model"])
             .select("task", "model", "threshold", "accuracy", "precision", "recall", "f1",
                     *[F.col(c).cast("long").alias(c) for c in ["tp", "fp", "fn", "tn"]]).toPandas().set_index("task"))
cm.reset_index().to_csv(C.TABLE_DIR / "07_confusion_matrices.csv", index=False)
fig, axes = plt.subplots(1, len(clf_tasks), figsize=(4.2 * len(clf_tasks), 3.6))
for ax, task in zip(np.atleast_1d(axes), clf_tasks):
    k = cm.loc[task]; M = np.array([[k.tn, k.fp], [k.fn, k.tp]])
    ax.imshow(M, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}", ha="center", va="center", color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks([0, 1], ["no", "yes"]); ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{TASK_LABEL[task]}\n{LABEL[k.model]}, threshold {k.threshold:.2f}", fontsize=8)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "07_confusion_matrices.png", bbox_inches="tight"); plt.show()

# %%
# Sensitivity: CV metric across the grid of each task's tree models (depth × min snaps per leaf), from notebook 06
cv_res_s = spark.read.csv(str(C.TABLE_DIR / "06_cv_results.csv"), header=True, inferSchema=True)
grid_s = (cv_res_s.filter(F.col("model").isin("rf", "gbt"))
                  .groupBy("task", "model", "maxDepth").pivot("minInstancesPerNode").agg(F.max("cv_mean")).orderBy("task", "model", "maxDepth"))
grid_pd = grid_s.toPandas()
fig, axes = plt.subplots(2, len(TASKS), figsize=(3.2 * len(TASKS), 5.6))
for c, task in enumerate(TASKS):
    for r_, m in enumerate(["rf", "gbt"]):
        ax = axes[r_, c]
        hm = grid_pd[(grid_pd.task == task) & (grid_pd.model == m)].drop(columns=["task", "model"]).set_index("maxDepth").dropna(axis=1, how="all")
        if hm.empty:
            ax.axis("off"); continue
        ax.imshow(hm.values, cmap="viridis", aspect="auto")
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                ax.text(j, i, f"{hm.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        ax.set_xticks(range(hm.shape[1]), [f"{float(v):g}" for v in hm.columns]); ax.set_yticks(range(hm.shape[0]), [f"{v:g}" for v in hm.index])
        ax.set_xlabel("min snaps per leaf", fontsize=8); ax.set_ylabel("max depth", fontsize=8)
        ax.set_title(f"{task} · {m}", fontsize=8)
fig.suptitle("Sensitivity of the tree models to their hyperparameters (CV metric)", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "07_sensitivity.png", bbox_inches="tight"); plt.show()
sens = (cv_res_s.groupBy("task", "model").agg(F.round(F.min("cv_mean"), 4).alias("min"), F.round(F.max("cv_mean"), 4).alias("max"))
                .orderBy("task", "model").toPandas())
sens.to_csv(C.TABLE_DIR / "07_sensitivity_ranges.csv", index=False); sens

# %% [markdown]
# ## 5. Feature importance by block (GBT models)

# %%
def block(n):
    if n.startswith("trk_"): return "tracking traits"
    if n.startswith("sw_") or n.startswith("miss_sw"): return "stopwatch"
    if n in FS["size"]: return "size"
    return "context"
imp_rows = []
for task in TASKS:
    mdl = PipelineModel.load(str(C.MODEL_DIR / task / "gbt"))
    attrs = mdl.transform(data[task].limit(5)).schema["features"].metadata["ml_attr"]["attrs"]
    names = [a["name"] for a in sorted([a for g in attrs.values() for a in g], key=lambda a: a["idx"])]
    imp = mdl.stages[-1].featureImportances.toArray()
    imp_rows += [(task, n_, block(n_), float(v)) for n_, v in zip(names, imp)]
imp_s = spark.createDataFrame(imp_rows, "task string, feature string, block string, importance double")
imp_s.toPandas().to_csv(C.TABLE_DIR / "07_gbt_feature_importance.csv", index=False)
shares = (imp_s.groupBy("task").pivot("block", ["context", "tracking traits", "stopwatch", "size"]).agg(F.sum("importance")).na.fill(0.0)
               .withColumn("_t", sum(F.col(f"`{b}`") for b in ["context", "tracking traits", "stopwatch", "size"])))
for b in ["context", "tracking traits", "stopwatch", "size"]:
    shares = shares.withColumn(b, F.col(f"`{b}`") / F.col("_t"))
shares = shares.drop("_t").orderBy("task").toPandas().set_index("task")
fig, ax = plt.subplots(figsize=(8, 3.4))
left = np.zeros(len(shares)); bc = {"context": "#9aa5b1", "tracking traits": "#2a6fdb", "stopwatch": "#e07b39", "size": "#7b61c9"}
for b in ["context", "tracking traits", "stopwatch", "size"]:
    ax.barh([TASK_LABEL[t] for t in shares.index], shares[b], left=left, color=bc[b], label=b); left += shares[b].values
ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.set_xlabel("Share of GBT importance")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "07_importance_blocks.png", bbox_inches="tight"); plt.show()
shares.round(3)

# %% [markdown]
# ## 6. Player-level ranking of the 2025 rookies
# Spearman correlation (in Spark) between a player's mean prediction and his actual average outcome (players with 30+ test snaps).

# %%
pl_means = (P.filter((F.col("split") == "test") & F.col("model").isin("benchmark", "traits", "stopwatch", "traits_size", "gbt", "oracle"))
              .groupBy("task", "model", "nfl_id").agg(F.count(F.lit(1)).alias("n"), F.avg("target").alias("y"), F.avg("p").alias("p"))
              .filter("n >= 30"))
pr = (C.spearman(pl_means, "y", "p", by=["task", "model"]).withColumnRenamed("n", "players")
        .orderBy("task", "model").toPandas())
pr.round(3).to_csv(C.TABLE_DIR / "07_player_ranking.csv", index=False)
pr.pivot_table(index="model", columns="task", values="spearman").round(3)
