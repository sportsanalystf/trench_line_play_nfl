# %% [markdown]
# # 03 · Combine trait features: first-step quickness, lateral agility, burst
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# Turns 10 Hz Combine drill tracking into three movement traits for every OL and DL prospect:
#
# | Trait | Per-attempt metrics (from tracking) | Drills used (see `config.TRAIT_DRILLS`) |
# |---|---|---|
# | **First-step quickness** | `t_1yd`, `t_2yd` = time from first movement to 1 / 2 yd away; `s03`, `s05` = speed 0.3 / 0.5 s after first movement | OL: pass-set drop, pull drills, 40 · DL: pass-rush, wave start, run-and-club, 40 |
# | **Lateral agility** | at every direction **reversal**: `rev_time` (time spent below 70% of entry speed), `rev_brake` (peak braking into it), `rev_reacc` (peak re-acceleration out of it); `move_speed` (mean speed while moving) | OL: 5-yd wave slide-and-shuffle, pass-pro mirror · DL: wave lateral reaction, four-bag, body control · both: tracked shuttle / 3-cone |
# | **Burst** | `acc_peak` = peak signed acceleration in the first 1.5 s; `s10` = speed at 1.0 s; `d10` = ground covered in the first 1.0 s | OL: pull drills, screen, 40 · DL: pass-rush, run-and-club, hoop, 40 |
#
# Steps:
# 1. Frame-level kinematics with Spark window functions (smoothed speed, **signed** acceleration, displacement from the start point, reversal detection).
# 2. Attempt-level metrics → each player's **best attempt** per drill.
# 3. **Trait indices**: within each position group, every drill × metric is z-scored (oriented so higher = better) using **2023–24 players only** (the 2025 test class never informs the scaling). Each trait index = mean of the player's available z-scores.
# 4. Matching **stopwatch indices** from 10-yard split (first step), shuttle + 3-cone (lateral), broad + vertical + 40 (burst), and **size-adjusted** trait indices (residual after body weight).
#
# **Inputs:** `processed/combine_tracking.parquet`, `processed/line_players.parquet`
# **Outputs:** `processed/trait_attempts.parquet`, `processed/trait_drill_best.parquet`, `processed/trait_indices.parquet`, `outputs/tables/03_*.csv`, `outputs/figures/03_*.png`

# %%
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import config as C

spark = C.get_spark("03_traits")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
T0 = time.time()
players = spark.read.parquet(C.p("line_players.parquet"))
ct = spark.read.parquet(C.p("combine_tracking.parquet"))
used = sorted({d for g in C.TRAIT_DRILLS.values() for ds in g.values() for d in ds})
ct = ct.filter(F.col("drill_name").isin(used))
print(f"frames in trait drills: {ct.count():,}")

# %% [markdown]
# ## 1. Frame-level kinematics
# Within each attempt (`event_id`, ordered by time):
# * `s_sm` = 3-frame centred mean of speed; `a_tan` = centred difference of `s_sm` (**signed**: + speeding up, − braking; the raw `a` column is an unsigned magnitude).
# * **Onset** = first frame with `s_sm` > 0.5 yd/s; `t_rel` = seconds since onset; `disp` = straight-line distance from the onset position.
# * **Reversal**: the path over the previous 0.4 s and the next 0.4 s both exceed 0.3 yd and point more than 120° apart. Consecutive flagged frames are one reversal, anchored at its slowest frame.

# %%
ONSET, K = 0.5, 4                     # onset speed (yd/s); look-back/look-ahead frames for reversals
w = Window.partitionBy("event_id").orderBy("ts")
w3 = w.rowsBetween(-1, 1)
k = (ct
     .withColumn("s_sm", F.avg("s").over(w3))
     .withColumn("dt2", F.lead(F.col("ts").cast("double")).over(w) - F.lag(F.col("ts").cast("double")).over(w))
     .withColumn("a_tan", (F.lead("s_sm").over(w) - F.lag("s_sm").over(w)) / F.col("dt2"))
     .withColumn("ux1", F.col("x") - F.lag("x", K).over(w)).withColumn("uy1", F.col("y") - F.lag("y", K).over(w))
     .withColumn("ux2", F.lead("x", K).over(w) - F.col("x")).withColumn("uy2", F.lead("y", K).over(w) - F.col("y"))
     .withColumn("n1", F.sqrt(F.col("ux1")**2 + F.col("uy1")**2)).withColumn("n2", F.sqrt(F.col("ux2")**2 + F.col("uy2")**2))
     .withColumn("cos12", (F.col("ux1") * F.col("ux2") + F.col("uy1") * F.col("uy2")) / (F.col("n1") * F.col("n2")))
     .withColumn("rev_flag", ((F.col("n1") > 0.3) & (F.col("n2") > 0.3) & (F.col("cos12") < -0.5)).cast("int")))
onset = (k.filter(F.col("s_sm") > ONSET).withColumn("r", F.row_number().over(w)).filter("r = 1")
          .select("event_id", F.col("t_sec").alias("t_on"), F.col("x").alias("x_on"), F.col("y").alias("y_on")))
k = (k.join(onset, "event_id", "inner")
      .withColumn("t_rel", F.round(F.col("t_sec") - F.col("t_on"), 2))
      .withColumn("disp", F.sqrt((F.col("x") - F.col("x_on"))**2 + (F.col("y") - F.col("y_on"))**2)))
# reversal apex = slowest frame within each run of flagged frames
wr = w.rowsBetween(-3, 3)
k = (k.withColumn("min_s_flag", F.min(F.when(F.col("rev_flag") == 1, F.col("s_sm"))).over(wr))
      .withColumn("apex", ((F.col("rev_flag") == 1) & (F.col("s_sm") == F.col("min_s_flag"))).cast("int")))
k = k.select("event_id", "nfl_id", "line_group", "draft_year", "drill_name", "attempt", "ts", "t_sec", "t_rel",
             "s_sm", "a_tan", "step_xy", "disp", "apex")
k.cache()
print(f"frames with kinematics: {k.count():,}; reversal apexes: {k.filter('apex = 1').count():,}")

# %% [markdown]
# ## 2. Attempt-level metrics
# Reversal metrics use time-range windows around each apex (±1 s); an attempt's value is the median over its reversals.

# %%
# rangeBetween needs integer bounds, so time is ordered in tenths of a second (frames are 0.1 s apart)
k = k.withColumn("t_ds", F.round(F.col("t_sec") * 10).cast("long"))
wt = Window.partitionBy("event_id").orderBy("t_ds")
rv = (k.withColumn("entry_peak", F.max("s_sm").over(wt.rangeBetween(-10, 0)))
       .withColumn("rev_brake", F.min("a_tan").over(wt.rangeBetween(-6, 0)))
       .withColumn("rev_reacc", F.max("a_tan").over(wt.rangeBetween(0, 6))))
# time below 70% of the entry speed within ±1 s of the apex (frames × 0.1 s)
slow = (k.alias("f").join(rv.filter("apex = 1").select("event_id", F.col("t_sec").alias("t_apex"), "entry_peak",
                                                     "rev_brake", "rev_reacc").alias("a"), "event_id")
         .filter((F.col("f.t_sec") >= F.col("a.t_apex") - 1.0) & (F.col("f.t_sec") <= F.col("a.t_apex") + 1.0))
         .groupBy("event_id", "t_apex", "entry_peak", "rev_brake", "rev_reacc")
         .agg((F.sum((F.col("f.s_sm") < 0.7 * F.col("a.entry_peak")).cast("int")) * 0.1).alias("rev_time"))
         .filter(F.col("entry_peak") > 1.0))          # ignore "reversals" from a near standstill
rev_att = (slow.groupBy("event_id").agg(F.expr("percentile_approx(rev_time, 0.5)").alias("rev_time"),
                                         F.expr("percentile_approx(rev_brake, 0.5)").alias("rev_brake"),
                                         F.expr("percentile_approx(rev_reacc, 0.5)").alias("rev_reacc"),
                                         F.count("*").alias("n_reversals")))

def speed_at(t):
    return F.max(F.when((F.col("t_rel") >= t - 0.05) & (F.col("t_rel") < t + 0.05), F.col("s_sm")))

att = (k.groupBy("event_id", "nfl_id", "line_group", "draft_year", "drill_name", "attempt")
       .agg(F.min(F.when((F.col("t_rel") >= 0) & (F.col("disp") >= 1.0), F.col("t_rel"))).alias("t_1yd"),
            F.min(F.when((F.col("t_rel") >= 0) & (F.col("disp") >= 2.0), F.col("t_rel"))).alias("t_2yd"),
            speed_at(0.3).alias("s03"), speed_at(0.5).alias("s05"), speed_at(1.0).alias("s10"),
            F.sum(F.when((F.col("t_rel") > 0) & (F.col("t_rel") <= 1.0), F.col("step_xy"))).alias("d10"),
            F.max(F.when((F.col("t_rel") >= 0) & (F.col("t_rel") <= 1.5), F.col("a_tan"))).alias("acc_peak"),
            F.avg(F.when(F.col("s_sm") > ONSET, F.col("s_sm"))).alias("move_speed"),
            F.count("*").alias("n_frames"))
       .join(rev_att, "event_id", "left"))
att.write.mode("overwrite").parquet(C.p("trait_attempts.parquet"))
att = spark.read.parquet(C.p("trait_attempts.parquet"))
summ = (att.groupBy("line_group", "drill_name").agg(F.count("*").alias("attempts"), F.countDistinct("nfl_id").alias("players"),
        *[F.round(F.expr(f"percentile_approx({m}, 0.5)"), 2).alias(m) for m in
          ["t_1yd", "s05", "acc_peak", "s10", "rev_time", "rev_reacc", "move_speed", "n_reversals"]])
        .orderBy("line_group", "drill_name").toPandas())
summ.to_csv(C.TABLE_DIR / "03_attempt_metric_medians.csv", index=False)
summ

# %% [markdown]
# ## 3. Best attempt per player × drill × metric
# Best = minimum for "lower is better" metrics (`t_1yd`, `t_2yd`, `rev_time`, `rev_brake`), maximum otherwise. Only the drill × metric pairs assigned to a trait in `config.py` are kept.

# %%
metric_sign = {m: s for tr in C.TRAIT_METRICS.values() for m, s in tr.items()}
aggs = [(F.min(m) if s < 0 else F.max(m)).alias(m) for m, s in metric_sign.items()]
best = att.groupBy("nfl_id", "line_group", "draft_year", "drill_name").agg(*aggs, F.count("*").alias("attempts"))
best.write.mode("overwrite").parquet(C.p("trait_drill_best.parquet"))

rows = []
for grp, traits in C.TRAIT_DRILLS.items():
    for trait, drills in traits.items():
        for d in drills:
            for m, s in C.TRAIT_METRICS[trait].items():
                rows.append((grp, trait, d, m, s))
comp = pd.DataFrame(rows, columns=["line_group", "trait", "drill_name", "metric", "sign"])
comp["component"] = comp.trait + "|" + comp.drill_name + "|" + comp.metric
bp = best.toPandas()
long = bp.melt(id_vars=["nfl_id", "line_group", "draft_year", "drill_name"], value_vars=list(metric_sign),
               var_name="metric", value_name="value").dropna()
long = long.merge(comp, on=["line_group", "drill_name", "metric"])
print("trait components with data:", long.component.nunique())

# %% [markdown]
# ## 4. Trait indices (z-scores fitted on 2023–24 players)
# A component needs values for at least 60% of its group's players to count. Each player's trait index is the mean of his available oriented z-scores, so a prospect who skipped a drill is still scored from the others.

# %%
fit = long[long.draft_year != C.TEST_DRAFT_CLASS]
stats = fit.groupby(["line_group", "component"]).value.agg(["mean", "std", "count"]).reset_index()
n_players = bp[bp.draft_year != C.TEST_DRAFT_CLASS].groupby("line_group").nfl_id.nunique().rename("n_group")
stats = stats.merge(n_players, on="line_group")
stats["coverage"] = stats["count"] / stats.n_group
keep = stats[stats.coverage >= 0.6]
print(f"components kept: {len(keep)} of {len(stats)}")
z = long.merge(keep[["line_group", "component", "mean", "std"]], on=["line_group", "component"])
z["z"] = z.sign * (z.value - z["mean"]) / z["std"]
z["z"] = z.z.clip(-3, 3)                                   # winsorize extreme z-scores
idx = z.pivot_table(index=["nfl_id", "line_group"], columns="trait", values="z", aggfunc="mean")
idx.columns = [f"trk_{c}" for c in idx.columns]
ncomp = z.groupby(["nfl_id", "line_group", "trait"]).z.size().unstack()
ncomp.columns = [f"n_{c}" for c in ncomp.columns]
idx = idx.join(ncomp).reset_index()
# component-level wide table too (used for the drill-level correlation table)
comp_wide = z.pivot_table(index=["nfl_id", "line_group"], columns="component", values="z").reset_index()
keep.to_csv(C.TABLE_DIR / "03_trait_components.csv", index=False)
keep.groupby(["line_group", "component"]).coverage.first().unstack(0).round(2).head(40)

# %% [markdown]
# ## 5. Stopwatch indices and size-adjusted trait indices
# * **Stopwatch indices**: the same z-score recipe applied to the traditional drill each trait is usually judged by.
# * **Size-adjusted** (`adj_*`): the residual of each tracking trait index after a linear fit on body weight within the position group (fitted on 2023–24 players). It separates "moves well for his size" from "is light".

# %%
pl = players.select("nfl_id", "line_group", "draft_year", "nfl_position", "combine_weight", "combine_height",
                    "arm_length", "ten_yd_split", "short_shuttle", "three_cone", "broad_jump", "vertical", "forty").toPandas()
for trait, parts in C.TRAD_TRAITS.items():
    zs = []
    for col, s in parts.items():
        ref = pl[pl.draft_year != C.TEST_DRAFT_CLASS].groupby("line_group")[col].agg(["mean", "std"])
        mu = pl.line_group.map(ref["mean"]); sd = pl.line_group.map(ref["std"])
        zs.append((s * (pl[col] - mu) / sd).clip(-3, 3))
    pl[f"sw_{trait}"] = pd.concat(zs, axis=1).mean(axis=1, skipna=True)
out = pl.merge(idx, on=["nfl_id", "line_group"], how="left")
for trait in C.TRAITS:
    out[f"adj_{trait}"] = np.nan
    for g in C.GROUPS:
        m = (out.line_group == g) & out[f"trk_{trait}"].notna() & out.combine_weight.notna()
        f = m & (out.draft_year != C.TEST_DRAFT_CLASS)
        b1, b0 = np.polyfit(out.loc[f, "combine_weight"], out.loc[f, f"trk_{trait}"], 1)
        out.loc[m, f"adj_{trait}"] = out.loc[m, f"trk_{trait}"] - (b0 + b1 * out.loc[m, "combine_weight"])
keep_cols = (["nfl_id", "line_group"] + [f"{p}_{t}" for p in ("trk", "sw", "adj") for t in C.TRAITS] +
             [f"n_{t}" for t in C.TRAITS])
spark.createDataFrame(out[keep_cols].astype({c: float for c in keep_cols if c not in ("nfl_id", "line_group")})) \
     .write.mode("overwrite").parquet(C.p("trait_indices.parquet"))
cov = out.groupby("line_group")[[f"trk_{t}" for t in C.TRAITS] + [f"sw_{t}" for t in C.TRAITS]].apply(lambda d: d.notna().mean()).round(2)
cov.to_csv(C.TABLE_DIR / "03_index_coverage.csv")
print("share of players with each index:"); cov

# %% [markdown]
# ## 6. How the traits relate to each other, to size and to the stopwatch
# Spearman correlations within each position group. High tracking-vs-stopwatch correlation means tracking re-measures the stopwatch; low means it adds a different view.

# %%
rows = []
for g in C.GROUPS:
    d = out[out.line_group == g]
    for t in C.TRAITS:
        for other, lab in [(f"sw_{t}", "same-trait stopwatch index"), ("combine_weight", "body weight")] + \
                          [(f"trk_{o}", f"tracking {o}") for o in C.TRAITS if o != t]:
            dd = d[[f"trk_{t}", other]].dropna()
            rows.append({"group": g, "trait": t, "vs": lab, "n": len(dd),
                         "spearman": round(dd.iloc[:, 0].rank().corr(dd.iloc[:, 1].rank()), 2)})
rel = pd.DataFrame(rows)
rel.to_csv(C.TABLE_DIR / "03_trait_relationships.csv", index=False)
rel.pivot_table(index=["group", "trait"], columns="vs", values="spearman")

# %%
# Figure: trait index distributions by NFL position within each group
fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharey="row")
pos_order = {"OL": ["T", "G", "C"], "DL": ["DE", "OLB", "DT", "NT"]}
for r, g in enumerate(C.GROUPS):
    d = out[out.line_group == g]
    for c, t in enumerate(C.TRAITS):
        ax = axes[r, c]
        cats = [p for p in pos_order[g] if (d.nfl_position == p).sum() >= 3]
        data = [d.loc[d.nfl_position == p, f"trk_{t}"].dropna() for p in cats]
        bp_ = ax.boxplot(data, patch_artist=True, widths=.55, medianprops={"color": "#14213D"})
        ax.set_xticks(range(1, len(cats) + 1), cats)   # works on old and new matplotlib
        for b in bp_["boxes"]:
            b.set_facecolor("#2a6fdb" if g == "OL" else "#e07b39"); b.set_alpha(.55)
        ax.axhline(0, color="#999", lw=.8, ls=":")
        ax.set_title(f"{g} · {t.replace('_', ' ')}", fontsize=10)
    axes[r, 0].set_ylabel("Tracking trait index (z)")
fig.suptitle("Combine trait indices by NFL position (higher = better)", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "03_trait_indices_by_position.png", bbox_inches="tight"); plt.show()

# %%
# Figure: tracking vs stopwatch index per trait (OL and DL)
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for r, g in enumerate(C.GROUPS):
    d = out[out.line_group == g]
    for c, t in enumerate(C.TRAITS):
        ax = axes[r, c]; dd = d[[f"sw_{t}", f"trk_{t}"]].dropna()
        ax.scatter(dd[f"sw_{t}"], dd[f"trk_{t}"], s=16, alpha=.7, color="#2a6fdb" if g == "OL" else "#e07b39")
        rho = dd.iloc[:, 0].rank().corr(dd.iloc[:, 1].rank())
        ax.set_title(f"{g} · {t.replace('_', ' ')}: ρ = {rho:.2f} (n = {len(dd)})", fontsize=9)
        ax.set_xlabel("Stopwatch index"); ax.set_ylabel("Tracking index")
fig.suptitle("Tracking re-measures some stopwatch information, not all of it", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "03_tracking_vs_stopwatch.png", bbox_inches="tight"); plt.show()
print(f"done in {time.time() - T0:,.0f} s")
