# %% [markdown]
# # 05 · Dataset preparation and splitting (five modeling tasks)
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# Builds one modeling table per task, all at the **snap level** (one row = one lineman on one regular-season snap):
#
# | Task | Group | Rows | Target | Type |
# |---|---|---|---|---|
# | `dl_pressure` | DL | pass-rush snaps (dropback, recorded get-off) | pressure (time-to-pressure, hurry or sack) | classification |
# | `dl_getoff` | DL | same snaps | get-off time (s) | regression |
# | `ol_pressure` | OL | pass-block snaps (T/G/C with a pressure-allowed value) | pressure allowed | classification |
# | `ol_pressure_prob` | OL | same snaps | peak pressure probability allowed (0–1) | regression |
# | `ol_run` | OL | run snaps where the lineman is at the point of attack (credit 0.5 or 1, notebook 04) | run success (EPA > 0) | classification |
#
# Features (all known before the snap, or measured at the Combine):
# * **context**: down, distance, field position, clock, score margin and win probability from the player's team's side, formation, receiver alignment, where he lined up, home/away, NFL experience; for `ol_run` also the run gap, his slot (LT … RT) and his credit;
# * **trait indices** from notebook 03: tracking (`trk_*`), stopwatch (`sw_*`) and size-adjusted (`adj_*`) first step, lateral agility and burst;
# * **size**: weight, height, arm length.
#
# **Splits (by player, shared by every task of a group):** test = 2025 class; validation = 20% of 2023–24 players, stratified by draft year and position; train = the rest, with a player-grouped `fold` (0–4). Class weights are computed on the training rows only. Body-size features are winsorized at the 1st / 99th percentile of training players.
#
# **Inputs:** notebooks 01, 03, 04 outputs.  **Outputs:** `processed/task_<name>.parquet`, `outputs/feature_sets.json`, `outputs/tables/05_*.csv`.

# %%
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
from pyspark.sql import functions as F
import config as C

spark = C.get_spark("05_prep")
plays = spark.read.parquet(C.p("line_player_play.parquet")).filter(
    (F.col("season_type") == "REG") & (F.coalesce(F.col("play_nullified_by_penalty"), F.lit("N")) != "Y")
    & (F.col("down") >= 1))          # drops two-point conversions (down 0)
players = spark.read.parquet(C.p("line_players.parquet"))
traits = spark.read.parquet(C.p("trait_indices.parquet"))
ing = spark.read.parquet(C.p("ingame_snaps.parquet")).select("game_id", "play_id", "nfl_id", "slot", "credit", "run_success", "run_gap")
key = ["game_id", "play_id", "nfl_id"]

# %% [markdown]
# ## 1. Pre-snap context (from the lineman's team's point of view)

# %%
is_home = F.col("team_abbr") == F.col("home_team_abbr")
clock_s = F.split("game_clock", ":")[0].cast("int") * 60 + F.split("game_clock", ":")[1].cast("int")
ctx = (plays
    .withColumn("is_home", is_home.cast("int"))
    .withColumn("team_score_diff", F.when(is_home, F.col("pre_snap_home_score") - F.col("pre_snap_visitor_score"))
                                    .otherwise(F.col("pre_snap_visitor_score") - F.col("pre_snap_home_score")))
    .withColumn("team_win_prob", F.when(is_home, F.col("pre_snap_home_team_win_probability"))
                                  .otherwise(F.col("pre_snap_visitor_team_win_probability")))
    .withColumn("yards_from_own_goal", F.when(F.col("yardline_number") == 50, 50)
                .when(F.col("yardline_side") == F.col("possession_team"), F.col("yardline_number"))
                .otherwise(100 - F.col("yardline_number")))
    .withColumn("half_seconds_left", F.when(F.col("quarter").isin(1, 3), clock_s + 900).otherwise(clock_s))
    .withColumn("is_two_minute", (F.col("half_seconds_left") <= 120).cast("int"))
    .withColumn("yards_to_go", F.least(F.col("yards_to_go"), F.lit(20)))
    .withColumn("dist_bucket", F.when(F.col("yards_to_go") <= 3, "short").when(F.col("yards_to_go") <= 7, "medium").otherwise("long"))
    .withColumn("down_cat", F.col("down").cast("string")))

def bucket(df, col, min_share=0.01):
    n = df.count()
    keep = [r[col] for r in df.groupBy(col).count().filter(F.col("count") >= min_share * n).collect() if r[col] is not None]
    return df.withColumn(col, F.when(F.col(col).isin(keep), F.col(col)).when(F.col(col).isNull(), "UNKNOWN").otherwise("OTHER"))
for c in ["offense_formation", "receiver_alignment", "lined_up_position"]:
    ctx = bucket(ctx, c)

# %% [markdown]
# ## 2. Task populations and targets

# %%
pressure = ((F.col("time_to_pressure").isNotNull()) | (F.col("time_to_qb_hurry").isNotNull()) |
            (F.coalesce(F.col("sack"), F.lit(0)) > 0)).cast("int")
dl_rush = (ctx.filter((F.col("line_group") == "DL") & F.col("player_get_off").isNotNull() &
                      F.col("pass_result").isin("C", "I", "IN", "S", "R")))
ol_pb = ctx.filter((F.col("line_group") == "OL") & F.col("pressure_allowed").isNotNull() &
                   F.col("lined_up_position").isin("T", "G", "C"))
ol_run = (ctx.filter((F.col("line_group") == "OL") & F.col("pass_result").isNull())
             .join(ing, key, "inner").filter(F.col("credit") > 0))
tasks = {
    "dl_pressure":      dict(df=dl_rush.withColumn("target", pressure), group="DL", kind="classification"),
    "dl_getoff":        dict(df=dl_rush.withColumn("target", F.col("player_get_off")), group="DL", kind="regression"),
    "ol_pressure":      dict(df=ol_pb.withColumn("target", F.col("pressure_allowed").cast("int")), group="OL", kind="classification"),
    "ol_pressure_prob": dict(df=ol_pb.filter(F.col("peak_pressure_probability_allowed").isNotNull())
                                    .withColumn("target", F.col("peak_pressure_probability_allowed")), group="OL", kind="regression"),
    "ol_run":           dict(df=ol_run.withColumn("target", F.col("run_success")), group="OL", kind="classification"),
}
for n, t in tasks.items():
    print(f"{n:<17} {t['df'].count():>7,} rows")

# %% [markdown]
# ## 3. Player features and the player-grouped split

# %%
SIZE = ["combine_weight", "combine_height", "arm_length"]
pl = (players.select("nfl_id", "draft_year", "nfl_position", *SIZE).join(traits.drop("line_group"), "nfl_id", "left"))
for t in C.TRAITS:
    pl = pl.withColumn(f"miss_sw_{t}", F.col(f"sw_{t}").isNull().cast("int"))
plp = pl.join(players.select("nfl_id", "line_group"), "nfl_id").toPandas()

rng = np.random.default_rng(C.SEED)
plp["split"] = np.where(plp.draft_year == C.TEST_DRAFT_CLASS, "test", "train")
plp["fold"] = -1
plp["pos_group"] = np.where(plp.line_group == "DL", np.where(plp.nfl_position.isin(["DE", "OLB"]), "edge", "interior"),
                            plp.nfl_position.fillna("OL"))
for (_, _, _), grp in plp[plp.split == "train"].groupby(["line_group", "draft_year", "pos_group"]):
    idx = rng.permutation(grp.index.values)
    n_val = int(round(C.VALIDATION_FRACTION * len(idx)))
    plp.loc[idx[:n_val], "split"] = "validation"
    for j, i in enumerate(idx[n_val:]):
        plp.loc[i, "fold"] = j % C.N_FOLDS
# winsorize size on training players
for c in SIZE:
    lo, hi = plp.loc[plp.split == "train", c].quantile([0.01, 0.99])
    plp[c] = plp[c].clip(lo, hi)
plp.groupby(["line_group", "split"]).nfl_id.count().unstack().to_csv(C.TABLE_DIR / "05_players_per_split.csv")
plp.groupby(["line_group", "split"]).nfl_id.count().unstack()

# %% [markdown]
# ## 4. Assemble, weight and save each task table

# %%
CTX_CAT = ["down_cat", "dist_bucket", "offense_formation", "receiver_alignment", "lined_up_position"]
CTX_NUM = ["yards_to_go", "quarter", "half_seconds_left", "is_two_minute", "yards_from_own_goal",
           "team_score_diff", "team_win_prob", "is_home", "nfl_experience"]
RUN_CAT, RUN_NUM = ["run_gap", "slot"], ["credit"]
TRK = [f"trk_{t}" for t in C.TRAITS]; SW = [f"sw_{t}" for t in C.TRAITS]; ADJ = [f"adj_{t}" for t in C.TRAITS]
MISS = [f"miss_sw_{t}" for t in C.TRAITS]
pl_cols = ["nfl_id", "split", "fold", "nfl_position", "pos_group", "draft_year"] + TRK + SW + ADJ + MISS + SIZE
pls = spark.createDataFrame(plp[pl_cols].astype({c: float for c in TRK + SW + ADJ + SIZE}))
summary = []
for name, t in tasks.items():
    extra_cat, extra_num = (RUN_CAT, RUN_NUM) if name == "ol_run" else ([], [])
    df = (t["df"].join(pls, "nfl_id", "inner")
          .withColumn("nfl_experience", F.col("season") - F.col("draft_year"))
          .select(*key, "season", "week", "target", *CTX_CAT, *CTX_NUM, *extra_cat, *extra_num, *pl_cols[1:]))
    df = df.dropna(subset=["target"] + CTX_NUM)
    base_w = F.col("credit") if name == "ol_run" else F.lit(1.0)
    if t["kind"] == "classification":
        p_tr = df.filter("split = 'train'").agg(F.avg("target")).first()[0]
        cw = F.when(F.col("target") == 1, 0.5 / p_tr).otherwise(0.5 / (1 - p_tr))
        df = df.withColumn("weight", F.when(F.col("split") == "train", cw * base_w).otherwise(base_w))
    else:
        df = df.withColumn("weight", base_w)
    df.write.mode("overwrite").parquet(C.p(f"task_{name}.parquet"))
    d = spark.read.parquet(C.p(f"task_{name}.parquet"))
    s = d.groupBy("split").agg(F.countDistinct("nfl_id").alias("players"), F.count("*").alias("rows"),
                               F.round(F.avg("target"), 4).alias("target_mean")).toPandas()
    s["task"] = name; summary.append(s)
summary = pd.concat(summary)[["task", "split", "players", "rows", "target_mean"]]
summary.to_csv(C.TABLE_DIR / "05_task_summary.csv", index=False)

feature_sets = {"context_cat": CTX_CAT, "context_num": CTX_NUM, "run_cat": RUN_CAT, "run_num": RUN_NUM,
                "trk": TRK, "sw": SW + MISS, "adj": ADJ, "size": SIZE,
                "tasks": {n: {"group": t["group"], "kind": t["kind"]} for n, t in tasks.items()}}
json.dump(feature_sets, open(C.OUTPUT_DIR / "feature_sets.json", "w"), indent=2)
summary.pivot_table(index="task", columns="split", values=["players", "rows", "target_mean"])
