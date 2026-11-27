# %% [markdown]
# # 04 · In-game movement, lineman side and run-gap attribution
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# 1. **In-game movement** from 10 Hz NFL tracking, measured from the `ball_snap` frame of every OL and DL snap: first step (`g_t_1yd`, `g_s05`), burst (`g_s10`, `g_d10`) and lateral movement (`g_lat_max` = peak sideways speed in the first 1.5 s). These are the in-game counterparts of the Combine traits, used for the **translation check** (never as model inputs: they happen after the snap).
# 2. **Lineman side.** The data only says T / G / C. The ball is always spotted between the hash marks (y = 23.58–29.75), so a guard or tackle lined up clearly outside the hashes must be on the side of the ball he is on. Combined with the offense's direction, that gives **left or right** on those snaps; every other snap takes the player's majority side for that season and position.
# 3. **Run-gap attribution.** The play-by-play text records where each run went ("left guard", "up the middle", "right end", …). That is where the ball carrier actually went, so a cutback is credited to the gap he cut to, which is the closest lineman to where the run happened. Credit: **1.0** to the lineman at the gap, **0.5** to the adjacent lineman, **0** (away) otherwise.
#
# | Gap in play-by-play | Full credit (1.0) | Half credit (0.5) |
# |---|---|---|
# | up the middle | C | LG, RG |
# | left / right guard | LG / RG | C, LT / C, RT |
# | left / right tackle | LT / RT | LG / RG |
# | left / right end | none (edge: TE / back) | LT / RT |
#
# Run success = EPA > 0 on the play (the standard "successful play" definition).
#
# **Inputs:** `processed/line_player_play.parquet`, `processed/line_game_tracking.parquet`, `processed/trait_indices.parquet`, `processed/line_players.parquet`
# **Outputs:** `processed/ingame_snaps.parquet`, `processed/ol_side_majority.parquet`, `outputs/tables/04_*.csv`, `outputs/figures/04_*.png`

# %%
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import config as C

spark = C.get_spark("04_ingame")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
T0 = time.time()
plays = spark.read.parquet(C.p("line_player_play.parquet")).filter(F.col("season_type") == "REG")
gt = spark.read.parquet(C.p("line_game_tracking.parquet"))
players = spark.read.parquet(C.p("line_players.parquet"))

# Play types: a "pass" play has a pass result; a "run" play has none and a run gap in its description
GAP_RE = r"(left end|left tackle|left guard|up the middle|right guard|right tackle|right end)"
plays = (plays
    .withColumn("run_gap", F.lower(F.regexp_extract(F.lower("play_description"), GAP_RE, 1)))
    .withColumn("run_gap", F.when(F.col("run_gap") == "", None).otherwise(F.col("run_gap")))
    .withColumn("play_type", F.when(F.col("pass_result").isNotNull(), "pass")
                .when(F.col("run_gap").isNotNull(), "run").otherwise("other")))
plays.groupBy("line_group", "play_type").count().orderBy("line_group", "play_type").show()

# %% [markdown]
# ## 1. Snap-relative in-game movement

# %%
key = ["game_id", "play_id", "nfl_id"]
snap = (gt.filter(F.col("event") == "ball_snap").groupBy(*key)
          .agg(F.min(F.col("ts").cast("double")).alias("snap_t")))
w = Window.partitionBy(*key).orderBy("ts")
g = (gt.join(snap, key, "inner")
       .withColumn("t_rel", F.round(F.col("ts").cast("double") - F.col("snap_t"), 2))
       .filter((F.col("t_rel") >= -0.3) & (F.col("t_rel") <= 2.5))
       .withColumn("s_sm", F.avg("s").over(w.rowsBetween(-1, 1)))
       .withColumn("vy", F.col("s_sm") * F.cos(F.radians("dir")))           # sideways (across the field)
       .withColumn("step_xy", F.sqrt((F.col("x") - F.lag("x").over(w))**2 + (F.col("y") - F.lag("y").over(w))**2)))
at_snap = g.filter(F.col("t_rel") == 0).select(*key, F.col("x").alias("x0"), F.col("y").alias("y0"))
g = g.join(at_snap, key, "inner").withColumn("disp", F.sqrt((F.col("x") - F.col("x0"))**2 + (F.col("y") - F.col("y0"))**2))

def speed_at(t):
    return F.max(F.when((F.col("t_rel") >= t - 0.05) & (F.col("t_rel") < t + 0.05), F.col("s_sm")))

mv = (g.groupBy(*key).agg(
        F.min(F.when((F.col("t_rel") > 0) & (F.col("disp") >= 1.0), F.col("t_rel"))).alias("g_t_1yd"),
        speed_at(0.5).alias("g_s05"), speed_at(1.0).alias("g_s10"),
        F.sum(F.when((F.col("t_rel") > 0) & (F.col("t_rel") <= 1.0), F.col("step_xy"))).alias("g_d10"),
        F.max(F.when((F.col("t_rel") > 0) & (F.col("t_rel") <= 1.5), F.abs("vy"))).alias("g_lat_max"),
        F.first("y0").alias("y_snap")))
print(f"snaps with in-game movement: {mv.count():,}")

# %% [markdown]
# ## 2. Offensive-line side (left / right) from position at the snap

# %%
MARGIN = 0.75
ol = (plays.filter((F.col("line_group") == "OL") & F.col("lined_up_position").isin("T", "G", "C"))
           .join(mv.select(*key, "y_snap"), key, "left"))
ol = (ol.withColumn("above", F.when(F.col("y_snap") > C.HASH_HI + MARGIN, 1)
                               .when(F.col("y_snap") < C.HASH_LO - MARGIN, -1))
        # facing +x (play_direction 'right') the offense's left is +y
        .withColumn("left_vote", F.when(F.col("play_direction") == "right", F.col("above")).otherwise(-F.col("above"))))
maj = (ol.filter(F.col("lined_up_position").isin("T", "G") & F.col("left_vote").isNotNull())
         .groupBy("nfl_id", "season", "lined_up_position")
         .agg(F.avg("left_vote").alias("vote_mean"), F.count("*").alias("votes")))
maj.write.mode("overwrite").parquet(C.p("ol_side_majority.parquet"))
mp = maj.toPandas()
print(f"player-season-position groups with votes: {len(mp)}; consistent (|mean| ≥ 0.6): {(mp.vote_mean.abs() >= 0.6).mean():.0%}")
print("share of T/G snaps with a direct vote:",
      round(ol.filter(F.col("lined_up_position").isin("T", "G")).select(F.avg(F.col("left_vote").isNotNull().cast("int"))).first()[0], 3))

ol = (ol.join(maj, ["nfl_id", "season", "lined_up_position"], "left")
        .withColumn("side", F.when(F.col("lined_up_position") == "C", "M")
                    .when(F.col("left_vote") == 1, "L").when(F.col("left_vote") == -1, "R")
                    .when(F.col("vote_mean") >= 0.6, "L").when(F.col("vote_mean") <= -0.6, "R"))
        .withColumn("slot", F.when(F.col("side") == "M", F.lit("C"))
                    .when(F.col("side").isNotNull(), F.concat(F.col("side"), F.col("lined_up_position")))))
ol.groupBy("slot").count().orderBy("slot").show()

# Validation: two rookies at the same position on the same play must be on opposite sides
pairs = (ol.filter(F.col("lined_up_position").isin("T", "G") & F.col("side").isin("L", "R"))
           .groupBy("game_id", "play_id", "lined_up_position").agg(F.collect_list("side").alias("sides"))
           .filter(F.size("sides") == 2))
pv = pairs.select(F.avg((F.col("sides")[0] != F.col("sides")[1]).cast("int")).alias("opposite"), F.count("*").alias("n")).first()
print(f"side check: {pv.n} plays with two rookie {'{T or G}'} → opposite sides on {pv.opposite:.1%}")

# %% [markdown]
# ## 3. Run-gap credit

# %%
credit_map = {
    "up the middle": {"C": 1.0, "LG": 0.5, "RG": 0.5},
    "left guard": {"LG": 1.0, "C": 0.5, "LT": 0.5}, "right guard": {"RG": 1.0, "C": 0.5, "RT": 0.5},
    "left tackle": {"LT": 1.0, "LG": 0.5}, "right tackle": {"RT": 1.0, "RG": 0.5},
    "left end": {"LT": 0.5}, "right end": {"RT": 0.5},
}
cm = spark.createDataFrame([(gap, slot, c) for gap, d in credit_map.items() for slot, c in d.items()],
                           ["run_gap", "slot", "credit"])
ol_run = (ol.filter(F.col("play_type") == "run").join(cm, ["run_gap", "slot"], "left")
            .withColumn("credit", F.when(F.col("slot").isNull(), None).otherwise(F.coalesce("credit", F.lit(0.0))))
            .withColumn("run_success", (F.col("expected_points_added") > 0).cast("int")))
rs = (ol_run.groupBy("credit").agg(F.count("*").alias("snaps"), F.round(F.avg("run_success"), 3).alias("success_rate"),
      F.round(F.avg("yards_gained"), 2).alias("yards")).orderBy("credit").toPandas())
rs.to_csv(C.TABLE_DIR / "04_run_credit_summary.csv", index=False)
rs

# %% [markdown]
# ## 4. Assemble one in-game row per lineman-snap

# %%
cols = key + ["line_group", "season", "week", "play_type", "lined_up_position", "run_gap"]
base = plays.select(*cols)
olx = ol.select(*key, "side", "slot")
runx = ol_run.select(*key, "credit", "run_success")
snaps = (base.join(mv.drop("y_snap"), key, "left").join(olx, key, "left").join(runx, key, "left"))
snaps.write.mode("overwrite").parquet(C.p("ingame_snaps.parquet"))
snaps = spark.read.parquet(C.p("ingame_snaps.parquet"))
print(f"in-game rows: {snaps.count():,}")

# %% [markdown]
# ## 5. Translation check: do Combine traits show up in NFL movement?
# Player-level means of the in-game counterparts (players with 50+ snaps of that type), against the Combine trait index. DL use pass-rush snaps (rows with a recorded get-off); OL use pass-block snaps and run snaps separately.

# %%
rush_keys = plays.filter((F.col("line_group") == "DL") & F.col("player_get_off").isNotNull() & (F.col("play_type") == "pass")).select(*key)
pblock_keys = plays.filter((F.col("line_group") == "OL") & F.col("pressure_allowed").isNotNull()).select(*key)
run_keys = plays.filter((F.col("line_group") == "OL") & (F.col("play_type") == "run")).select(*key)
gm = ["g_t_1yd", "g_s05", "g_s10", "g_d10", "g_lat_max"]
def player_means(keys, label):
    return (snaps.join(keys, key, "left_semi").groupBy("nfl_id").agg(F.count("*").alias("n"), *[F.avg(c).alias(c) for c in gm])
            .filter("n >= 50").withColumn("context", F.lit(label)))
pm = (player_means(rush_keys, "DL pass rush").unionByName(player_means(pblock_keys, "OL pass block"))
      .unionByName(player_means(run_keys, "OL run block")))
ti = spark.read.parquet(C.p("trait_indices.parquet"))
pos = players.select("nfl_id", "nfl_position", "combine_weight")
tp = pm.join(ti, "nfl_id").join(pos, "nfl_id").toPandas()
pairs = {"first_step": [("g_t_1yd", -1), ("g_s05", 1)], "lateral": [("g_lat_max", 1)], "burst": [("g_s10", 1), ("g_d10", 1)]}
rows = []
for ctx, d in tp.groupby("context"):
    for trait, outs in pairs.items():
        for col, sgn in outs:
            for src in ("trk", "sw", "adj"):
                dd = d[[f"{src}_{trait}", col]].dropna()
                if len(dd) < 15:
                    continue
                rows.append({"context": ctx, "trait": trait, "index": src, "in_game": col, "n": len(dd),
                             "spearman_oriented": round(sgn * dd.iloc[:, 0].rank().corr(dd.iloc[:, 1].rank()), 3)})
tr = pd.DataFrame(rows)
tr.to_csv(C.TABLE_DIR / "04_translation.csv", index=False)
print("oriented so that + means 'better at the Combine → better in games'")
tr.pivot_table(index=["context", "trait", "in_game"], columns="index", values="spearman_oriented").round(2)

# %%
# Figure: translation heatmap (tracking index vs in-game counterpart)
h = tr[tr["index"].isin(["trk", "sw"])].copy()
h["row"] = h.trait.str.replace("_", " ") + " → " + h.in_game
hm = h.pivot_table(index="row", columns=["context", "index"], values="spearman_oriented")
fig, ax = plt.subplots(figsize=(10, 4.2))
im = ax.imshow(hm.values, cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_yticks(range(len(hm.index)), hm.index)
ax.set_xticks(range(len(hm.columns)), [f"{c}\n{'tracking' if i == 'trk' else 'stopwatch'}" for c, i in hm.columns], fontsize=8)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        v = hm.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.35 else "black")
fig.colorbar(im, ax=ax, label="Spearman ρ (+ = better at Combine, better in games)")
ax.set_title("Translation: Combine trait index vs the same player's in-game movement", fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "04_translation_heatmap.png", bbox_inches="tight"); plt.show()

# %%
# Figure: run success by credit, and success at the point of attack by slot
po = ol_run.filter(F.col("credit") > 0).groupBy("slot").agg(F.count("*").alias("n"), F.avg("run_success").alias("rate")).toPandas()
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
labels = ["side unknown" if pd.isna(c) else ("away (0)" if c == 0 else f"credit {c:g}") for c in rs.credit]
axes[0].bar(labels, rs.success_rate, color=["#cccccc" if pd.isna(c) else "#2a6fdb" for c in rs.credit])
axes[0].set_xlabel("Run-gap credit"); axes[0].set_ylabel("Run success rate (EPA > 0)"); axes[0].set_title("All rookie-OL run snaps", fontsize=10)
po = po.set_index("slot").reindex(["LT", "LG", "C", "RG", "RT"]).dropna()
axes[1].bar(po.index, po.rate, color="#2a6fdb"); axes[1].set_title("Point-of-attack runs by lineman slot", fontsize=10)
for i, (s_, r_) in enumerate(po.iterrows()):
    axes[1].text(i, r_.rate + 0.005, f"n={int(r_.n):,}", ha="center", fontsize=8)
for ax in axes:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.suptitle("Run-blocking outcome after gap attribution", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "04_run_attribution.png", bbox_inches="tight"); plt.show()
print(f"done in {time.time() - T0:,.0f} s")
