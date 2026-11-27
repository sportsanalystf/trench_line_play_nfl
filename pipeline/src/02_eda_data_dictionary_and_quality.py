# %% [markdown]
# # 02 · Dataset, EDA and data-quality assessment
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# 1. **Data dictionary**: source tables (grain, key, size) and every column the project uses, with its role.
# 2. **Analysis populations** for the three outcomes: DL pass rush, OL pass blocking, OL run blocking.
# 3. **Descriptive statistics**: Combine measurables by group, outcome rates, snaps per player.
# 4. **Data-quality assessment**: missing stopwatch results, drill-name inconsistency, the unreliable `dis` column, unsigned acceleration, row count vs documentation, season labelling, run-gap coverage, missing lineman side.
# 5. **Visualizations** saved to `outputs/figures/02_*.png`.
# 6. **Whole-release view** (team EDA by Isaac Marr): size of the full data set by season and season type, Combine drill coverage for every position, the lineman drills, how many players have enough snaps, DL get-off by alignment, and a first look at the 10-yard split against get-off.
#
# Sections 1–5 use notebook 01 outputs; section 6 reads the raw CSVs. **Outputs:** `outputs/tables/02_*.csv`, `outputs/figures/02_*.png`.

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
import config as C

spark = C.get_spark("02_eda")
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
pd.set_option("display.max_colwidth", 110, "display.width", 200)
BLUE, ORANGE, GREY = "#2a6fdb", "#e07b39", "#8a8a8a"
players = spark.read.parquet(C.p("line_players.parquet"))
plays = spark.read.parquet(C.p("line_player_play.parquet"))
ct = spark.read.parquet(C.p("combine_tracking.parquet"))
gt = spark.read.parquet(C.p("line_game_tracking.parquet"))

# %% [markdown]
# ## 1. Data dictionary
# ### 1a. Source tables

# %%
tables = pd.DataFrame([
    ("players.csv", "player", "nfl_id", "Demographics, draft round / pick"),
    ("combine_results.csv", "player", "nfl_id", "Measurables and stopwatch drills"),
    ("combine_tracking.csv", "10 Hz frame of a drill attempt", "event_id + time", "Combine optical tracking (x, y, speed, accel., direction)"),
    ("player_play.csv", "player × play", "game_id + play_id + nfl_id", "Play context, play-by-play text, NGS pass-rush / pass-block metrics"),
    ("games.csv", "game", "game_id", "Season, week, season type"),
    ("game_tracking_2023–25.csv", "10 Hz frame of an NFL play", "game_id + play_id + nfl_id + time", "In-game tracking of the 510 rookies (~2 GB)"),
    ("player_career_successes.csv", "player", "nfl_id", "Career snaps and starts"),
], columns=["file", "grain (one row =)", "key", "content"])
tables["rows used (OL + DL cohort)"] = ["239", "239", f"{ct.count():,}", f"{plays.count():,}", "1,002", f"{gt.count():,}", "239"]
tables.to_csv(C.TABLE_DIR / "02_source_tables.csv", index=False)
tables

# %% [markdown]
# ### 1b. Columns used in modeling
# Roles: **target**, **context** (known before the snap), **trait** (from Combine tracking, notebook 03), **stopwatch**, **size**, **id/split**, **attribution** (notebook 04), **excluded** (post-snap, would leak the outcome).

# %%
dd = [
 ("pressure (DL)", "target", "player_play", "1 if the rusher has a time-to-pressure, a QB hurry, or a sack"),
 ("player_get_off (DL)", "target", "player_play", "Seconds from snap until the rusher crosses the line"),
 ("pressure_allowed (OL)", "target", "player_play", "True if the lineman allowed a hurry, hit or sack"),
 ("peak_pressure_probability_allowed (OL)", "target", "player_play", "Peak NGS pressure probability he allowed (0–1)"),
 ("run_success (OL)", "target", "derived", "EPA > 0 on a run where he is at the point of attack"),
 ("down, yards_to_go, dist_bucket, quarter, half_seconds_left, is_two_minute", "context", "player_play", "Game situation"),
 ("yards_from_own_goal, team_score_diff, team_win_prob, is_home", "context", "derived", "Field position, score and win probability from his team's side"),
 ("offense_formation, receiver_alignment, lined_up_position", "context", "player_play", "Rare levels (< 1%) grouped as OTHER"),
 ("nfl_experience", "context", "derived", "season − draft year"),
 ("run_gap, slot, credit (OL run)", "attribution", "nb 04", "Gap from play-by-play text; lineman slot LT…RT; credit 1 / 0.5"),
 ("trk_first_step, trk_lateral, trk_burst", "trait", "nb 03", "Tracking trait indices (z-scores within group, higher = better)"),
 ("adj_first_step, adj_lateral, adj_burst", "trait", "nb 03", "Same, after removing the effect of body weight"),
 ("sw_first_step, sw_lateral, sw_burst (+ miss flags)", "stopwatch", "nb 03", "10-yd split; shuttle + 3-cone; broad + vertical + 40"),
 ("combine_weight, combine_height, arm_length", "size", "combine_results", "Measurables"),
 ("nfl_id, draft_year, split, fold", "id/split", "players / nb 05", "Player-grouped splits; 2025 class = test"),
 ("pass_result, dropback_duration, time_to_pressure_allowed, pass_rushers_encountered, sack", "excluded", "player_play", "Post-snap: define or follow the outcome"),
]
data_dict = pd.DataFrame(dd, columns=["column(s)", "role", "source", "description"])
data_dict.to_csv(C.TABLE_DIR / "02_data_dictionary.csv", index=False)
data_dict

# %% [markdown]
# ## 2. Analysis populations

# %%
reg = plays.filter(F.col("season_type") == "REG")
GAP_RE = r"(left end|left tackle|left guard|up the middle|right guard|right tackle|right end)"
reg = reg.withColumn("run_gap", F.regexp_extract(F.lower("play_description"), GAP_RE, 1))
not_null = F.coalesce(F.col("play_nullified_by_penalty"), F.lit("N")) != "Y"
pop = []
dl = reg.filter(F.col("line_group") == "DL")
pop += [("DL", "all regular-season rows", dl.count()),
        ("DL", "pass-rush snaps (dropback + get-off, not nullified)",
         dl.filter(F.col("player_get_off").isNotNull() & F.col("pass_result").isin("C", "I", "IN", "S", "R") & not_null).count())]
ol = reg.filter((F.col("line_group") == "OL") & F.col("lined_up_position").isin("T", "G", "C"))
pop += [("OL", "all regular-season rows at T/G/C", ol.count()),
        ("OL", "pass-block snaps (pressure_allowed recorded)", ol.filter(F.col("pressure_allowed").isNotNull() & not_null).count()),
        ("OL", "run snaps (no pass result)", ol.filter(F.col("pass_result").isNull()).count()),
        ("OL", "run snaps with a run gap in the text", ol.filter(F.col("pass_result").isNull() & (F.col("run_gap") != "")).count())]
pop = pd.DataFrame(pop, columns=["group", "population", "rows"]); pop.to_csv(C.TABLE_DIR / "02_populations.csv", index=False)
pop

# %% [markdown]
# ## 3. Descriptive statistics
# ### 3a. Combine measurables and stopwatch drills by group

# %%
meas = ["combine_height", "combine_weight", "arm_length", "forty", "ten_yd_split", "vertical", "broad_jump",
        "three_cone", "short_shuttle", "bench_reps", "age_at_draft"]
desc = []
for g in C.GROUPS:
    d = players.filter(F.col("line_group") == g).select(meas).toPandas()
    s = d.describe().T[["count", "mean", "std", "min", "50%", "max"]]; s["missing_%"] = (100 * d.isna().mean()).round(1)
    s.insert(0, "group", g); desc.append(s)
desc = pd.concat(desc).round(2); desc.to_csv(C.TABLE_DIR / "02_combine_descriptives.csv")
desc

# %% [markdown]
# ### 3b. Outcomes

# %%
pr = (dl.filter(F.col("player_get_off").isNotNull() & F.col("pass_result").isin("C", "I", "IN", "S", "R") & not_null)
        .withColumn("pressure", ((F.col("time_to_pressure").isNotNull()) | F.col("time_to_qb_hurry").isNotNull() |
                                 (F.coalesce(F.col("sack"), F.lit(0)) > 0)).cast("int")))
pb = ol.filter(F.col("pressure_allowed").isNotNull() & not_null).withColumn("pa", F.col("pressure_allowed").cast("int"))
rn = ol.filter(F.col("pass_result").isNull() & (F.col("run_gap") != "") & not_null).withColumn("success", (F.col("expected_points_added") > 0).cast("int"))
outc = pd.DataFrame([
    ("DL pressure rate", pr.agg(F.avg("pressure")).first()[0], pr.select("nfl_id").distinct().count()),
    ("DL mean get-off (s)", pr.agg(F.avg("player_get_off")).first()[0], None),
    ("OL pressure-allowed rate", pb.agg(F.avg("pa")).first()[0], pb.select("nfl_id").distinct().count()),
    ("OL mean peak pressure probability", pb.agg(F.avg("peak_pressure_probability_allowed")).first()[0], None),
    ("OL run success rate (EPA > 0)", rn.agg(F.avg("success")).first()[0], rn.select("nfl_id").distinct().count()),
], columns=["outcome", "value", "players"])
outc["value"] = outc.value.round(4); outc.to_csv(C.TABLE_DIR / "02_outcomes.csv", index=False)
outc

# %% [markdown]
# ## 4. Data-quality assessment

# %%
dq = []
for g in C.GROUPS:
    for c in ["three_cone", "short_shuttle", "ten_yd_split", "forty", "broad_jump", "vertical"]:
        dq.append(("missing value", f"{g} · {c}", f"{desc[(desc.group == g)].loc[c, 'missing_%']}% of prospects", "trait index from other components; stopwatch index + missing flag"))
names = ct.filter(F.col("drill_name").contains("MIRROR")).select("drill_name").distinct().count()
dq.append(("inconsistent label", "combine_tracking.drill_name", "PASS_PRO-MIRROR_DRILL (2024–25) vs PASS_PRO_MIRROR_DRILL (2023)", f"standardized in nb 01 ({names} name left)"))
dis = ct.filter(F.col("step_xy") > 0.05).select(F.expr("percentile_approx(dis / (step_xy * step_xy), 0.5)").alias("r"),
                                                 F.corr("dis", F.col("step_xy") * F.col("step_xy")).alias("c")).first()
dq.append(("unreliable column", "combine_tracking.dis", f"dis ≈ (x/y step)²: median ratio {dis.r:.2f}, corr {dis.c:.3f}", "distance recomputed from x, y"))
dq.append(("semantics", "combine_tracking.a", f"min = {ct.agg(F.min('a')).first()[0]:.2f}: unsigned magnitude", "signed acceleration from Δspeed (nb 03)"))
dq.append(("row count", "player_play.csv", "314,197 rows vs 316,338 on the competition page", "used as delivered"))
jan = plays.filter((F.substring(F.col("game_id").cast("string"), 5, 2) == "01") &
                   (F.substring(F.col("game_id").cast("string"), 1, 4).cast("int") != F.col("season"))).count()
dq.append(("mislabel risk", "game_id", f"{jan:,} January cohort rows with year ≠ season", "season from games.csv"))
dq.append(("scope", "season_type", f"{plays.filter(F.col('season_type') == 'PRE').count():,} preseason cohort rows", "excluded"))
no_gap = ol.filter(F.col("pass_result").isNull() & (F.col("run_gap") == "")).count()
dq.append(("missing value", "play_description run gap", f"{no_gap:,} OL non-pass rows without a gap (kneels, aborted snaps, penalties)", "excluded from run blocking"))
dq.append(("missing field", "lineman side", "data gives T/G/C only; no left/right", "inferred from position at the snap vs hash marks (nb 04)"))
dq.append(("coverage", "in-game tracking", "only the 510 rookies are tracked: no ball carrier, no blockers or rushers they face", "outcomes from NGS fields + play-by-play text"))
dq.append(("constant column", "player_play.unblocked_pressure", f"non-null on {ol.filter(F.col('unblocked_pressure').isNotNull()).count()} OL rows", "not used"))
dq = pd.DataFrame(dq, columns=["issue", "where", "finding", "treatment"]); dq.to_csv(C.TABLE_DIR / "02_data_quality.csv", index=False)
dq

# %% [markdown]
# ## 5. Visualizations

# %%
# Fig 1: stopwatch vs tracked-drill coverage by group
rows = []
for g in C.GROUPS:
    n = players.filter(F.col("line_group") == g).count()
    for c in ["three_cone", "short_shuttle", "ten_yd_split", "broad_jump"]:
        rows.append((g, f"stopwatch: {c}", 1 - desc[desc.group == g].loc[c, "missing_%"] / 100))
    for d_ in sorted({d for ds in C.TRAIT_DRILLS[g].values() for d in ds} - {"SHORT_SHUTTLE", "THREE_CONE_DRILL", "FORTY_YARD_DASH"}):
        rows.append((g, f"tracked: {d_.lower().replace('_drill', '').replace('_', ' ')[:28]}",
                     ct.filter((F.col("line_group") == g) & (F.col("drill_name") == d_)).select("nfl_id").distinct().count() / n))
cov = pd.DataFrame(rows, columns=["group", "drill", "share"])
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, g in zip(axes, C.GROUPS):
    d = cov[cov.group == g].sort_values("share")
    ax.barh(d.drill, d.share, color=[ORANGE if s.startswith("stop") else BLUE for s in d.drill])
    for i, v in enumerate(d.share): ax.text(v + .01, i, f"{v:.0%}", va="center", fontsize=8)
    ax.set_xlim(0, 1.12); ax.set_title(f"{g} ({players.filter(F.col('line_group') == g).count()} prospects)", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.suptitle("Tracked position drills cover nearly every lineman; agility stopwatch drills don't", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_coverage_by_group.png", bbox_inches="tight"); plt.show()

# %%
# Fig 2: outcome rates by down and distance for the three outcomes
def by_sit(df, col):
    return (df.filter(F.col("down").between(1, 3)).withColumn("dist", F.when(F.col("yards_to_go") <= 3, "1–3").when(F.col("yards_to_go") <= 7, "4–7").otherwise("8+"))
              .groupBy("down", "dist").agg(F.avg(col).alias("rate")).toPandas().pivot(index="down", columns="dist", values="rate")[["1–3", "4–7", "8+"]])
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, (df, col, title, c) in zip(axes, [(pr, "pressure", "DL: pressure rate", ORANGE), (pb, "pa", "OL: pressure-allowed rate", BLUE),
                                          (rn, "success", "OL: run success rate (EPA > 0)", BLUE)]):
    by_sit(df, col).plot.bar(ax=ax, rot=0, color=["#d9d9d9", "#9aa5b1", c], width=.8)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Down"); ax.legend(title="Yards to go", fontsize=7, frameon=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
fig.suptitle("Every outcome moves with down and distance, so game context is the benchmark to beat", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_outcomes_by_situation.png", bbox_inches="tight"); plt.show()

# %%
# Fig 3: snaps per player by outcome
spp = {"DL pass rush": pr, "OL pass block": pb, "OL run (all gaps)": rn}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
for ax, (lab, df) in zip(axes, spp.items()):
    s = df.groupBy("nfl_id").count().toPandas()["count"]
    ax.hist(s, bins=25, color=ORANGE if lab.startswith("DL") else BLUE, alpha=.85)
    ax.set_title(f"{lab}: median {s.median():.0f}, max {s.max():,}", fontsize=9); ax.set_xlabel("Snaps per player")
axes[0].set_ylabel("Players")
fig.suptitle("Usage is very uneven: a few starters supply most snaps", fontweight="bold")
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_snaps_per_player.png", bbox_inches="tight"); plt.show()

# %%
# Fig 4: where guards and tackles line up at the snap, relative to the hash marks (basis of the side inference)
snap_y = (gt.filter(F.col("event") == "ball_snap").select("game_id", "play_id", "nfl_id", "y")
            .join(ol.filter(F.col("lined_up_position").isin("T", "G", "C")).select("game_id", "play_id", "nfl_id", "lined_up_position"),
                  ["game_id", "play_id", "nfl_id"]).sample(fraction=0.3, seed=C.SEED).toPandas())
fig, ax = plt.subplots(figsize=(9, 3.4))
for pos, col in [("C", GREY), ("G", BLUE), ("T", ORANGE)]:
    ax.hist(snap_y[snap_y.lined_up_position == pos].y, bins=80, alpha=.55, color=col, label=pos, density=True)
for h in (C.HASH_LO, C.HASH_HI):
    ax.axvline(h, color="#333", ls="--", lw=1)
ax.text((C.HASH_LO + C.HASH_HI) / 2, ax.get_ylim()[1] * .9, "ball is spotted\nbetween hashes", ha="center", fontsize=8)
ax.set_xlabel("y at the snap (yards from sideline)"); ax.legend(frameon=False)
ax.set_title("Guards and tackles outside the hashes reveal which side of the ball they play", fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_ol_position_vs_hashes.png", bbox_inches="tight"); plt.show()

# %%
# Fig 5: the 'dis' data-quality problem
smp = ct.filter(F.col("step_xy").isNotNull()).sample(fraction=0.05, seed=C.SEED).select("dis", "step_xy").toPandas()
fig, ax = plt.subplots(figsize=(5, 3.8))
ax.scatter(smp.step_xy, smp.dis, s=3, alpha=.3, color=GREY); lim = smp.step_xy.quantile(.995); xs = np.linspace(0, lim, 50)
ax.plot(xs, xs, color=ORANGE, lw=1.5, label="dis = step (expected)"); ax.plot(xs, xs ** 2, color=BLUE, lw=1.5, label="dis = step²")
ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_xlabel("Step length from x, y (yd)"); ax.set_ylabel("Reported dis"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Combine 'dis' is a squared distance", fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_dis_quality_check.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 6. Whole-release view (team EDA by Isaac Marr)
# Sections 1–5 describe the 239 linemen. Isaac's exploration first looked at the **whole release** and then narrowed to the linemen; it is reproduced here from the raw CSVs (explicit schemas, as in his `Data_Loading` notebook), so the report can describe the full data before the cohort filter.
# ### 6a. Size of the release by season and season type

# %%
games_raw = spark.read.csv(C.raw("games.csv"), header=True, nullValue="NA",
                           schema="game_id LONG, game_key INT, season INT, season_type STRING, week INT, home_team_abbr STRING, visitor_team_abbr STRING")
gt_raw = spark.read.csv(C.game_tracking_files(), header=True, nullValue="NA",
                        schema="game_id LONG, play_id INT, nfl_id INT, time STRING, x DOUBLE, y DOUBLE, s DOUBLE, a DOUBLE, dis DOUBLE, o DOUBLE, dir DOUBLE, event STRING")
# player_play: read every column as text (no inferSchema pass) and cast only what is needed
pp_raw = (spark.read.option("header", True).option("nullValue", "NA").option("multiLine", True).option("escape", '"')
          .csv(C.raw("player_play.csv"))
          .select(F.col("game_id").cast("long"), F.col("play_id").cast("int"), F.col("nfl_id").cast("int"),
                  "lined_up_position", F.col("pressure_allowed"), F.col("player_get_off").cast("double")))
g = F.broadcast(games_raw.select("game_id", "season", "season_type"))
frames = gt_raw.join(g, "game_id").groupBy("season", "season_type").agg(F.count("*").alias("tracking_frames"))
pprows = pp_raw.join(g, "game_id").groupBy("season", "season_type").agg(F.count("*").alias("player_play_rows"))
release = (frames.join(pprows, ["season", "season_type"], "outer").orderBy("season", "season_type").toPandas())
release.to_csv(C.TABLE_DIR / "02_release_size.csv", index=False)
tot = release[["tracking_frames", "player_play_rows"]].sum()
regpost = release[release.season_type.isin(["REG", "POST"])][["tracking_frames", "player_play_rows"]].sum()
print(f"all games: {tot.tracking_frames:,.0f} tracking frames, {tot.player_play_rows:,.0f} player-play rows")
print(f"regular + post season: {regpost.tracking_frames:,.0f} frames, {regpost.player_play_rows:,.0f} player-play rows")
release

# %% [markdown]
# ### 6b. Combine tracking: what was tracked, for whom
# The Combine file also tracks the **ball** (only in the WR / TE route drills, so it cannot time a lineman's reaction to the snap). Players are grouped by the drill family they ran.

# %%
ct_raw = spark.read.csv(C.raw("combine_tracking.csv"), header=True, nullValue="NA",
                        schema="draft_year INT, event_id STRING, nfl_id INT, entity_type STRING, time STRING, drill_type STRING, "
                               "drill_name STRING, attempt INT, x DOUBLE, y DOUBLE, s DOUBLE, a DOUBLE, dis DOUBLE, dir DOUBLE")
cpos = spark.read.csv(C.raw("combine_results.csv"), header=True, nullValue="NA",
                      schema="draft_year INT, nfl_id INT, combine_position STRING").select("nfl_id", "combine_position")
print("frames by entity type:", {r.entity_type: r["count"] for r in ct_raw.groupBy("entity_type").count().collect()})
print("drill attempts (events):", ct_raw.select("event_id").distinct().count(),
      "| tracked players:", ct_raw.filter(F.col("entity_type") == "PLAYER").select("nfl_id").distinct().count())
ctp = ct_raw.filter(F.col("entity_type") == "PLAYER")
coverage = (ctp.join(cpos, "nfl_id").groupBy("drill_type").pivot("combine_position").agg(F.countDistinct("nfl_id"))
               .join(ctp.groupBy("drill_type").agg(F.count("*").alias("frames"), F.countDistinct("nfl_id").alias("players"),
                                                    F.countDistinct("event_id").alias("attempts")), "drill_type")
               .orderBy(F.desc("players")).toPandas().fillna(0))
coverage.to_csv(C.TABLE_DIR / "02_drill_coverage.csv", index=False)
coverage

# %%
# the lineman drills, by name: players and attempts (PASS_PRO-MIRROR vs PASS_PRO_MIRROR is the naming inconsistency of section 4)
lineman_drills = (ctp.filter(F.col("drill_type").isin("SKILL_DRILLS_OL", "SKILL_DRILLS_DL"))
                     .groupBy("drill_type", "drill_name").agg(F.countDistinct("nfl_id").alias("players"),
                                                             F.countDistinct("event_id").alias("attempts"))
                     .orderBy("drill_type", F.desc("players")).toPandas())
lineman_drills.to_csv(C.TABLE_DIR / "02_lineman_drills.csv", index=False)
lineman_drills

# %% [markdown]
# ### 6c. How many players have enough NFL snaps?
# Regular + post season. Pass-block snaps = rows with a `pressure_allowed` value; rush snaps = rows with a `player_get_off` value. The thresholds show how many players a player-level analysis can use.

# %%
rp = pp_raw.join(g.filter(F.col("season_type").isin("REG", "POST")), "game_id")
per_player = (cpos.join(rp, "nfl_id", "left").groupBy("nfl_id", "combine_position")
                  .agg(F.count("play_id").alias("snaps"), F.count("pressure_allowed").alias("ol_pass_block_snaps"),
                       F.count("player_get_off").alias("dl_rush_snaps")).toPandas())
snaps_by_pos = per_player.groupby("combine_position")[["snaps", "ol_pass_block_snaps", "dl_rush_snaps"]].describe().round(0).T
snaps_by_pos.to_csv(C.TABLE_DIR / "02_snaps_by_position.csv")
rows = []
for grp, col in [("OL", "ol_pass_block_snaps"), ("DL", "dl_rush_snaps")]:
    v = per_player[per_player.combine_position == grp][col]
    rows.append({"group": grp, "snap type": col.replace("_", " "), "players": len(v), "0 snaps": int((v == 0).sum()),
                 "50+": int((v >= 50).sum()), "100+": int((v >= 100).sum()), "200+": int((v >= 200).sum())})
thresholds = pd.DataFrame(rows)
thresholds.to_csv(C.TABLE_DIR / "02_snap_thresholds.csv", index=False)
thresholds

# %% [markdown]
# ### 6d. DL get-off by alignment, and a first look at the 10-yard split
# Get-off = time for a rusher to cross the line after the snap. Edge rushers and interior linemen have different distributions, so later analyses keep alignment (or the share of edge snaps) as context. Isaac's first scatter, the 10-yard split against a player's average get-off, is the starting point of the player-level analysis in notebook 07b.

# %%
dl_ids = cpos.filter(F.col("combine_position") == "DL").select("nfl_id")
dl_go = (rp.join(dl_ids, "nfl_id").filter(F.col("player_get_off").isNotNull() & F.col("lined_up_position").isin("EDGE", "INTERIOR_LINE"))
           .select("nfl_id", "player_get_off", "lined_up_position"))
go = dl_go.toPandas()
print(go.groupby("lined_up_position").player_get_off.describe().round(3))
fig, ax = plt.subplots(figsize=(8, 3.6))
bins = np.linspace(0, 2, 41)
for pos, col, lab in [("EDGE", ORANGE, "edge"), ("INTERIOR_LINE", BLUE, "interior")]:
    v = go[go.lined_up_position == pos].player_get_off
    ax.hist(v, bins=bins, density=True, alpha=.6, color=col, label=f"{lab} (median {v.median():.2f} s, n = {len(v):,})")
ax.set_xlabel("Get-off: time to cross the line after the snap (s)"); ax.set_ylabel("Share of snaps (density)"); ax.legend(frameon=False)
ax.set_title("Edge rushers get off faster than interior linemen", fontweight="bold", fontsize=10)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_dl_getoff_by_alignment.png", bbox_inches="tight"); plt.show()

# %%
stop = spark.read.csv(C.raw("combine_results.csv"), header=True, nullValue="NA",
                      schema="draft_year INT, nfl_id INT, combine_position STRING, combine_height DOUBLE, combine_weight DOUBLE, "
                             "hand_size DOUBLE, arm_length DOUBLE, wing_span DOUBLE, ten_yd_split DOUBLE").select("nfl_id", "ten_yd_split")
first_look = (dl_go.groupBy("nfl_id").agg(F.count("*").alias("rush_snaps"), F.avg("player_get_off").alias("mean_get_off"),
                                         F.avg((F.col("lined_up_position") == "EDGE").cast("double")).alias("edge_share"))
                   .filter(F.col("rush_snaps") >= 100).join(stop, "nfl_id").dropna(subset=["ten_yd_split"]).toPandas())
r = first_look[["ten_yd_split", "mean_get_off"]].corr().iloc[0, 1]
fig, ax = plt.subplots(figsize=(6, 4.2))
sc = ax.scatter(first_look.ten_yd_split, first_look.mean_get_off, c=first_look.edge_share, cmap="viridis", vmin=0, vmax=1, s=36,
                edgecolors="white", linewidths=.6)
fig.colorbar(sc, ax=ax, label="share of snaps at edge")
ax.set_xlabel("Combine 10-yard split (s)"); ax.set_ylabel("Average NFL get-off (s)")
ax.set_title(f"Faster 10-yard split, faster get-off (r = {r:.2f}, {len(first_look)} DL with 100+ rush snaps)", fontweight="bold", fontsize=9.5)
fig.tight_layout(); fig.savefig(C.FIG_DIR / "02_ten_yd_split_vs_getoff.png", bbox_inches="tight"); plt.show()

