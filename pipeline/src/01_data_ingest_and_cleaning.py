# %% [markdown]
# # 01 · Data ingest and cleaning
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
# *Do Combine first-step quickness, lateral agility and burst predict NFL pass-rush, pass-blocking and run-blocking?*
#
# **What this notebook does**
# 1. Reads the raw NFL Big Data Bowl 2027 CSV files with **explicit Spark schemas** (no schema inference on ~2 GB of tracking).
# 2. Defines the study cohort: every prospect who worked out with the **OL** or **DL** group at the 2023–2025 Combines (`line_group` = OL / DL).
# 3. Cleans each table: parses timestamps, standardises drill names, attaches season / week to plays, removes physically impossible tracking frames.
# 4. Filters the ~2 GB of in-game tracking down to the OL + DL cohort with a broadcast semi-join before anything is shuffled.
# 5. Writes every cleaned table to **Parquet** (partitioned where useful) and saves a row-count log.
#
# **Inputs:** `data/*.csv` (raw).  **Outputs:** `processed/*.parquet`, `outputs/tables/01_ingest_log.csv`.
#
# Run order: 01 → 07.

# %%
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # make config.py importable from notebooks/

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (StructType, StructField, IntegerType, LongType, DoubleType,
                               StringType, BooleanType)
import config as C

spark = C.get_spark("01_ingest")
print("Spark", spark.version, "| data dir:", C.DATA_DIR, "| processed dir:", C.PROCESSED_DIR)
T0 = time.time()

# %% [markdown]
# ## 1. Explicit schemas
# Declaring schemas up front makes reads faster, catches type drift, and documents the data. Missing values are written as `NA` in the raw files.

# %%
I, L, D, S, B = IntegerType(), LongType(), DoubleType(), StringType(), BooleanType()
def schema(*cols):
    return StructType([StructField(n, t, True) for n, t in cols])

players_schema = schema(("nfl_id", I), ("display_name", S), ("draft_year", I), ("nfl_position", S),
    ("birth_date", S), ("college_name", S), ("college_conference", S), ("draft_round", I),
    ("draft_pick_within_round", I), ("draft_overall_pick", I))

combine_schema = schema(("draft_year", I), ("nfl_id", I), ("combine_position", S), ("combine_height", D),
    ("combine_weight", D), ("hand_size", D), ("arm_length", D), ("wing_span", D), ("ten_yd_split", D),
    ("forty", D), ("vertical", D), ("broad_jump", D), ("three_cone", D), ("short_shuttle", D),
    ("bench_reps", D), ("ngs_athleticism_score", D), ("ngs_college_production_score", D), ("ngs_final_score", D))

ctrack_schema = schema(("draft_year", I), ("event_id", S), ("nfl_id", I), ("entity_type", S), ("time", S),
    ("drill_type", S), ("drill_name", S), ("attempt", I), ("x", D), ("y", D), ("s", D), ("a", D),
    ("dis", D), ("dir", D))

games_schema = schema(("game_id", L), ("game_key", L), ("season", I), ("season_type", S), ("week", I),
    ("home_team_abbr", S), ("visitor_team_abbr", S))

career_schema = schema(("nfl_id", I), ("career_offensive_snaps", I), ("career_defensive_snaps", I),
    ("career_special_teams_snaps", I), ("career_games_active", I), ("career_games_started", I),
    ("ap_all_pro_1st_team", I), ("ap_all_pro_2nd_team", I), ("pro_bowl_original_ballot", I))

pp_cols = [("game_id", L), ("play_id", I), ("team_abbr", S), ("nfl_id", I), ("lined_up_position", S),
    ("play_description", S), ("quarter", I), ("down", I), ("yards_to_go", I), ("possession_team", S),
    ("yardline_side", S), ("yardline_number", I), ("game_clock", S), ("pre_snap_home_score", I),
    ("pre_snap_visitor_score", I), ("play_direction", S), ("pre_snap_home_team_win_probability", D),
    ("pre_snap_visitor_team_win_probability", D), ("home_team_win_probability_added", D),
    ("visitor_team_win_probility_added", D), ("expected_points_added", D), ("expected_points", D),
    ("offense_formation", S), ("receiver_alignment", S), ("pass_result", S), ("pass_length", D),
    ("play_nullified_by_penalty", S), ("penalty_yards", D), ("pre_penalty_yards_gained", D),
    ("yards_gained", D), ("team_coverage_man_zone", S), ("team_coverage_type", S),
    ("in_motion_at_ball_snap", B), ("target", B), ("rec_yards", D), ("rush_yards", D),
    ("yards_after_catch", D), ("fumble", D), ("fumble_lost", D), ("route_ran", S),
    ("separation_at_pass_forward", D), ("cushion", D), ("expected_yards_after_catch", D),
    ("dropback_duration", D), ("extended_sack_allowed", D), ("pass_rushers_encountered", D),
    ("peak_pressure_probability_allowed", D), ("pressure_allowed", B), ("sack_allowed", D),
    ("time_to_pressure_allowed", D), ("quick_pressure", D), ("unblocked_pressure", D), ("blitzing", B),
    ("player_get_off", D), ("sack", D), ("tackle", D), ("assist", D), ("caused_forced_fumble", D),
    ("recovered_fumble", D), ("tackle_for_loss", D), ("time_to_pressure", D), ("time_to_qb_hurry", D),
    ("coverage_assignment", S), ("coverage_assignment_at_snap", S)]
pp_schema = schema(*pp_cols)

gtrack_schema = schema(("game_id", L), ("play_id", I), ("nfl_id", I), ("time", S), ("x", D), ("y", D),
    ("s", D), ("a", D), ("dis", D), ("o", D), ("dir", D), ("event", S))

def read_csv(name, sch):
    # multiLine + escape: play descriptions contain quoted commas/line breaks; without these
    # options Spark splits ~13k records in player_play.csv into broken rows.
    return (spark.read.option("header", True).option("nullValue", "NA").option("emptyValue", None)
            .option("multiLine", True).option("escape", '"')
            .schema(sch).csv(C.raw(name)))

players  = read_csv("players.csv", players_schema)
combine  = read_csv("combine_results.csv", combine_schema)
ctrack   = read_csv("combine_tracking.csv", ctrack_schema)
games    = read_csv("games.csv", games_schema)
career   = read_csv("player_career_successes.csv", career_schema)
pp       = read_csv("player_play.csv", pp_schema)
gtrack   = (spark.read.option("header", True).option("nullValue", "NA").schema(gtrack_schema)
            .csv(C.game_tracking_files()))
print("game tracking files:", [Path(f).name for f in C.game_tracking_files()])

log = []   # (table, stage, rows) for the ingest log
def note(table, stage, df):
    n = df.count(); log.append((table, stage, n)); print(f"{table:<22}{stage:<34}{n:>12,}")
    return n

for name, df in [("players", players), ("combine_results", combine), ("combine_tracking", ctrack),
                 ("games", games), ("player_career", career), ("player_play", pp), ("game_tracking", gtrack)]:
    note(name, "raw", df)

# %% [markdown]
# ## 2. Study cohort: Combine OL and DL groups
# The Combine position group defines the cohort, because it decides which drills a prospect ran. `DL` includes edge rushers later listed as OLB in the NFL.

# %%
cohort = (combine.filter(F.col("combine_position").isin(C.GROUPS))
          .select("nfl_id", F.col("combine_position").alias("line_group")).distinct())
cohort.groupBy("line_group").count().show()

# Duplicate-key checks (each should be 0)
for name, df, key in [("players", players, ["nfl_id"]), ("combine_results", combine, ["nfl_id"]),
                      ("games", games, ["game_id"]), ("player_play", pp, ["game_id", "play_id", "nfl_id"])]:
    dup = df.groupBy(*key).count().filter("count > 1").count()
    print(f"duplicate {key} in {name}: {dup}")

# %% [markdown]
# ## 3. Player and Combine tables
# * Age at the draft (April 25 of the draft year) from `birth_date`.
# * Undrafted free agents have no pick: `is_udfa = 1` and pick set to 260 (one past the last pick) so tree models can still use it. Draft pick is only used as an **optional control**, never as a main predictor.

# %%
players_c = (players
    .withColumn("birth_date", F.to_date("birth_date"))
    .withColumn("age_at_draft", F.round(F.datediff(F.make_date("draft_year", F.lit(4), F.lit(25)),
                                                     F.col("birth_date")) / 365.25, 2))
    .withColumn("is_udfa", F.col("draft_overall_pick").isNull().cast("int"))
    .withColumn("draft_pick_filled", F.coalesce("draft_overall_pick", F.lit(260)))
    .drop("college_name"))

line_players = (players_c.join(cohort, "nfl_id", "inner")
                .join(combine.drop("draft_year"), "nfl_id", "left")
                .join(career, "nfl_id", "left"))
note("line_players", "cohort + combine + career", line_players)
line_players.write.mode("overwrite").parquet(C.p("line_players.parquet"))

# %% [markdown]
# ## 4. Combine tracking (10 Hz)
# Cleaning steps:
# * keep `PLAYER` frames only (ball frames are not used for linemen);
# * standardise the drill-name typo `PASS_PRO-MIRROR_DRILL` → `PASS_PRO_MIRROR_DRILL` (found during EDA);
# * parse the ISO timestamp and add `t_sec` = seconds since the first frame of the attempt;
# * drop frames with impossible speed (> 12 yd/s ≈ 24.5 mph, faster than any recorded NFL player);
# * `dis` in this file does **not** match the x/y displacement (see notebook 02), so it is replaced by a step length computed from x and y.

# %%
from pyspark.sql.window import Window

ct = (ctrack.filter(F.col("entity_type") == "PLAYER")
      .join(F.broadcast(cohort), "nfl_id", "inner")
      .withColumn("drill_name", F.regexp_replace("drill_name", "PASS_PRO-MIRROR_DRILL", "PASS_PRO_MIRROR_DRILL"))
      .withColumn("ts", F.to_timestamp("time"))
      .drop("entity_type", "time"))
note("combine_tracking", "OL + DL player frames", ct)

w_ev = Window.partitionBy("event_id").orderBy("ts")
ct = (ct.withColumn("t_sec", (F.col("ts").cast("double") -
                              F.min(F.col("ts").cast("double")).over(Window.partitionBy("event_id"))))
        .withColumn("step_xy", F.sqrt((F.col("x") - F.lag("x").over(w_ev))**2 +
                                       (F.col("y") - F.lag("y").over(w_ev))**2)))
bad = ct.filter(F.col("s") > 12)
note("combine_tracking", "frames with s > 12 yd/s (dropped)", bad)
ct = ct.filter(~(F.col("s") > 12))
note("combine_tracking", "clean player frames", ct)
(ct.repartition("drill_type").write.mode("overwrite").partitionBy("drill_type")
   .parquet(C.p("combine_tracking.parquet")))

# %% [markdown]
# ## 5. Plays: attach season, week and season type
# `game_id` encodes the calendar date, so January playoff games carry the *next* year (e.g. `2026011701` belongs to the 2025 season). Joining `games.csv` gives the true `season`.

# %%
pp_keep = [c for c, _ in pp_cols if c not in (
    "route_ran", "separation_at_pass_forward", "cushion", "expected_yards_after_catch",
    "target", "rec_yards", "rush_yards", "yards_after_catch", "fumble", "fumble_lost",
    "coverage_assignment", "coverage_assignment_at_snap", "home_team_win_probability_added",
    "visitor_team_win_probility_added")]
plays = (pp.select(*pp_keep)
         .join(F.broadcast(games.select("game_id", "season", "season_type", "week",
                                         "home_team_abbr", "visitor_team_abbr")), "game_id", "left"))
print("plays with no matching game:", plays.filter(F.col("season").isNull()).count())

# play_description is kept: notebook 04 parses the run gap ("left guard", "up the middle", ...) from it
line_plays = plays.join(F.broadcast(cohort), "nfl_id", "inner")
note("player_play", "OL + DL cohort rows", line_plays)
line_plays.write.mode("overwrite").partitionBy("season").parquet(C.p("line_player_play.parquet"))

# %% [markdown]
# ## 6. In-game tracking: filter ~2 GB to the OL + DL cohort
# A broadcast join on the cohort ids keeps only OL and DL frames before anything is shuffled. The season comes from `games` (same January caveat as above).

# %%
gt = (gtrack.join(F.broadcast(cohort), "nfl_id", "inner")
      .withColumn("ts", F.to_timestamp("time")).drop("time")
      .join(F.broadcast(games.select("game_id", "season")), "game_id", "left"))
(gt.repartition("season").write.mode("overwrite").partitionBy("season")
   .parquet(C.p("line_game_tracking.parquet")))
gt_saved = spark.read.parquet(C.p("line_game_tracking.parquet"))
_ = note("game_tracking", "OL + DL cohort frames", gt_saved)

# %% [markdown]
# ## 7. Ingest log

# %%
log_df = pd.DataFrame(log, columns=["table", "stage", "rows"])
log_df.to_csv(C.TABLE_DIR / "01_ingest_log.csv", index=False)
print(f"done in {time.time() - T0:,.0f} s")
log_df
