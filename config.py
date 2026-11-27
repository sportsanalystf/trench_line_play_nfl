"""
config.py -- shared paths, constants and the Spark session builder for every notebook.

Project: Trench play -- do Combine first-step quickness, lateral agility and burst
predict NFL pass-rush (DL), pass-blocking (OL) and run-blocking (OL) performance?

Every notebook in pipeline/ imports this file, so all
paths are defined in one place. Nothing here needs editing; override with
environment variables instead:

    BDB_DATA_DIR          folder holding the raw Big Data Bowl 2027 CSVs. Default: data/
                          inside the repo if it holds the CSVs, otherwise ../data next to
                          the repo (on Rivanna: /standard/ds7200-apt4c/DS5110_Fall26_Group3/data)
    BDB_PROCESSED_DIR     where cleaned Parquet tables are written (default: ./processed)
    SPARK_DRIVER_MEMORY   e.g. "32g" on a Rivanna node (default "6g")
    SPARK_MASTER          default "local[*]" (all cores of the node you were given)
    CV_PARALLELISM        models fitted in parallel during cross-validation (default 2)
    QUICK=1               1-setting grids, for testing the code end to end
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
def _default_data_dir():
    inside = PROJECT_ROOT / "data"
    return inside if any(inside.glob("*.csv")) else PROJECT_ROOT.parent / "data"

DATA_DIR = Path(os.environ.get("BDB_DATA_DIR", _default_data_dir()))
PROCESSED_DIR = Path(os.environ.get("BDB_PROCESSED_DIR", PROJECT_ROOT / "processed"))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"
for _d in (PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 5110
N_FOLDS = 5
GROUPS = ["OL", "DL"]            # Combine position groups in the study
TEST_DRAFT_CLASS = 2025          # held-out class = test set
VALIDATION_FRACTION = 0.20       # share of 2023-24 players held out for validation

# NFL hash marks (yards from the sideline at y = 0): the ball is always spotted between them
HASH_LO, HASH_HI = 23.58, 29.75

# ---------------------------------------------------------------------------
# Which Combine drills measure which trait, per position group.
#   first_step : quickness of the first 0.3-0.5 s after the player starts moving
#   lateral    : change of direction / side-to-side movement (reversals)
#   burst      : acceleration and ground covered in the first 1-1.5 s
# Tracked 3-cone / shuttle are added to `lateral` for the ~40% who ran them.
# ---------------------------------------------------------------------------
TRAIT_DRILLS = {
    "OL": {
        "first_step": ["PASS_RUSH_DROP", "OL_PULL_DRILL_FOLD_BLOCK_RIGHT", "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT",
                       "OL_PULL_DRILL_LONG_PULL_RIGHT", "FORTY_YARD_DASH"],
        "lateral":    ["FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", "PASS_PRO_MIRROR_DRILL", "SHORT_SHUTTLE", "THREE_CONE_DRILL"],
        "burst":      ["OL_PULL_DRILL_FOLD_BLOCK_RIGHT", "OL_PULL_DRILL_DEEP_SHORT_PULL_LEFT", "OL_PULL_DRILL_LONG_PULL_RIGHT",
                       "SCREEN_DRILL", "FORTY_YARD_DASH"],
    },
    "DL": {
        "first_step": ["PASS_RUSH_DRILL", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "RUN_AND_CLUB_DRILL",
                       "FORTY_YARD_DASH"],
        "lateral":    ["FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "FOUR_BAG_AGILITY_DRILL", "BODY_CONTROL_DRILL",
                       "SHORT_SHUTTLE", "THREE_CONE_DRILL"],
        "burst":      ["PASS_RUSH_DRILL", "RUN_AND_CLUB_DRILL", "RUN_THE_HOOP_DRILL", "FORTY_YARD_DASH"],
    },
}
# Per-attempt metrics used for each trait; +1 = higher is better, -1 = lower is better
TRAIT_METRICS = {
    "first_step": {"t_1yd": -1, "t_2yd": -1, "s03": 1, "s05": 1},
    "lateral":    {"rev_time": -1, "rev_brake": -1, "rev_reacc": 1, "move_speed": 1},
    "burst":      {"acc_peak": 1, "s10": 1, "d10": 1},
}
# Traditional stopwatch comparators for each trait (+1 higher is better)
TRAD_TRAITS = {
    "first_step": {"ten_yd_split": -1},
    "lateral":    {"short_shuttle": -1, "three_cone": -1},
    "burst":      {"broad_jump": 1, "vertical": 1, "forty": -1},
}
TRAITS = ["first_step", "lateral", "burst"]


def p(name: str) -> str:
    """Path (string) of a processed Parquet table."""
    return str(PROCESSED_DIR / name)


def raw(name: str) -> str:
    """Path (string) of a raw CSV file."""
    return str(DATA_DIR / name)


def game_tracking_files():
    """All in-game tracking CSVs (game_tracking_2023.csv ... 2025.csv, or split parts)."""
    files = sorted(str(f) for f in DATA_DIR.glob("game_tracking_*.csv"))
    if not files:
        raise FileNotFoundError(f"no game_tracking_*.csv in {DATA_DIR}")
    return files


def get_spark(app_name: str = "ds5110_trench"):
    """Create (or reuse) a local SparkSession sized from environment variables."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "6g"))
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "32"))
        .config("spark.sql.session.timeZone", "UTC")
        # Spark 4 turns ANSI mode on by default (errors on divide-by-zero / bad casts);
        # the feature code relies on Spark 3 semantics (NULL instead of an error).
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")  # Arrow needs extra JVM flags on Java 17+
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
