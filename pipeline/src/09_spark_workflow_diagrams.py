# %% [markdown]
# # 09 · Spark workflow diagrams: how the analysis runs
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# Draws four diagrams of how the project's data moves through Spark, annotated with this run's real row counts, file counts and partition counts:
#
# | Figure | Shows | Course module |
# |---|---|---|
# | `09_pipeline_flowchart.png` | End-to-end data flow: 9 raw CSVs → Parquet → Combine traits / in-game movement → 5 task tables → ML Pipelines + CV → evaluation, with rows at each step and the Spark operations used | M1 big-data processing steps, M3 ingestion |
# | `09_spark_architecture_rivanna.png` | How the job runs on Rivanna: Slurm node → driver JVM (Catalyst, DAG scheduler) → task slots on every core; Parquet on scratch; and how the same code maps to a multi-node cluster | M2 driver / executors / cluster manager |
# | `09_spark_stages_shuffle.png` | One heavy step (in-game snap metrics, notebook 04) as Spark actually runs it: file partitions → tasks → shuffle (`Exchange`) → stage boundary → window functions → aggregation, i.e. map → shuffle → reduce | M1 MapReduce, M2 DAG / lazy evaluation |
# | `09_ml_pipeline_cv.png` | The Spark ML Pipeline stages, the model ladder and player-grouped `CrossValidator` with parallel fits | M6 / M8 MLlib pipelines, model tuning |
#
# The partition counts and the physical plan of the notebook-04 step are read from Spark itself (`explain`), not typed in. Notebook 10 measures what each of these design choices buys in run time.
#
# **Inputs:** `processed/*.parquet` (notebooks 01–05), `outputs/tables/01_ingest_log.csv`, `05_task_summary.csv`, `06_cv_results.csv` (optional).
# **Outputs:** `outputs/figures/09_*.png`, `outputs/tables/09_physical_plan_snap_metrics.txt`.

# %%
import sys, re, os
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from IPython.display import Image, display
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import config as C

spark = C.get_spark("09_diagrams")
CORES = spark.sparkContext.defaultParallelism
print("Spark", spark.version, "| task slots (cores):", CORES)

# palette (categorical slots in fixed order) and inks
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#257de0", "#f47230", "#00b894", "#f2a600",
                                                           "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
INK, INK2, GRID = "#191919", "#666666", "#d9d9d9"
TINT = {BLUE: "#e6f0fc", ORANGE: "#fdeee6", AQUA: "#e0f6f1", YELLOW: "#fdf4df", VIOLET: "#ecE8fb",
        GREEN: "#e0f1e3", MAGENTA: "#fcecf3", RED: "#fce6e6", INK2: "#f2f2f2"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "figure.dpi": 110, "savefig.dpi": 200})

# %% [markdown]
# ## 0. Real numbers from this run
# Row counts come from the notebook 01 ingest log and the notebook 05 task summary; file and partition counts come from Spark reading the Parquet tables.

# %%
def read_csv(name):
    f = C.TABLE_DIR / name
    return pd.read_csv(f) if f.exists() else None

def fmt(n):
    if n is None or (isinstance(n, float) and np.isnan(n)): return "?"
    n = float(n)
    return f"{n/1e6:.1f}M" if n >= 1e6 else (f"{n/1e3:.0f}k" if n >= 1e4 else f"{n:,.0f}")

log = read_csv("01_ingest_log.csv")
def logged(table, stage):
    if log is None: return None
    r = log[(log.table == table) & (log.stage == stage)]
    return None if r.empty else int(r.rows.iloc[0])

raw_bytes = {f.name: f.stat().st_size for f in C.DATA_DIR.glob("*.csv")} if C.DATA_DIR.exists() else {}
gt_bytes = sum(v for k, v in raw_bytes.items() if k.startswith("game_tracking"))

def parquet_info(name):
    path = Path(C.p(name))
    files = list(path.rglob("*.parquet"))
    parts = sorted({p.parent.name for p in files if "=" in p.parent.name})
    return dict(files=len(files), mb=sum(f.stat().st_size for f in files) / 1e6, partitions=parts,
                spark_partitions=spark.read.parquet(str(path)).rdd.getNumPartitions())

pq = {n: parquet_info(n) for n in ["line_game_tracking.parquet", "combine_tracking.parquet", "line_player_play.parquet"]}
tasks = read_csv("05_task_summary.csv")
task_rows = tasks.groupby("task").rows.sum().to_dict() if tasks is not None else {}
cv = read_csv("06_cv_results.csv")
n_fits = len(cv) if cv is not None else None

N = dict(
    gt_raw=logged("game_tracking", "raw"), gt_line=logged("game_tracking", "OL + DL cohort frames"),
    ct_raw=logged("combine_tracking", "raw"), ct_line=logged("combine_tracking", "clean player frames"),
    pp_raw=logged("player_play", "raw"), pp_line=logged("player_play", "OL + DL cohort rows"),
    players_raw=logged("players", "raw"), players_line=logged("line_players", "cohort + combine + career"))
print({k: fmt(v) for k, v in N.items()})
print(f"raw CSV total {sum(raw_bytes.values())/1e9:.2f} GB (game tracking {gt_bytes/1e9:.2f} GB)")
pd.DataFrame({k: {"files": v["files"], "MB": round(v["mb"], 1), "Spark read partitions": v["spark_partitions"],
                  "directory partitions": ", ".join(x.split("=")[1] for x in v["partitions"])[:80]}
              for k, v in pq.items()}).T

# %% [markdown]
# ## 1. Drawing helpers

# %%
def box(ax, x, y, w, h, title, body="", color=BLUE, title_size=9.5, body_size=7.6, fill=None, lw=1.4, align="left",
        mono=False):
    """Rounded box with a bold title (one or more lines) and optional body text (top-left anchored)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.012",
                                fc=fill or TINT.get(color, "#f5f5f5"), ec=color, lw=lw, zorder=2))
    tx = x + 0.008 if align == "left" else x + w / 2
    ha = "left" if align == "left" else "center"
    ax.text(tx, y + h - 0.010, title, ha=ha, va="top", fontsize=title_size, fontweight="bold", color=INK, zorder=3)
    if body:
        n_title = title.count("\n") + 1
        ax.text(tx, y + h - 0.012 - (0.010 + 0.020 * n_title) * title_size / 9.5, body, ha=ha, va="top",
                fontsize=body_size, color=INK, zorder=3, linespacing=1.35,
                family="DejaVu Sans Mono" if mono else None)

def arrow(ax, x0, y0, x1, y1, color=INK2, lw=1.3, style="-|>", ls="-", rad=0.0, label=None, lsize=7.2, loff=(0, 0.012)):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0,
                                connectionstyle=f"arc3,rad={rad}", mutation_scale=11))
    if label:
        ax.text((x0 + x1) / 2 + loff[0], (y0 + y1) / 2 + loff[1], label, ha="center", va="bottom",
                fontsize=lsize, color=INK2, zorder=3)

def canvas(w=16, h=9):
    fig = plt.figure(figsize=(w, h)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax

def save(fig, name):
    path = C.FIG_DIR / name
    fig.savefig(path, facecolor="white"); plt.close(fig)
    display(Image(str(path), width=1100))
    return path

# %% [markdown]
# ## 2. End-to-end pipeline flowchart
# Two branches (Combine drills and NFL games) are engineered separately in Spark and meet in the task tables. Each box lists the main Spark operations; numbers are this run's rows.

# %%
fig, ax = canvas(17, 9.6)
ax.text(0.012, 0.975, "Trench Play: how the data flows through Spark", fontsize=15, fontweight="bold", color=INK, va="top")
ax.text(0.012, 0.942, f"9 raw CSV files ({sum(raw_bytes.values())/1e9:.1f} GB here) → Parquet → features → 5 modeling tasks → "
        "9 Spark ML Pipelines each → test on the 2025 draft class.   Numbers are rows in this run.",
        fontsize=9.5, color=INK2, va="top")

# column x positions
X = [0.012, 0.175, 0.345, 0.535, 0.705, 0.862]; W = [0.150, 0.155, 0.175, 0.155, 0.143, 0.128]
# stage headers
heads = ["RAW (CSV)", "01 INGEST + CLEAN", "03 / 04 FEATURE ENGINEERING", "05 TASK TABLES", "06 TRAIN + TUNE", "07 EVALUATE"]
for x, w, h in zip(X, W, heads):
    ax.text(x + w / 2, 0.895, h, ha="center", fontsize=8.5, fontweight="bold", color=INK2)
    ax.plot([x, x + w], [0.885, 0.885], color=GRID, lw=1)

# --- raw inputs
gtb = f"{gt_bytes/1e9:.1f} GB" if gt_bytes else ""
box(ax, X[0], 0.700, W[0], 0.165, "combine_tracking.csv",
    f"{fmt(N['ct_raw'])} frames, 10 Hz\nposition drills of all\n510 rookies (x, y, s, a, dir, o)", ORANGE)
box(ax, X[0], 0.540, W[0], 0.135, "players · combine_results\nplayer_career_successes",
    f"{fmt(N['players_raw'])} rookies, 2023–25 classes\nsize, stopwatch drills", INK2)
box(ax, X[0], 0.330, W[0], 0.185, "game_tracking_2023–25.csv",
    f"{fmt(N['gt_raw'])} frames {gtb}\nevery NFL snap, 10 Hz\n(the big table: ~90% of bytes)", BLUE)
box(ax, X[0], 0.170, W[0], 0.135, "player_play.csv · games.csv",
    f"{fmt(N['pp_raw'])} player-snap rows\nPFF-style outcomes, play-by-play\ntext, down / distance", BLUE)

# --- ingest
box(ax, X[1], 0.170, W[1], 0.695, "Spark DataFrame ingest",
    "• explicit StructType schemas\n  (no inferSchema pass)\n• CSV options multiLine + escape\n  (quoted play text)\n"
    f"• filter to the OL + DL cohort:\n  {fmt(N['players_line'])} linemen (121 OL, 118 DL)\n"
    "• broadcast join cohort ids\n  onto the big tables\n• derived: timestamps, t_sec,\n  step length\n\n"
    "write → Parquet (columnar,\nsnappy), partitioned on disk:\n"
    f"• combine_tracking by drill_type\n  ({len(pq['combine_tracking.parquet']['partitions'])} folders, {fmt(N['ct_line'])} frames)\n"
    f"• line_game_tracking by season\n  ({len(pq['line_game_tracking.parquet']['partitions'])} folders, {fmt(N['gt_line'])} frames,\n"
    f"  {pq['line_game_tracking.parquet']['mb']:.0f} MB)\n"
    f"• line_player_play\n  ({fmt(N['pp_line'])} rows)\n\n02 EDA reads these tables:\ndictionary, quality, figures", GREEN)

# --- feature engineering
box(ax, X[2], 0.560, W[2], 0.305, "03 Combine traits",
    "Window.partitionBy(player, drill, attempt)\n.orderBy(time):\n• smoothed speed, signed accel (lag)\n• movement onset, direction reversals\n"
    "groupBy attempt → metrics:\n  t_1yd, t_2yd, s03, s05 · rev_time, brake,\n  re-accel · acc_peak, s10, d10\n"
    "→ best per drill → z-scores (fit on\n  2023–24 only) → 3 trait indices\n  first step · lateral agility · burst", ORANGE)
box(ax, X[2], 0.170, W[2], 0.365, "04 In-game movement + run credit",
    "• snap frame per (game, play, player)\n  → rangeBetween windows on t (0.1 s)\n• get-off, speed at 0.5 / 1.0 s, lateral\n"
    "• OL side from y at the snap vs the\n  hash marks + season majority vote\n• run gap from play text (regexp_extract)\n"
    "  → credit 1.0 at the gap, 0.5 adjacent\n• translation check: Combine trait vs\n  in-game counterpart (Spearman ρ)\n\n"
    "aggregates 12M frames to one row\nper lineman-snap", BLUE)

# --- tasks
rows = lambda t: fmt(task_rows.get(t))
box(ax, X[3], 0.170, W[3], 0.695, "One table per task",
    "join snaps × trait indices × size\n+ pre-snap context (down, distance,\n  score, formation, alignment …)\n\n"
    f"DL pass rush\n• dl_pressure   {rows('dl_pressure')} snaps\n• dl_getoff       {rows('dl_getoff')} snaps\n\n"
    f"OL pass blocking\n• ol_pressure        {rows('ol_pressure')}\n• ol_pressure_prob {rows('ol_pressure_prob')}\n\n"
    f"OL run blocking\n• ol_run   {rows('ol_run')} (weight = credit)\n\n"
    "Split by PLAYER\n• test = 2025 draft class\n• validation = 20% of 2023–24\n• train → 5 player-grouped folds\n"
    "• class weights from train only", VIOLET)

# --- train
box(ax, X[4], 0.170, W[4], 0.695, "Spark ML Pipelines",
    "StringIndexer → OneHotEncoder\n→ Imputer → VectorAssembler\n→ StandardScaler → estimator\n\n"
    "model ladder (per task)\n• benchmark: context only\n• + first step / lateral / burst\n• + all traits · stopwatch · size\n"
    "• Random Forest · GBT\n\nCrossValidator(foldCol='fold')\nparam grids, parallel fits\n"
    + (f"\n{n_fits} (task, model, setting)\nCV results" if n_fits else "") + "\n\nbest model refit on all train", AQUA)

# --- evaluate
box(ax, X[5], 0.170, W[5], 0.695, "Held-out evaluation",
    "validation → champion\n2025 class = test only\n\nAUROC, accuracy, P / R / F1,\nconfusion matrix, lift\nR², RMSE, MAE\n\n"
    "player bootstrap CIs\npaired Δ vs benchmark\n(does a trait add signal?)\n\nsensitivity: depth ×\nmin snaps per leaf\n\n"
    "08 animations · 09 / 10\nthis view + Spark\nperformance", RED)

# arrows
arrow(ax, X[0] + W[0], 0.78, X[1], 0.78); arrow(ax, X[0] + W[0], 0.60, X[1], 0.60)
arrow(ax, X[0] + W[0], 0.42, X[1], 0.42); arrow(ax, X[0] + W[0], 0.24, X[1], 0.24)
arrow(ax, X[1] + W[1], 0.71, X[2], 0.71)
arrow(ax, X[1] + W[1], 0.35, X[2], 0.35)
arrow(ax, X[2] + W[2], 0.71, X[3], 0.62)
arrow(ax, X[2] + W[2], 0.35, X[3], 0.42)
arrow(ax, X[3] + W[3], 0.52, X[4], 0.52); arrow(ax, X[4] + W[4], 0.52, X[5], 0.52)

# legend / footer
ax.text(0.012, 0.115, "Every box runs as Spark jobs on one Rivanna node (local mode, all cores). Only small "
        "results (≤ a few thousand rows: player tables, metrics, figures) are pulled to pandas with toPandas().",
        fontsize=8.5, color=INK2)
ax.text(0.012, 0.085, "Lazy evaluation: each notebook builds a plan of transformations; nothing runs until an action "
        "(count, write.parquet, fit, toPandas). Parquet between notebooks = checkpoints that cut the lineage.",
        fontsize=8.5, color=INK2)
for i, (c, t) in enumerate([(ORANGE, "Combine branch"), (BLUE, "NFL game branch"), (GREEN, "ingest"),
                            (VIOLET, "task tables"), (AQUA, "MLlib"), (RED, "evaluation")]):
    ax.add_patch(Rectangle((0.012 + i * 0.105, 0.035), 0.014, 0.022, fc=TINT[c], ec=c, lw=1.2))
    ax.text(0.031 + i * 0.105, 0.046, t, va="center", fontsize=8.5, color=INK)
save(fig, "09_pipeline_flowchart.png")

# %% [markdown]
# ## 3. How the job runs on Rivanna (driver, executors, task slots)
# On one Rivanna node Spark runs in **local mode**: the driver and a single executor share one JVM, and every core is a task slot. The right panel shows how the same notebooks would map onto a multi-node cluster (the course's driver / cluster-manager / executor picture): no code changes, only `master` and the executor settings.

# %%
slots = int(os.environ.get("SLURM_CPUS_PER_TASK", 16))
fig, ax = canvas(16, 8.6)
ax.text(0.012, 0.975, "How the pipeline runs on UVA Rivanna", fontsize=15, fontweight="bold", color=INK, va="top")
ax.text(0.012, 0.942, "One Slurm job → one node → Spark in local mode: driver + executor in one JVM, one task per core. "
        "Same code scales out by changing only the master URL.", fontsize=9.5, color=INK2, va="top")

# user + slurm
box(ax, 0.012, 0.700, 0.140, 0.180, "You (login node /\nOpen OnDemand)",
    "sbatch run_pipeline.slurm\n-A ds7200-apt4c\n-c 16  --mem=64G", INK2)
box(ax, 0.012, 0.430, 0.140, 0.220, "Slurm scheduler",
    "queues the job, then\nallocates 1 'standard'\nnode: 16 cores, 64 GB\n\nmodule load spark\nmodule load miniforge", INK2)
arrow(ax, 0.082, 0.700, 0.082, 0.650)
arrow(ax, 0.152, 0.540, 0.190, 0.540)

# node
ax.add_patch(FancyBboxPatch((0.190, 0.115), 0.430, 0.780, boxstyle="round,pad=0,rounding_size=0.012",
                            fc="white", ec=INK2, lw=1.6, ls="--"))
ax.text(0.200, 0.880, "Compute node (one Slurm allocation)", fontsize=10, fontweight="bold", color=INK, va="top")
box(ax, 0.205, 0.690, 0.185, 0.160, "Python process",
    "jupyter nbconvert runs each\nnotebook; PySpark API calls\ngo to the JVM via Py4J;\ntoPandas() for small results", VIOLET)
box(ax, 0.405, 0.540, 0.200, 0.310, "Driver (SparkSession)",
    "• Catalyst optimizer:\n  logical plan → physical plan\n  (column pruning, predicate\n  pushdown, broadcast joins)\n"
    "• DAG scheduler: plan → stages\n  split at every shuffle\n• task scheduler: 1 task per\n  partition, per free core\n• Spark UI :4040", BLUE)
arrow(ax, 0.390, 0.770, 0.405, 0.770, label="Py4J", loff=(0, 0.006))
# executor with slots
ax.add_patch(FancyBboxPatch((0.205, 0.265), 0.400, 0.255, boxstyle="round,pad=0,rounding_size=0.012",
                            fc=TINT[AQUA], ec=AQUA, lw=1.4))
ax.text(0.213, 0.510, f"Executor (same JVM in local mode): {slots} task slots = local[*]", fontsize=9.5,
        fontweight="bold", color=INK, va="top")
cols_ = 8; rows_ = int(np.ceil(slots / cols_))
for i in range(slots):
    r, c = divmod(i, cols_)
    ax.add_patch(Rectangle((0.218 + c * 0.047, 0.420 - r * 0.062), 0.040, 0.048, fc="white", ec=AQUA, lw=1.1))
    ax.text(0.238 + c * 0.047, 0.444 - r * 0.062, f"core {i+1}\ntask", ha="center", va="center", fontsize=6.3, color=INK)
ax.text(0.213, 0.300, "memory: spark.driver.memory = 48g (run_pipeline.slurm) → execution (shuffles, sorts, joins)\n"
        "+ storage (cache / persist); spill to local disk when full", fontsize=7.6, color=INK, va="top")
arrow(ax, 0.505, 0.540, 0.505, 0.520, label="tasks", loff=(0.02, -0.008))

# storage
box(ax, 0.205, 0.130, 0.190, 0.115, "/standard/ds7200-apt4c/…",
    "raw CSVs (read once, notebook 01)\nnotebooks, outputs/", ORANGE)
box(ax, 0.410, 0.130, 0.195, 0.115, "/scratch/$USER/…/processed",
    "Parquet tables between notebooks\n(fast parallel file system)", ORANGE)
arrow(ax, 0.300, 0.245, 0.300, 0.265, style="<|-|>"); arrow(ax, 0.507, 0.245, 0.507, 0.265, style="<|-|>")

# cluster view
ax.add_patch(FancyBboxPatch((0.650, 0.115), 0.338, 0.780, boxstyle="round,pad=0,rounding_size=0.012",
                            fc="#fafafa", ec=GRID, lw=1.2))
ax.text(0.660, 0.880, "Same code on a multi-node cluster", fontsize=10, fontweight="bold", color=INK, va="top")
ax.text(0.660, 0.850, "master = yarn / spark://host:7077 instead of local[*]", fontsize=8, color=INK2, va="top")
box(ax, 0.665, 0.690, 0.140, 0.120, "Driver", "plans + schedules\n(as on the left)", BLUE)
box(ax, 0.830, 0.690, 0.145, 0.120, "Cluster manager", "YARN / Standalone /\nSlurm-launched", INK2)
arrow(ax, 0.805, 0.750, 0.830, 0.750, style="<|-|>")
for j in range(3):
    xx = 0.668 + j * 0.107
    box(ax, xx, 0.380, 0.098, 0.230, f"Worker node {j+1}", "", AQUA)
    ax.text(xx + 0.006, 0.555, "Executor", fontsize=7.6, color=INK, fontweight="bold")
    for k in range(4):
        r, c = divmod(k, 2)
        ax.add_patch(Rectangle((xx + 0.010 + c * 0.042, 0.490 - r * 0.050), 0.036, 0.040, fc="white", ec=AQUA, lw=1, zorder=3))
        ax.text(xx + 0.028 + c * 0.042, 0.510 - r * 0.050, "task", ha="center", va="center", fontsize=6.2, zorder=4)
    ax.text(xx + 0.006, 0.405, "cache · shuffle files", fontsize=6.6, color=INK2)
    arrow(ax, 0.735, 0.690, xx + 0.049, 0.610, lw=1.0)
ax.text(0.665, 0.335, "• data partitions are spread over the\n  executors; a shuffle moves rows\n  between nodes over the network\n"
        "• the course sizing rule: ≤ 5 cores per\n  executor, leave 1 core + 1 GB per node\n  for the OS, 1 executor for the driver\n"
        "• why not here: Rivanna's own guidance is\n  local mode for one node; our largest\n  table (12M frames) fits in one node's RAM",
        fontsize=7.8, color=INK, va="top", linespacing=1.4)
save(fig, "09_spark_architecture_rivanna.png")

# %% [markdown]
# ## 4. One heavy step as Spark runs it: stages and the shuffle
# The core of notebook 04 is: find each player's snap frame, align every frame to it, and compute speed and movement metrics with window functions, then reduce to one row per lineman-snap. Below, Spark plans the same computation; the diagram is drawn from the operators in its physical plan.

# %%
gt = spark.read.parquet(C.p("line_game_tracking.parquet"))
key = ["game_id", "play_id", "nfl_id"]
snap = gt.filter(F.col("event") == "ball_snap").groupBy(*key).agg(F.min(F.col("ts").cast("double")).alias("snap_t"))
w = Window.partitionBy(*key).orderBy("ts")
g = (gt.join(snap, key, "inner")
       .withColumn("t_rel", F.round(F.col("ts").cast("double") - F.col("snap_t"), 2))
       .filter((F.col("t_rel") >= 0) & (F.col("t_rel") <= 1.5))
       .withColumn("s_sm", F.avg("s").over(w.rowsBetween(-1, 1)))
       .withColumn("step", F.sqrt((F.col("x") - F.lag("x").over(w)) ** 2 + (F.col("y") - F.lag("y").over(w)) ** 2)))
mv = g.groupBy(*key).agg(F.max("s_sm").alias("s_max"), F.sum("step").alias("d15"),
                         F.max(F.when(F.col("t_rel") <= 1.0, F.col("s_sm"))).alias("s10"))
plan = mv._jdf.queryExecution().executedPlan().toString()
(C.TABLE_DIR / "09_physical_plan_snap_metrics.txt").write_text(plan)
ops = [ln.strip().lstrip(":+- ").split(" ")[0].split("(")[0] for ln in plan.splitlines() if ln.strip()]
ops = [o for o in ops if o and o[0].isalpha()]
n_exchange = sum(o in ("Exchange", "ShuffleExchange") for o in ops)
print("operators in the physical plan (top = last):"); print(pd.Series(ops).value_counts().to_string())
print("\nshuffles (Exchange):", n_exchange, "| broadcast:", any("Broadcast" in o for o in ops))
print("\n".join(plan.splitlines()[:40]))

# %%
n_in = pq["line_game_tracking.parquet"]["spark_partitions"]
n_sh = int(spark.conf.get("spark.sql.shuffle.partitions"))
aqe = spark.conf.get("spark.sql.adaptive.enabled")
read_schemas = re.findall(r"ReadSchema: struct<([^>]*)>", plan)
n_cols = [len(r.split(",")) for r in read_schemas]
n_all_cols = len(gt.columns)
pushed = re.findall(r"PushedFilters: \[([^\]]*)", plan)
snap_push = next((p for p in pushed if "ball_snap" in p), "")
n_out = mv.count()
print(f"columns read per scan: {n_cols} of {n_all_cols} | pushed to the snap scan: {snap_push[:90]} | output rows {n_out:,}")

fig, ax = canvas(16, 8.6)
ax.text(0.012, 0.975, "Inside one step: how Spark executes the in-game snap metrics (notebook 04)", fontsize=15,
        fontweight="bold", color=INK, va="top")
ax.text(0.012, 0.942, f"Drawn from the physical plan (explain): two map stages, {n_exchange} shuffle exchanges, one reduce stage. "
        "Narrow operations are pipelined inside a stage; each Exchange is a stage boundary (MapReduce's shuffle).",
        fontsize=9.5, color=INK2, va="top")

def stage_panel(x, y, w, h, color, title, ops, task_side, n_tasks, n_show, task_label):
    """Stage box: operator chain + a column of task boxes on one side. Returns task box anchor points."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=TINT[color], ec=color, lw=1.5))
    ax.text(x + 0.010, y + h - 0.012, title, fontsize=9.8, fontweight="bold", color=INK, va="top")
    tw = 0.070
    tx = x + w - tw - 0.012 if task_side == "right" else x + 0.012
    ox = x + 0.012 if task_side == "right" else x + tw + 0.026
    ow = w - tw - 0.038
    top = y + h - 0.058
    step = min(0.052, (h - 0.075) / len(ops))
    for i, op in enumerate(ops):
        yy = top - i * step
        ax.add_patch(FancyBboxPatch((ox, yy - step * 0.74), ow, step * 0.74, boxstyle="round,pad=0,rounding_size=0.005",
                                    fc="white", ec=color, lw=0.9, zorder=3))
        ax.text(ox + 0.006, yy - step * 0.37, op, fontsize=7.6, va="center", color=INK, zorder=4)
        if i < len(ops) - 1:
            arrow(ax, ox + ow / 2, yy - step * 0.74, ox + ow / 2, yy - step, lw=0.8)
    # task column
    ax.text(tx + tw / 2, top + 0.004, task_label, ha="center", va="bottom", fontsize=7.2, color=INK, fontweight="bold")
    th = (h - 0.090) / n_show
    anchors = []
    for i in range(n_show):
        ty = top - (i + 1) * th + 0.004
        ax.add_patch(Rectangle((tx, ty), tw, th - 0.008, fc="white", ec=color, lw=1, zorder=3))
        lbl = f"task {i+1}" if i < n_show - 1 or n_show == n_tasks else f"… task {n_tasks}"
        ax.text(tx + tw / 2, ty + (th - 0.008) / 2, lbl, ha="center", va="center", fontsize=6.8, zorder=4)
        anchors.append((tx + tw if task_side == "right" else tx, ty + (th - 0.008) / 2))
    return anchors

cols_txt = f"{n_cols[0]} of {n_all_cols}" if n_cols else "only the needed"
m1 = stage_panel(0.012, 0.525, 0.330, 0.385, BLUE, f"Stage 1a · scan frames · {n_in} input partitions",
                 ["FileScan parquet: 12M frames, season folders", f"ReadSchema: {cols_txt} columns (column pruning)",
                  "Filter isnotnull(keys)", "Project game, play, nfl_id, x, y, s, ts",
                  "Exchange hashpartitioning(game, play, nfl_id)"], "right", n_in, min(n_in, 5), "map tasks")
snap_txt = "PushedFilters: event = 'ball_snap'" if snap_push else "Filter event = 'ball_snap'"
m2 = stage_panel(0.012, 0.120, 0.330, 0.385, BLUE, f"Stage 1b · scan snap events · {n_in} input partitions",
                 ["FileScan parquet (same files)", f"{snap_txt} (predicate pushdown)",
                  "HashAggregate partial_min(ts)  ← combiner", "Exchange hashpartitioning(game, play, nfl_id)"],
                 "right", n_in, min(n_in, 5), "map tasks")
r = stage_panel(0.475, 0.120, 0.330, 0.790, AQUA, f"Stage 2 · reduce · {n_sh} shuffle partitions" + (" (AQE may coalesce)" if aqe == "true" else ""),
                ["HashAggregate final min(ts) → snap time", "Sort (both sides) by game, play, nfl_id",
                 "SortMergeJoin frames ⋈ snap time", "Filter 0 ≤ t_rel ≤ 1.5 s after the snap",
                 "Window: avg(s) rows −1..+1 (smoothing)", "Window: lag(x), lag(y) → step length",
                 "HashAggregate by (game, play, nfl_id)", "no new Exchange: same key as the join",
                 "→ count() / write.parquet"],
                "left", n_sh, min(n_sh, 9), "reduce tasks")
for (mx, my) in m1 + m2:
    for (rx, ry) in r:
        ax.plot([mx, rx], [my, ry], color=ORANGE, lw=0.5, alpha=0.5, zorder=1)
ax.text(0.4085, 0.515, "SHUFFLE\nall-to-all", ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK,
        bbox=dict(fc="white", ec=ORANGE, lw=1, boxstyle="round,pad=0.35"), zorder=5)

box(ax, 0.825, 0.555, 0.163, 0.355, "Result", f"{n_out:,} lineman-snaps\n(one row each:\ns_max, s10, d15 …)\n\n"
    "12M frames in →\nshuffled by key →\nreduced to one row\nper key", GREEN)
box(ax, 0.825, 0.120, 0.163, 0.410, "How to read it",
    "narrow (no data movement):\nscan, filter, project,\nwindow within a key\n\n"
    "wide (shuffle = new stage):\nExchange before the\njoin and the groupBy\n\n"
    f"tasks per stage =\npartitions; {CORES} run at\nonce (one per core)\n\n"
    "the partial min before\nthe shuffle is Spark's\nMapReduce combiner", INK2)
arrow(ax, 0.805, 0.73, 0.825, 0.73)
ax.text(0.012, 0.075, "Why the shuffle key is (game, play, player): ~170k distinct keys spread evenly over the reduce tasks, so no "
        "task gets a 'hot' key (notebook 10 measures the skew if the key were the player alone).", fontsize=8.5, color=INK2)
ax.text(0.012, 0.045, "Lazy evaluation: building this query ran nothing; the three stages only execute when an action "
        "(count / write) is called. The full plan text is in outputs/tables/09_physical_plan_snap_metrics.txt.",
        fontsize=8.5, color=INK2)
save(fig, "09_spark_stages_shuffle.png")

# %% [markdown]
# ## 5. Spark ML Pipeline and cross-validation (notebook 06)

# %%
fig, ax = canvas(16, 7.2)
ax.text(0.012, 0.970, "Modeling in Spark MLlib: one Pipeline per model, tuned with player-grouped CrossValidator",
        fontsize=15, fontweight="bold", color=INK, va="top")
ax.text(0.012, 0.930, "Every transformer is fitted inside each CV fold (no leakage from the held-out fold); "
        "the 2025 class is never touched until notebook 07.", fontsize=9.5, color=INK2, va="top")
stages = [("task table", "Parquet, one row\nper lineman-snap", VIOLET),
          ("StringIndexer", "down, distance\nbucket, formation,\nalignment, position", BLUE),
          ("OneHotEncoder", "→ sparse vectors\n(mostly zeros)", BLUE),
          ("Imputer", "median, fitted on\nfitting folds only", BLUE),
          ("VectorAssembler", "context + chosen\ntrait / size block", BLUE),
          ("StandardScaler", "linear models only", BLUE),
          ("Estimator", "LogisticRegression /\nLinearRegression,\nRandomForest, GBT", AQUA)]
bw, gap = 0.118, 0.022
for i, (t, b, c) in enumerate(stages):
    x = 0.012 + i * (bw + gap)
    box(ax, x, 0.655, bw, 0.200, t, b, c)
    if i < len(stages) - 1:
        arrow(ax, x + bw, 0.755, x + bw + gap, 0.755)
ax.add_patch(FancyBboxPatch((0.150, 0.630), 0.840, 0.250, boxstyle="round,pad=0,rounding_size=0.012",
                            fc="none", ec=INK2, lw=1, ls="--"))
ax.text(0.155, 0.640, "pyspark.ml.Pipeline  (fit() on a DataFrame → PipelineModel)", fontsize=8, color=INK2)

# CV folds
ax.text(0.012, 0.560, "CrossValidator(foldCol='fold')", fontsize=10.5, fontweight="bold", color=INK)
ax.text(0.012, 0.530, "folds are groups of PLAYERS, so a player's snaps are never in both the fitting and the scoring part",
        fontsize=8, color=INK2)
for f in range(5):
    for j in range(5):
        x = 0.012 + j * 0.060; y = 0.440 - f * 0.065
        held = (j == f)
        ax.add_patch(Rectangle((x, y), 0.055, 0.050, fc=TINT[ORANGE] if held else TINT[BLUE],
                               ec=ORANGE if held else BLUE, lw=1))
        ax.text(x + 0.0275, y + 0.025, "score" if held else "fit", ha="center", va="center", fontsize=7)
    ax.text(0.320, 0.465 - f * 0.065, f"fold {f}", fontsize=7.5, color=INK2, va="center")
ax.text(0.012, 0.095, "× every parameter setting (regParam; depth × min snaps per leaf; …)", fontsize=8, color=INK)
cvp = os.environ.get("CV_PARALLELISM", "4")
box(ax, 0.380, 0.110, 0.285, 0.400, f"Parallel fits (parallelism = {cvp})",
    f"CrossValidator sends up to {cvp} (fold, setting)\nfits to Spark at once; each fit is itself\n"
    "a set of distributed jobs over the rows\n(data parallelism: every task computes\npartial gradients / split statistics on its\n"
    "partition, the driver combines them).\n\nRF / GBT: tree-level statistics are\naggregated per partition (histograms of\n"
    "binned features), then reduced.\n\naverage fold metric → best setting →\nrefit on all training players", AQUA)
box(ax, 0.690, 0.110, 0.300, 0.400, "Model ladder per task (9 models)",
    "benchmark    context only\nfirst_step   context + trk_first_step\nlateral      context + trk_lateral\n"
    "burst        context + trk_burst\ntraits       context + all 3 tracking traits\n"
    "stopwatch    context + stopwatch traits\ntraits_size  + size-adjusted traits, size\n"
    "rf           Random Forest, all features\ngbt          gradient-boosted trees\n\n"
    "classification: AUROC   regression: R²\n→ validation picks the champion\n→ 2025 class scored once (notebook 07)", RED,
    mono=True, body_size=7.4)
arrow(ax, 0.330, 0.300, 0.380, 0.300)
arrow(ax, 0.665, 0.300, 0.690, 0.300)
save(fig, "09_ml_pipeline_cv.png")

# %%
print("figures written:", sorted(p.name for p in C.FIG_DIR.glob("09_*.png")))
spark.stop()
