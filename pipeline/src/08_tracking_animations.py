# %% [markdown]
# # 08 · Tracking animations: seeing the traits and the outcomes
# **DS 5110 Big Data Systems — Final Project: Trench Play (OL and DL)**
#
# **What this notebook does**
# Renders short animations from the 10 Hz tracking so a viewer can *see* what every metric in the project measures:
#
# | File | What it shows |
# |---|---|
# | `combine_<group>_<trait>.mp4` (6 clips) | Two Combine prospects side by side, one near the **top** (90th percentile) and one near the **bottom** (10th) of their group on that drill. Left: their paths, aligned so both start at the origin and move "up". Right: speed and signed-acceleration traces drawn in sync. Every metric of the trait is marked the moment it happens. |
# | `game_dl_pass_rush.mp4` | NFL pass-rush snaps for a high-burst and a low-burst DL (by Combine index): movement from the snap, get-off, speed at 0.5 s / 1.0 s, and whether the rep ended in pressure. |
# | `game_ol_run_credit.mp4` | A real run play with two rookie linemen tracked: their positions against the hash marks (how left/right is inferred), the gap recorded in the play-by-play, and the credit each lineman receives. |
#
# Each clip also gets a still **storyboard** PNG (key frames) for the report and slides.
#
# The metric definitions match notebooks 03 and 04 exactly; the small per-frame calculations are redone here in pandas, for only the handful of attempts being drawn. Players are shown anonymously as "Player A / B" with their NFL position.
#
# **Inputs:** `processed/combine_tracking.parquet`, `trait_attempts.parquet`, `trait_indices.parquet`, `line_players.parquet`, `line_game_tracking.parquet`, `line_player_play.parquet`, `ingame_snaps.parquet`
# **Outputs:** `outputs/animations/*.mp4` (or `.gif` if no ffmpeg is available), `outputs/animations/*_storyboard.png`
#
# Needs `imageio-ffmpeg` for MP4 (`pip install --user imageio-ffmpeg`; it ships its own ffmpeg). Without it the clips are written as GIFs.

# %%
import sys, shutil, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import config as C

spark = C.get_spark("08_animations")
ANIM_DIR = C.OUTPUT_DIR / "animations"; ANIM_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
COL_A, COL_B = "#2a6fdb", "#e07b39"          # Player A (top), Player B (bottom)
FPS, SLOW = 10, 0.5                           # tracking is 10 Hz; SLOW = playback speed (0.5 = half speed)
T0 = time.time()

try:                                          # MP4 through imageio-ffmpeg's bundled binary, else system ffmpeg, else GIF
    import imageio_ffmpeg
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    HAVE_FFMPEG = True
except ImportError:
    HAVE_FFMPEG = shutil.which("ffmpeg") is not None
print("video format:", "mp4" if HAVE_FFMPEG else "gif")

def save(anim, name, fig):
    fps = int(FPS * SLOW)
    if HAVE_FFMPEG:
        path = ANIM_DIR / f"{name}.mp4"
        anim.save(path, writer=animation.FFMpegWriter(fps=fps, bitrate=2400, codec="libx264",
                                                       extra_args=["-pix_fmt", "yuv420p"]))
    else:
        path = ANIM_DIR / f"{name}.gif"
        anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print("saved", path.name)
    return path

# %% [markdown]
# ## 1. Per-frame kinematics (same definitions as notebook 03)

# %%
def kinematics(df):
    """df: frames of ONE attempt or snap, columns ts (timestamp), x, y, s. Returns a copy with kinematics."""
    d = df.sort_values("ts").reset_index(drop=True).copy()
    d["t"] = (d.ts - d.ts.iloc[0]).dt.total_seconds()
    d["s_sm"] = d.s.rolling(3, center=True, min_periods=1).mean()
    d["a_tan"] = (d.s_sm.shift(-1) - d.s_sm.shift(1)) / (d.t.shift(-1) - d.t.shift(1))
    K = 4
    ux1, uy1 = d.x - d.x.shift(K), d.y - d.y.shift(K)
    ux2, uy2 = d.x.shift(-K) - d.x, d.y.shift(-K) - d.y
    n1, n2 = np.hypot(ux1, uy1), np.hypot(ux2, uy2)
    cos = (ux1 * ux2 + uy1 * uy2) / (n1 * n2)
    d["rev_flag"] = (n1 > 0.3) & (n2 > 0.3) & (cos < -0.5)
    sflag = d.s_sm.where(d.rev_flag)
    d["apex"] = d.rev_flag & (d.s_sm == sflag.rolling(7, center=True, min_periods=1).min())
    return d

def onset_align(d, onset_speed=0.5):
    """Time from first movement; positions translated to start at 0 and rotated so the first 0.5 s points up."""
    i0 = d.index[d.s_sm > onset_speed][0]
    d = d.copy(); d["t_rel"] = (d.t - d.t[i0]).round(2)
    x0, y0 = d.x[i0], d.y[i0]
    j = d.index[(d.t_rel >= 0.5)][0] if (d.t_rel >= 0.5).any() else d.index[-1]
    ang = np.arctan2(d.y[j] - y0, d.x[j] - x0)
    rot = np.pi / 2 - ang
    X, Y = d.x - x0, d.y - y0
    d["px"] = X * np.cos(rot) - Y * np.sin(rot); d["py"] = X * np.sin(rot) + Y * np.cos(rot)
    d["disp"] = np.hypot(X, Y)
    return d

def attempt_metrics(d):
    m = {}
    post = d[d.t_rel >= 0]
    for r, name in [(1.0, "t_1yd"), (2.0, "t_2yd")]:
        hit = post[post.disp >= r]; m[name] = hit.t_rel.iloc[0] if len(hit) else np.nan
    for t, name in [(0.3, "s03"), (0.5, "s05"), (1.0, "s10")]:
        near = post[(post.t_rel - t).abs() < 0.051]; m[name] = near.s_sm.iloc[0] if len(near) else np.nan
    first = post[post.t_rel <= 1.5]
    m["acc_peak"] = first.a_tan.max(); m["t_acc_peak"] = first.t_rel[first.a_tan.idxmax()] if first.a_tan.notna().any() else np.nan
    m["d10"] = np.hypot(post.px.diff(), post.py.diff())[post.t_rel <= 1.0].sum()
    revs = []
    for i in d.index[d.apex]:
        t_a = d.t[i]; win = d[(d.t >= t_a - 1.0) & (d.t <= t_a + 1.0)]
        entry = d[(d.t >= t_a - 1.0) & (d.t <= t_a)].s_sm.max()
        if entry <= 1.0:
            continue
        slow = win[win.s_sm < 0.7 * entry]
        revs.append({"t_apex": d.t_rel[i], "rev_time": 0.1 * len(slow), "slow_from": slow.t_rel.min(), "slow_to": slow.t_rel.max(),
                     "rev_brake": d[(d.t >= t_a - 0.6) & (d.t <= t_a)].a_tan.min(),
                     "rev_reacc": d[(d.t >= t_a) & (d.t <= t_a + 0.6)].a_tan.max(), "px": d.px[i], "py": d.py[i]})
    m["revs"] = revs
    m["rev_time"] = np.median([r["rev_time"] for r in revs]) if revs else np.nan
    return m

# %% [markdown]
# ## 2. Choose the prospects to compare
# For each group × trait, the demonstration drill and the metric used to pick players. Each player's best attempt on that metric is taken (as in notebook 03), then the attempts closest to the group's **90th** and **10th** percentile are drawn. Extremes are avoided because the very top and bottom values are often tracking glitches.

# %%
DEMOS = [  # (group, trait, drill, metric used to pick, +1 higher is better / -1 lower is better)
    ("DL", "first_step", "PASS_RUSH_DRILL", "s05", 1),
    ("DL", "lateral", "FRONT_START_WAVE_DRILL_AND_LATERAL_REACTION", "rev_time", -1),
    ("DL", "burst", "RUN_AND_CLUB_DRILL", "acc_peak", 1),
    ("OL", "first_step", "PASS_RUSH_DROP", "s05", 1),
    ("OL", "lateral", "FIVE_YARD_WAVE_DRILL_SLIDE_AND_SHUFFLE", "rev_time", -1),
    ("OL", "burst", "OL_PULL_DRILL_FOLD_BLOCK_RIGHT", "acc_peak", 1),
]
att = spark.read.parquet(C.p("trait_attempts.parquet"))
pl = spark.read.parquet(C.p("line_players.parquet")).select("nfl_id", "nfl_position")
ti = spark.read.parquet(C.p("trait_indices.parquet")).cache()

def pick(group, trait, drill, metric, sign):
    """All in Spark: each player's best attempt (window), the group's percentiles, the attempts nearest to them,
    and each player's drill percentile (share of the group he beats on this metric)."""
    a = att.filter((F.col("line_group") == group) & (F.col("drill_name") == drill) & F.col(metric).isNotNull())
    if metric == "rev_time":
        a = a.filter(F.col("n_reversals") >= 1)
    better_first = F.col(metric).asc() if sign < 0 else F.col(metric).desc()
    worse_first = F.col(metric).desc() if sign < 0 else F.col(metric).asc()
    best = (a.withColumn("_r", F.row_number().over(Window.partitionBy("nfl_id").orderBy(better_first)))
             .filter("_r = 1").drop("_r")
             .withColumn("pct", F.bround(100 * (F.rank().over(Window.orderBy(worse_first)) - 1)
                                         / F.count(F.lit(1)).over(Window.partitionBy())).cast("int"))
             .join(pl, "nfl_id", "left")
             .join(ti.select("nfl_id", F.col(f"trk_{trait}").alias("idx")), "nfl_id", "left")
             .cache())
    hi_q, lo_q = (0.9, 0.1) if sign > 0 else (0.1, 0.9)
    q_hi, q_lo, med = best.agg(F.percentile(metric, [hi_q, lo_q, 0.5])).first()[0]
    nearest = lambda v: (best.orderBy(F.abs(F.col(metric) - F.lit(v)), F.col(metric), "nfl_id")
                             .limit(1).toPandas().iloc[0])          # one row → a Series for the drawing code
    return nearest(q_hi), nearest(q_lo), med

choices = {}
for group, trait, drill, metric, sign in DEMOS:
    top, bot, med = pick(group, trait, drill, metric, sign)
    choices[(group, trait)] = dict(drill=drill, metric=metric, sign=sign, top=top, bot=bot)
    print(f"{group} {trait:<10} {drill:<45} A: {top[metric]:.2f}  B: {bot[metric]:.2f}  (group median {med:.2f})")
ev_ids = [c[k].event_id for c in choices.values() for k in ("top", "bot")]
frames = (spark.read.parquet(C.p("combine_tracking.parquet")).filter(F.col("event_id").isin(ev_ids))
          .select("event_id", "ts", "x", "y", "s").toPandas())
frames["ts"] = pd.to_datetime(frames.ts)

# %% [markdown]
# ## 3. Combine trait animations

# %%
TRAIT_TITLE = {"first_step": "First-step quickness", "lateral": "Lateral agility", "burst": "Burst"}
TRAIT_HOW = {
    "first_step": "time to cover 1 and 2 yards from the first move; speed 0.3 s and 0.5 s in",
    "lateral": "at each direction reversal: time slowed below 70% of entry speed, braking in, re-acceleration out",
    "burst": "peak signed acceleration in the first 1.5 s; speed at 1.0 s; ground covered in the first 1.0 s",
}
TMAX = {"first_step": 1.6, "lateral": None, "burst": 2.5}   # seconds after first movement worth showing

def combine_animation(group, trait):
    c = choices[(group, trait)]
    runs = []
    for key, col, lab in [("top", COL_A, "A"), ("bot", COL_B, "B")]:
        row = c[key]
        d = onset_align(kinematics(frames[frames.event_id == row.event_id]))
        m = attempt_metrics(d[(d.t_rel >= -0.3)].reset_index(drop=True))   # metrics on the whole attempt
        if TMAX[trait] is not None:
            d = d[d.t_rel <= TMAX[trait]]
        d = d[(d.t_rel >= -0.3)].reset_index(drop=True)
        pos = row.nfl_position if isinstance(row.nfl_position, str) else "?"
        idx = row.idx if pd.notna(row.idx) else np.nan
        runs.append(dict(d=d, m=m, col=col, lab=f"Player {lab} ({pos})", idx=idx, pct=int(row.pct)))
    n = max(len(r["d"]) for r in runs)
    tmax = max(r["d"].t_rel.max() for r in runs)

    fig = plt.figure(figsize=(14, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.6, 0.9], height_ratios=[1, 1], wspace=0.32, hspace=0.38)
    axf = fig.add_subplot(gs[:, 0]); axs = fig.add_subplot(gs[0, 1]); axa = fig.add_subplot(gs[1, 1], sharex=axs)
    axt = fig.add_subplot(gs[:, 2]); axt.axis("off")
    fig.suptitle(f"{group} · {TRAIT_TITLE[trait]} — {c['drill'].replace('_', ' ').title()}", fontweight="bold", fontsize=13, x=0.02, ha="left")
    fig.text(0.02, 0.925, f"Measured as: {TRAIT_HOW[trait]}.   Drill percentile = this drill and metric only; "
             f"trait index = all of the player's {TRAIT_TITLE[trait].lower()} drills combined (z-score).", fontsize=9, color="#555")

    allx = np.concatenate([r["d"].px for r in runs]); ally = np.concatenate([r["d"].py for r in runs])
    if trait == "burst":                                  # keep the 1-second "ground covered" rings in view
        rmax = max(r["m"]["d10"] for r in runs); allx = np.r_[allx, -rmax, rmax]; ally = np.r_[ally, -rmax, rmax]
    pad = 1.0; span = max(allx.max() - allx.min(), ally.max() - ally.min(), 4.5) / 2 + pad
    cx, cy = (allx.max() + allx.min()) / 2, (ally.max() + ally.min()) / 2
    axf.set_xlim(cx - span, cx + span); axf.set_ylim(cy - span, cy + span); axf.set_aspect("equal")
    axf.set_title("Path (aligned: start at 0, first move points up)", fontsize=9); axf.set_xlabel("yards"); axf.set_ylabel("yards")
    axf.grid(color="#eee")
    if trait == "first_step":
        for r_ in (1, 2):
            axf.add_patch(Circle((0, 0), r_, fill=False, ls="--", lw=1, color="#888"))
            axf.text(r_ * 0.71, r_ * 0.71, f"{r_} yd", fontsize=8, color="#666")
    if trait == "burst":
        for r in runs:
            axf.add_patch(Circle((0, 0), r["m"]["d10"], fill=False, ls=":", lw=1.2, color=r["col"], alpha=.0))  # revealed at 1.0 s
    axs.set_xlim(-0.3, tmax); axs.set_ylim(0, max(r["d"].s_sm.max() for r in runs) * 1.15)
    axs.set_ylabel("speed (yd/s)"); axs.axvline(0, color="#bbb", lw=.8)
    amin = min(np.nanmin(r["d"].a_tan) for r in runs); amax = max(np.nanmax(r["d"].a_tan) for r in runs)
    axa.set_ylim(amin * 1.15 - 0.5, amax * 1.15 + 0.5); axa.axhline(0, color="#bbb", lw=.8); axa.axvline(0, color="#bbb", lw=.8)
    axa.set_ylabel("signed accel. (yd/s²)"); axa.set_xlabel("seconds since first movement")
    if trait == "first_step":
        for t_ in (0.3, 0.5):
            axs.axvline(t_, color="#999", ls=":", lw=1); axs.text(t_, axs.get_ylim()[1] * 0.96, f"{t_} s", fontsize=8, color="#666", ha="center")
    if trait == "burst":
        axa.axvspan(0, 1.5, color="#f2f2f2", zorder=0); axa.text(0.75, axa.get_ylim()[1] * 0.9, "peak window 0–1.5 s", fontsize=8, ha="center", color="#777")
        axs.axvline(1.0, color="#999", ls=":", lw=1)

    art = []
    for r in runs:
        ln_f, = axf.plot([], [], color=r["col"], lw=2.2); dot, = axf.plot([], [], "o", color=r["col"], ms=10, mec="white")
        ln_s, = axs.plot([], [], color=r["col"], lw=2); ln_a, = axa.plot([], [], color=r["col"], lw=2)
        art.append(dict(f=ln_f, dot=dot, s=ln_s, a=ln_a))
    marks = []  # (time, callable that draws a mark once)
    for k_run, r in enumerate(runs):
        m, col = r["m"], r["col"]
        if trait == "first_step":
            for name, rad in [("t_1yd", 1), ("t_2yd", 2)]:
                if not np.isnan(m[name]):
                    ytxt = axs.get_ylim()[1] * (0.62 - 0.14 * k_run - (0.07 if name == "t_2yd" else 0))
                    marks.append((m[name], lambda ax=axs, t=m[name], c=col, nm=name, yt=ytxt: (ax.axvline(t, color=c, lw=1, alpha=.6),
                                                                                       ax.text(t + 0.02, yt, nm.replace("t_", "") + f" {t:.1f}s", color=c, fontsize=8))))
            for name, t_ in [("s03", 0.3), ("s05", 0.5)]:
                if not np.isnan(m[name]):
                    marks.append((t_, lambda t=t_, v=m[name], c=col: axs.plot(t, v, "o", color=c, ms=7, mec="white")))
        if trait == "lateral":
            for rv in m["revs"]:
                marks.append((rv["t_apex"], lambda rv=rv, c=col: (axf.plot(rv["px"], rv["py"], marker="X", color=c, ms=12, mec="white"),
                                                                  axs.axvspan(rv["slow_from"], rv["slow_to"], color=c, alpha=.15),
                                                                  axs.text(rv["t_apex"], axs.get_ylim()[1] * .92, f"{rv['rev_time']:.1f}s", color=c, fontsize=8, ha="center"))))
        if trait == "burst":
            if not np.isnan(m["t_acc_peak"]):
                marks.append((m["t_acc_peak"], lambda m=m, c=col: axa.plot(m["t_acc_peak"], m["acc_peak"], "*", color=c, ms=15, mec="white")))
            marks.append((1.0, lambda m=m, c=col: axf.add_patch(Circle((0, 0), m["d10"], fill=False, ls=":", lw=1.5, color=c))))
    drawn = set()

    def metric_text(r, t_now):
        m = r["m"]; done = lambda t: (not np.isnan(t)) and t <= t_now
        if trait == "first_step":
            lines = [f"speed at 0.3 s: {m['s03']:.1f} yd/s" if t_now >= 0.3 else "speed at 0.3 s: …",
                     f"speed at 0.5 s: {m['s05']:.1f} yd/s" if t_now >= 0.5 else "speed at 0.5 s: …",
                     f"1 yd in {m['t_1yd']:.1f} s" if done(m['t_1yd']) else "1 yd: …",
                     f"2 yd in {m['t_2yd']:.1f} s" if done(m['t_2yd']) else "2 yd: …"]
        elif trait == "lateral":
            seen = [rv for rv in m["revs"] if rv["t_apex"] <= t_now]
            lines = [f"reversals so far: {len(seen)}"]
            if seen:
                lines += [f"time slowed: {np.median([v['rev_time'] for v in seen]):.1f} s",
                          f"braking in: {np.median([v['rev_brake'] for v in seen]):.1f} yd/s²",
                          f"re-accel out: {np.median([v['rev_reacc'] for v in seen]):.1f} yd/s²"]
        else:
            lines = [f"peak accel: {m['acc_peak']:.1f} yd/s²" if done(m['t_acc_peak']) else "peak accel: …",
                     f"speed at 1.0 s: {m['s10']:.1f} yd/s" if t_now >= 1.0 else "speed at 1.0 s: …",
                     f"ground in 1 s: {m['d10']:.1f} yd" if t_now >= 1.0 else "ground in 1 s: …"]
        return "\n".join(lines)
    txt = axt.text(0, 1, "", va="top", fontsize=10, family="monospace", transform=axt.transAxes)
    clock = axt.text(0, 0.02, "", fontsize=11, fontweight="bold", transform=axt.transAxes)

    def frame(i):
        t_now = -0.3 + i / FPS
        for r, a in zip(runs, art):
            d = r["d"][r["d"].t_rel <= t_now + 1e-9]
            a["f"].set_data(d.px, d.py); a["s"].set_data(d.t_rel, d.s_sm); a["a"].set_data(d.t_rel, d.a_tan)
            if len(d): a["dot"].set_data([d.px.iloc[-1]], [d.py.iloc[-1]])
        for k, (t_m, fn) in enumerate(marks):
            if k not in drawn and t_m <= t_now + 1e-9:
                fn(); drawn.add(k)
        blocks = []
        for r in runs:
            blocks.append(f"{r['lab']}\n{'─' * 26}\n{metric_text(r, t_now)}\n"
                          f"drill percentile: {r['pct']}th\ntrait index: {r['idx']:+.2f}\n")
        txt.set_text("\n".join(blocks))
        txt.set_color("#222")
        clock.set_text(f"t = {t_now:+.1f} s  ({SLOW:g}× speed)")
        return []
    # colour the player headers by drawing two small legend patches
    axt.plot([], [], color=COL_A, lw=6, label=runs[0]["lab"] + " · near top"); axt.plot([], [], color=COL_B, lw=6, label=runs[1]["lab"] + " · near bottom")
    axt.legend(loc="lower left", bbox_to_anchor=(0, 0.08), frameon=False, fontsize=9)
    n_frames = int(round((tmax + 0.3) * FPS)) + 6
    anim = animation.FuncAnimation(fig, frame, frames=n_frames, interval=1000 / (FPS * SLOW), blit=False)
    path = save(anim, f"combine_{group.lower()}_{trait}", fig)
    return path, runs

combine_runs = {}
for group, trait, *_ in DEMOS:
    drawn_before = time.time()
    path, runs = combine_animation(group, trait)
    combine_runs[(group, trait)] = runs
    print(f"   {time.time() - drawn_before:.0f} s")

# %% [markdown]
# ### Storyboard: the six Combine clips at their final frame (static summary for the report)

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
for ax, (group, trait, drill, metric, sign) in zip(axes.flat, DEMOS):
    runs = combine_runs[(group, trait)]
    for r in runs:
        d = r["d"][r["d"].t_rel >= -0.3]
        ax.plot(d.t_rel, d.s_sm, color=r["col"], lw=2, label=f"{r['lab']} · {r['pct']}th pct")
        m = r["m"]
        if trait == "first_step":
            ax.plot([0.3, 0.5], [m["s03"], m["s05"]], "o", color=r["col"], mec="white", ms=7)
        if trait == "lateral":
            for rv in m["revs"]:
                ax.axvspan(rv["slow_from"], rv["slow_to"], color=r["col"], alpha=.13)
        if trait == "burst":
            ax.axvline(m["t_acc_peak"], color=r["col"], ls=":", lw=1.2)
    ax.set_title(f"{group} · {TRAIT_TITLE[trait]}\n{drill.replace('_', ' ').title()[:42]}", fontsize=9)
    ax.set_xlabel("s since first movement"); ax.set_ylabel("speed (yd/s)"); ax.legend(frameon=False, fontsize=7.5)
fig.suptitle("Combine traits: near-top vs near-bottom prospect on each demonstration drill", fontweight="bold")
fig.tight_layout(); fig.savefig(ANIM_DIR / "combine_storyboard.png", bbox_inches="tight", dpi=130); plt.show()

# %% [markdown]
# ## 4. In-game: DL pass-rush snaps
# A high-burst and a low-burst DL by Combine tracking index. For each, the regular-season third-down rush snap whose speed at 1.0 s is closest to his own median: a typical rep, not a highlight. The play's own NGS fields supply the get-off time and whether it ended in pressure.

# %%
plays = spark.read.parquet(C.p("line_player_play.parquet")).filter(F.col("season_type") == "REG")
ing = spark.read.parquet(C.p("ingame_snaps.parquet"))
rush = (plays.filter((F.col("line_group") == "DL") & F.col("player_get_off").isNotNull() & F.col("pass_result").isNotNull())
        .join(ing.select("game_id", "play_id", "nfl_id", "g_s05", "g_s10", "g_t_1yd"), ["game_id", "play_id", "nfl_id"])
        .withColumn("pressure", ((F.col("time_to_pressure").isNotNull()) | F.col("time_to_qb_hurry").isNotNull() |
                                 (F.coalesce(F.col("sack"), F.lit(0)) > 0)).cast("int"))
        .select("game_id", "play_id", "nfl_id", "down", "yards_to_go", "play_direction", "lined_up_position", "player_get_off",
                "pressure", "time_to_pressure", "g_s05", "g_s10", "g_t_1yd").cache())
n_ok = rush.groupBy("nfl_id").count().filter(F.col("count") >= 100).select("nfl_id")
dl_ti = (ti.filter((F.col("line_group") == "DL") & F.col("trk_burst").isNotNull()).join(n_ok, "nfl_id").join(pl, "nfl_id", "left")
           .select("nfl_id", "nfl_position", "trk_burst",
                   F.row_number().over(Window.orderBy(F.col("trk_burst").desc(), "nfl_id")).alias("from_top"),
                   F.row_number().over(Window.orderBy(F.col("trk_burst").asc(), "nfl_id")).alias("from_bottom")))
pick_dl = {r.nfl_id: r for r in dl_ti.filter((F.col("from_top") == 3) | (F.col("from_bottom") == 3)).collect()}
hi_id = next(k for k, r in pick_dl.items() if r.from_top == 3)      # near the top (avoid the single extreme)
lo_id = next(k for k, r in pick_dl.items() if r.from_bottom == 3)
def typical_snap(pid):
    """Spark: the player's third-down rush snap (any down if none) whose speed at 1.0 s is nearest his own median."""
    r = rush.filter((F.col("nfl_id") == pid) & F.col("g_s10").isNotNull())
    r3 = r.filter(F.col("down") == 3)
    r = r3 if r3.limit(1).count() else r
    med = r.agg(F.percentile("g_s10", 0.5)).first()[0]
    return r.orderBy(F.abs(F.col("g_s10") - F.lit(med)), "game_id", "play_id").limit(1).toPandas().iloc[0]
snaps = [typical_snap(hi_id), typical_snap(lo_id)]
keys = [(int(s.game_id), int(s.play_id), int(s.nfl_id)) for s in snaps]
gt = spark.read.parquet(C.p("line_game_tracking.parquet"))
cond = None
for g, p_, n_ in keys:
    c_ = (F.col("game_id") == g) & (F.col("play_id") == p_) & (F.col("nfl_id") == n_)
    cond = c_ if cond is None else (cond | c_)
gfr = gt.filter(cond).select("game_id", "play_id", "nfl_id", "ts", "x", "y", "s", "event").toPandas()
gfr["ts"] = pd.to_datetime(gfr.ts)

def snap_align(d, play_direction):
    d = kinematics(d)
    i0 = d.index[d.event == "ball_snap"][0]
    d["t_rel"] = (d.t - d.t[i0]).round(2)
    sign = -1 if play_direction == "right" else 1     # defense attacks toward decreasing x when the offense moves right
    d["px"] = (d.y - d.y[i0]); d["py"] = sign * (d.x - d.x[i0])   # py > 0 = upfield (toward the QB)
    return d[(d.t_rel >= -0.5) & (d.t_rel <= 2.5)].reset_index(drop=True)

runs = []
for s, col, lab in [(snaps[0], COL_A, "A"), (snaps[1], COL_B, "B")]:
    d = snap_align(gfr[(gfr.game_id == s.game_id) & (gfr.play_id == s.play_id) & (gfr.nfl_id == s.nfl_id)], s.play_direction)
    pos = pick_dl[s.nfl_id].nfl_position or "?"
    idx = pick_dl[s.nfl_id].trk_burst
    runs.append(dict(d=d, s=s, col=col, lab=f"Player {lab} ({pos}, {s.lined_up_position.lower().replace('_line', '')})", idx=idx))

fig = plt.figure(figsize=(14, 6.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.5, 0.95], wspace=0.3)
axf = fig.add_subplot(gs[0]); axs = fig.add_subplot(gs[1]); axt = fig.add_subplot(gs[2]); axt.axis("off")
fig.suptitle("DL pass rush in an NFL game: does Combine burst show up after the snap?", fontweight="bold", fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.925, "Snap-relative movement of one typical 3rd-down rep each. Get-off and pressure come from the play's NGS record.", fontsize=9.5, color="#555")
allx = np.concatenate([r["d"].px for r in runs]); ally = np.concatenate([r["d"].py for r in runs])
span = max(allx.max() - allx.min(), ally.max() - ally.min(), 4) / 2 + 1
axf.set_xlim((allx.max() + allx.min()) / 2 - span, (allx.max() + allx.min()) / 2 + span); axf.set_ylim(-1.5, -1.5 + 2 * span)
axf.set_aspect("equal"); axf.grid(color="#eee"); axf.axhline(0.5, color="#555", lw=1.5, ls="--")
axf.text(axf.get_xlim()[0] + 0.2, 0.65, "≈ line of scrimmage", fontsize=8, color="#555")
axf.set_title("Movement from the snap (up = toward the QB)", fontsize=9); axf.set_xlabel("yards"); axf.set_ylabel("yards upfield")
axs.set_xlim(-0.5, 2.5); axs.set_ylim(0, max(r["d"].s_sm.max() for r in runs) * 1.2); axs.axvline(0, color="#888", lw=1)
axs.text(0.02, axs.get_ylim()[1] * 0.95, "snap", fontsize=8, color="#666")
for t_ in (0.5, 1.0):
    axs.axvline(t_, color="#999", ls=":", lw=1)
axs.set_xlabel("seconds after the snap"); axs.set_ylabel("speed (yd/s)")
art = []
for r in runs:
    a = dict(f=axf.plot([], [], color=r["col"], lw=2.2)[0], dot=axf.plot([], [], "o", color=r["col"], ms=11, mec="white")[0],
             s=axs.plot([], [], color=r["col"], lw=2.2)[0])
    art.append(a)
txt = axt.text(0, 1, "", va="top", fontsize=10, family="monospace", transform=axt.transAxes)
clock = axt.text(0, 0.02, "", fontsize=11, fontweight="bold", transform=axt.transAxes)
marked = set()
def frame(i):
    t_now = -0.5 + i / FPS
    blocks = []
    for k, (r, a) in enumerate(zip(runs, art)):
        d = r["d"][r["d"].t_rel <= t_now + 1e-9]; s = r["s"]
        a["f"].set_data(d.px, d.py); a["s"].set_data(d.t_rel, d.s_sm)
        if len(d): a["dot"].set_data([d.px.iloc[-1]], [d.py.iloc[-1]])
        if t_now >= s.player_get_off and (k, "go") not in marked:
            axs.axvline(s.player_get_off, color=r["col"], lw=1.2); marked.add((k, "go"))
            axs.text(s.player_get_off, axs.get_ylim()[1] * (0.85 - 0.08 * k), f" get-off {s.player_get_off:.2f}s", color=r["col"], fontsize=8)
        out = ""
        if t_now >= 2.4:
            out = (f"PRESSURE at {s.time_to_pressure:.1f} s" if s.pressure and not np.isnan(s.time_to_pressure or np.nan)
                   else ("PRESSURE" if s.pressure else "no pressure"))
        blocks.append(f"{r['lab']}\nCombine burst index {r['idx']:+.2f}\n{'─' * 28}\n"
                      f"{'get-off: %.2f s' % s.player_get_off if t_now >= s.player_get_off else 'get-off: …'}\n"
                      f"{'speed at 0.5 s: %.1f yd/s' % s.g_s05 if t_now >= 0.5 else 'speed at 0.5 s: …'}\n"
                      f"{'speed at 1.0 s: %.1f yd/s' % s.g_s10 if t_now >= 1.0 else 'speed at 1.0 s: …'}\n"
                      f"{int(s.down)} & {int(s.yards_to_go)}   {out}\n")
    txt.set_text("\n".join(blocks)); clock.set_text(f"t = {t_now:+.1f} s from snap ({SLOW:g}× speed)")
    return []
axt.plot([], [], color=COL_A, lw=6, label="high Combine burst"); axt.plot([], [], color=COL_B, lw=6, label="low Combine burst")
axt.legend(loc="lower left", bbox_to_anchor=(0, 0.08), frameon=False, fontsize=9)
anim = animation.FuncAnimation(fig, frame, frames=31 + 6, interval=1000 / (FPS * SLOW))
save(anim, "game_dl_pass_rush", fig)

# %% [markdown]
# ## 5. In-game: OL run-blocking credit
# A run play with two rookie linemen tracked, both with an inferred side and at least one at the point of attack. Left: their actual movement, with the hash marks (the ball is always spotted between them, which is how left / right is inferred). Right: the five line slots, the gap the play-by-play recorded, and the credit each rookie receives.

# %%
olr = (ing.filter((F.col("line_group") == "OL") & (F.col("play_type") == "run") & F.col("slot").isNotNull())
          .join(plays.select("game_id", "play_id", "nfl_id", "play_description", "play_direction", "yards_gained",
                             "expected_points_added", "down", "yards_to_go"), ["game_id", "play_id", "nfl_id"]).cache())
per_run = olr.groupBy("game_id", "play_id").agg(F.count(F.lit(1)).alias("n"), F.countDistinct("slot").alias("n_slots"),
                                                F.max("credit").alias("cmax"), F.min("credit").alias("cmin"),
                                                F.first("yards_gained", ignorenulls=True).alias("yards"))
cands = per_run.filter((F.col("n") >= 2) & (F.col("n_slots") == F.col("n")) & (F.col("cmax") == 1.0) & (F.col("cmin") == 0))
if cands.limit(1).count() == 0:
    cands = per_run.filter((F.col("n") >= 2) & (F.col("cmax") >= 0.5))
ranked = cands.withColumn("rk", F.row_number().over(Window.orderBy(F.col("yards").desc(), "game_id", "play_id")))
quarter = ranked.count() // 4 + 1                                       # a decent, not extreme, run
gid, pid = ranked.filter(F.col("rk") == quarter).select("game_id", "play_id").first()
play = olr.filter((F.col("game_id") == gid) & (F.col("play_id") == pid)).toPandas()   # the 2–3 linemen to draw
pf = (gt.filter((F.col("game_id") == int(gid)) & (F.col("play_id") == int(pid)) & F.col("nfl_id").isin([int(v) for v in play.nfl_id]))
        .select("nfl_id", "ts", "x", "y", "s", "event").toPandas())
pf["ts"] = pd.to_datetime(pf.ts)
direction = play.play_direction.iloc[0]
tracks = {}
for _, r in play.iterrows():
    d = kinematics(pf[pf.nfl_id == r.nfl_id]); i0 = d.index[d.event == "ball_snap"][0]
    d["t_rel"] = (d.t - d.t[i0]).round(2); tracks[r.nfl_id] = d[(d.t_rel >= -0.5) & (d.t_rel <= 2.5)].reset_index(drop=True)
desc = play.play_description.iloc[0]
print("play:", desc[:120])

fig = plt.figure(figsize=(14, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.15)
axf = fig.add_subplot(gs[0]); axs = fig.add_subplot(gs[1]); axs.axis("off")
fig.suptitle("OL run blocking: who gets credit for this run?", fontweight="bold", fontsize=13, x=0.02, ha="left")
fig.text(0.02, 0.925, f"Play-by-play: “{desc[:110]}{'…' if len(desc) > 110 else ''}”", fontsize=9, color="#555")
allx = np.concatenate([t.x for t in tracks.values()]); ally = np.concatenate([t.y for t in tracks.values()])
cx = (allx.max() + allx.min()) / 2
axf.set_xlim(cx - 6, cx + 6); axf.set_ylim(min(ally.min(), C.HASH_LO) - 3, max(ally.max(), C.HASH_HI) + 3); axf.set_aspect("equal")
for h in (C.HASH_LO, C.HASH_HI):
    axf.axhline(h, color="#333", ls="--", lw=1); axf.text(axf.get_xlim()[0] + 0.2, h + 0.2, "hash", fontsize=8, color="#333")
axf.axhspan(C.HASH_LO, C.HASH_HI, color="#f3f3f3", zorder=0)
axf.text(cx, (C.HASH_LO + C.HASH_HI) / 2, "ball is spotted\nbetween the hashes", ha="center", va="center", fontsize=8, color="#888")
arrow_dx = 2.5 if direction == "right" else -2.5
axf.annotate("offense moving", xy=(cx + arrow_dx, axf.get_ylim()[1] - 1), xytext=(cx, axf.get_ylim()[1] - 1),
             arrowprops=dict(arrowstyle="->", color="#555"), fontsize=8, color="#555", va="center", ha="center")
axf.set_xlabel("x (yards, along the field)"); axf.set_ylabel("y (yards from sideline)"); axf.set_title("Rookie linemen on this play (10 Hz tracking)", fontsize=9)
cols = ["#2a6fdb", "#e07b39", "#2e9e6b", "#7b61c9"]
art = {}
for k, (nid, d) in enumerate(tracks.items()):
    r = play[play.nfl_id == nid].iloc[0]
    art[nid] = dict(f=axf.plot([], [], color=cols[k], lw=2)[0], dot=axf.plot([], [], "o", color=cols[k], ms=12, mec="white")[0],
                    lab=axf.text(0, 0, r.slot, fontsize=9, fontweight="bold", color=cols[k]), col=cols[k], slot=r.slot, credit=r.credit)
# right: schematic of the five slots, offense's view (left to right = LT ... RT)
slots = ["LT", "LG", "C", "RG", "RT"]; xs = np.arange(5) * 1.6
gap_x = {"left end": -0.95, "left tackle": 0.8, "left guard": 2.4, "up the middle": 3.2, "right guard": 4.0, "right tackle": 5.6, "right end": 7.35}
gap = play.run_gap.iloc[0]
axs.set_xlim(-1.8, 8.2); axs.set_ylim(-3.2, 4.2)
axs.text(3.2, 3.8, "Offensive line (offense's view)", ha="center", fontsize=10, fontweight="bold")
boxes = {}
for s_, x_ in zip(slots, xs):
    b = FancyBboxPatch((x_ - 0.6, 0), 1.2, 1.0, boxstyle="round,pad=0.05", fc="#f4f4f4", ec="#999")
    axs.add_patch(b); boxes[s_] = b; axs.text(x_, 0.5, s_, ha="center", va="center", fontsize=11, fontweight="bold")
gap_arrow = FancyArrowPatch((gap_x[gap], -2.6), (gap_x[gap], -0.25), arrowstyle="-|>", mutation_scale=22, color="#d23f6b", lw=2.5, alpha=0)
axs.add_patch(gap_arrow)
gap_txt = axs.text(gap_x[gap], -3.0, "", ha="center", fontsize=9, color="#d23f6b", fontweight="bold")
info = axs.text(-1.6, 3.1, "", va="top", fontsize=9.5, family="monospace")
def frame(i):
    t_now = -0.5 + i / FPS
    for nid, d in tracks.items():
        dd = d[d.t_rel <= t_now + 1e-9]; a = art[nid]
        a["f"].set_data(dd.x, dd.y)
        if len(dd):
            a["dot"].set_data([dd.x.iloc[-1]], [dd.y.iloc[-1]]); a["lab"].set_position((dd.x.iloc[-1] + 0.4, dd.y.iloc[-1] + 0.4))
    stage = 0 if t_now < 0 else (1 if t_now < 1.0 else 2)
    for nid, a in art.items():
        b = boxes.get(a["slot"])
        if b is not None:
            b.set_edgecolor(a["col"]); b.set_linewidth(3)
            if stage >= 2:
                b.set_facecolor({1.0: "#bfe3c8", 0.5: "#e6f2d9"}.get(a["credit"], "#f4f4f4"))
    if stage >= 1:
        gap_arrow.set_alpha(1); gap_txt.set_text(f"run went: {gap}")
    lines = ["1. At the snap: side from position vs hashes"]
    for nid, a in art.items():
        lines.append(f"   {a['slot']:<3} (rookie)")
    if stage >= 1:
        lines.append(f"2. Play-by-play gap: {gap}")
    if stage >= 2:
        r0 = play.iloc[0]
        lines.append("3. Credit:")
        for nid, a in art.items():
            lines.append(f"   {a['slot']:<3} → {a['credit']:g}" + ("  (point of attack)" if a["credit"] == 1 else ("  (adjacent)" if a["credit"] == 0.5 else "  (away)")))
        succ = "success" if r0.expected_points_added > 0 else "not a success"
        lines.append(f"4. Result: {int(r0.yards_gained)} yd, EPA {r0.expected_points_added:+.2f} → {succ}")
    info.set_text("\n".join(lines))
    return []
anim = animation.FuncAnimation(fig, frame, frames=31 + 10, interval=1000 / (FPS * SLOW))
save(anim, "game_ol_run_credit", fig)

# %% [markdown]
# ## 6. Files written

# %%
files = sorted(p.name for p in ANIM_DIR.iterdir())
pd.DataFrame({"file": files, "MB": [round((ANIM_DIR / f).stat().st_size / 1e6, 2) for f in files]})
