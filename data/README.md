# data/ (not in git)

The raw NFL Big Data Bowl 2027 files are **never committed**: they are licensed CC BY-NC 4.0 and are ~2.3 GB.

| Where | Path |
| --- | --- |
| Rivanna (shared, use this) | `/standard/ds7200-apt4c/DS5110_Fall26_Group3/data/` |
| Kaggle (original) | https://www.kaggle.com/competitions/nfl-big-data-bowl-2027/data (join the competition to download) |
| Local copy | put the CSVs here (`data/`) or in `../data` next to the repo |

Files: `players.csv`, `combine_results.csv`, `combine_tracking.csv`, `player_career_successes.csv`, `games.csv`, `player_play.csv`, `game_tracking_2023.csv`, `game_tracking_2024.csv`, `game_tracking_2025.csv`.

`config.py` finds them automatically (or set `BDB_DATA_DIR`). Cleaned Parquet tables from the pipeline go to `BDB_PROCESSED_DIR` (on Rivanna: `/standard/ds7200-apt4c/DS5110_Fall26_Group3/processed/`, shared by the team).
