# pipeline/

The shared analysis: run the notebooks in order, 01 → 11 (see the table in the main README).

- `*.ipynb`: the notebooks, with outputs from the latest full run on Rivanna.
- `src/*.py`: the same notebooks as plain Python (jupytext "percent" format), easier to read and review on GitHub.

Each notebook imports `../config.py` for every path and its Spark settings. Run them from this folder (JupyterLab does this automatically), or with `sbatch slurm/run_pipeline.slurm` from the repo root.

Rule for new code: compute in Spark (use the helpers in `config.py` for descriptive statistics and correlations), then `toPandas()` only the finished result to plot or save it.
