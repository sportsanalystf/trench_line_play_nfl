# Workflow: GitHub ⇄ Rivanna ⇄ laptop

GitHub is the single source of truth for **code and small results** (notebooks, `.py`, figures, tables). Data and large outputs never go to GitHub:
- **data** lives on Rivanna;
- **videos, play viewer and models** are regenerated on Rivanna and copied to laptops with a script.

```
            git push / pull                      git pull / push
  laptop  ◄────────────────►  GitHub (main)  ◄────────────────►  Rivanna clone
 (your Mac)                                                       /standard/ds7200-apt4c/DS5110_Fall26_Group3/ds5110-trench-play
     ▲                                                                   │ runs the pipeline (sbatch)
     └──────── scripts/pull_outputs_from_rivanna.sh (videos, viewer) ◄───┘
```

## One-time setup

### 1. GitHub (Faizan, once)
1. Create an **empty private** repo `ds5110-trench-play` on github.com (no README, no .gitignore).
2. Add Isaac, Jarmarr and Weston: repo → Settings → Collaborators.
3. From the repo folder on the Mac:
   ```bash
   bash scripts/first_push.sh https://github.com/<user>/ds5110-trench-play.git
   ```

### 2. Rivanna (each person, once)
GitHub needs a key or token to push from Rivanna. The SSH key is simplest:
```bash
ssh-keygen -t ed25519 -C "<computing-id>@virginia.edu"     # press Enter at every prompt
cat ~/.ssh/id_ed25519.pub                                  # copy → github.com → Settings → SSH and GPG keys → New SSH key
```
**The shared clone** (the one that runs the pipeline; Faizan sets it up once):
```bash
cd /standard/ds7200-apt4c/DS5110_Fall26_Group3
git clone git@github.com:<user>/ds5110-trench-play.git
cd ds5110-trench-play && git config core.sharedRepository group && chmod -R g+rwX .
mkdir -p ../processed && chmod g+rwX ../processed
module load spark miniforge && pip install --user -r requirements.txt
```
If the Parquet tables already exist from an earlier run, copy them into the shared folder once:
```bash
cp -r /scratch/xdy6sg/ds5110_line_play/processed/* /standard/ds7200-apt4c/DS5110_Fall26_Group3/processed/
```
**Your own clone** (to try changes without touching the shared one; optional): `cd ~ && git clone git@github.com:<user>/ds5110-trench-play.git`

### 3. Laptop (each person, once)
```bash
git clone https://github.com/<user>/ds5110-trench-play.git
```
Faizan's Mac copy is the existing `Final Project/trench_line_play` folder, already set up by `first_push.sh`.

## Every day

| I want to… | Do this |
| --- | --- |
| start working | `git pull` |
| save and share my work | `git add pipeline/<notebook> … && git commit -m "what I did" && git push` |
| run the pipeline | on Rivanna, in the shared clone: `git pull && sbatch slurm/run_pipeline.slurm` (or `… 03 04` for some steps) |
| share fresh pipeline results | in the shared clone after the job: `git add pipeline outputs/figures outputs/tables && git commit -m "pipeline run <date>" && git push` |
| get videos / play viewer on my laptop | `bash scripts/pull_outputs_from_rivanna.sh <computing-id>` |
| change a pipeline notebook | edit it, test it (Rivanna `sbatch slurm/run_pipeline.slurm <step>`), then commit; tell the team in the chat |

## Avoiding conflicts
- **Everyone edits the same `pipeline/` notebooks.** Say in the group chat which notebook you're working on.
- **Pull before you start and push soon after.** Small, frequent commits are easier to merge.
- **If two people need the same notebook at once, use a branch** (`git switch -c my-change`, then open a pull request on GitHub).
- **A new analysis goes into the step it belongs to.** Add it as a new section, or as a lettered step (`07b_…`) if it's large, and add it to the list in `slurm/run_pipeline.slurm`.
- **Executed notebooks change on every run.** Commit pipeline runs only from the shared Rivanna clone, so there is one place where results come from.
- **If `git pull` reports a conflict in an `.ipynb`,** keep the newer run: `git checkout --theirs pipeline/<nb>.ipynb && git add pipeline/<nb>.ipynb && git commit`.

## OneDrive note (Faizan's Mac)
The Mac repo sits inside OneDrive. That works, but OneDrive can briefly lock files while it syncs. If git reports `index.lock` or "file in use", wait a few seconds and repeat the command. To be safe, pause OneDrive syncing before a large `git pull`.
