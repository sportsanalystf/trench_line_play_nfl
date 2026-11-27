# outputs/

Written by the pipeline notebooks (`pipeline/`). Re-run a step to regenerate its outputs.

| Folder | In git? | What |
| --- | --- | --- |
| `figures/` | yes | every figure, prefixed by the step that made it (`02_…png` = notebook 02) |
| `tables/` | yes | every result table (CSV), same prefix rule |
| `feature_sets.json` | yes | feature lists used by the models (notebook 05) |
| `animations/*.png` | yes | still frames of the animations |
| `animations/*.mp4` | no (large) | videos from notebooks 08 and 11 |
| `play_viewer/*.html` | no (large) | interactive full-field play viewer (notebook 11) |
| `models/` | no (large) | fitted Spark ML pipelines (notebook 06) |

The large files are regenerated on Rivanna; copy them to your laptop with `scripts/pull_outputs_from_rivanna.sh`.
