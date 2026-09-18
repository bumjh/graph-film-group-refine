# Graph-FiLM Group Refine

This repository contains the Graph-FiLM group refinement experiments for retinal glaucoma classification.

The main model keeps an RGB multi-task baseline as the visual anchor, derives task-specific auxiliary latent features, propagates them through a fixed clinical graph, and converts superior/inferior/NVT group contexts into FiLM modulation parameters for final referable glaucoma prediction.

## Contents

- `graph_film_group_refine/train_rgs_graph_film_refine.py`: group-aware Graph-FiLM refinement.
- `graph_film_group_refine/train_rgs_graph_film_global_pool_refine.py`: global-pooling ablation.
- `graph_film_group_refine/seg_guided_clinical_gcn_film.py`: segmentation-guided anatomical-prior wrapper.
- `graph_film_group_refine/external_validate_graph_film.py`: zero-shot external validation utility.
- `scripts/`: PowerShell helpers for seed sweeps and external validation.

## Not Included

Large or private artifacts are intentionally excluded:

- datasets and derived CSV files
- checkpoints and model weights
- segmentation memmaps
- experiment spreadsheets
- local output folders

Place local data under `data/` or pass explicit paths through CLI arguments.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Train

```powershell
python -m graph_film_group_refine.train_rgs_graph_film_refine `
  --csv path\to\JustRAIGS_processed.csv `
  --image-dir path\to\images_or_cache `
  --cache-dir path\to\cache `
  --output-dir checkpoints\graph_film_refine `
  --epochs 15 `
  --batch-size 16
```

To initialize from an RGB baseline checkpoint:

```powershell
python -m graph_film_group_refine.train_rgs_graph_film_refine `
  --csv path\to\JustRAIGS_processed.csv `
  --image-dir path\to\images_or_cache `
  --cache-dir path\to\cache `
  --rgb-checkpoint path\to\rgb_baseline\best.pt `
  --freeze-rgb-baseline
```

## External Validation

```powershell
python -m graph_film_group_refine.external_validate_graph_film `
  --checkpoint checkpoints\graph_film_refine\best.pt `
  --datasets refuge,origa,g1020 `
  --refuge-root path\to\refuge `
  --origa-root path\to\origa_v2 `
  --threshold-modes sensitivity_0.95
```

## Notes

This code was extracted from research experiments and expects the same label column conventions used by the JustRAIGS training scripts. Review the CLI arguments in each module before running a new experiment.
