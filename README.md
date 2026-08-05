# Recovering the Cost of Shared Backbones in Multi-Task Time Series Classification

Reference implementation for a manuscript of the same title, currently under review.

## What this is

A shared 1D-CNN backbone with one classification head per dataset is trained jointly
across the 30 heterogeneous datasets of the UCR *Sensor* category, and the accuracy cost
of that sharing is measured directly against per-dataset specialists (HIVE-COTE 2.0,
locally-fit MiniRocket) and a shared-LSTM baseline. Two further conditions test whether a
cheap per-dataset adaptation — fine-tuning only the classification head, or the whole
model — recovers the gap, and whether that recovery is explained by the adapted features
specifically or simply by additional training iterations on a from-scratch model at the
same budget.

| Condition | CNN | LSTM |
| --- | --- | --- |
| Joint (shared backbone, no adaptation) | 0.395 | 0.405 |
| From scratch (no shared backbone, same budget) | 0.479 | 0.449 |
| Head-only fine-tune (backbone frozen) | 0.625 | 0.420 |
| Full fine-tune (backbone unfrozen) | 0.647 | 0.459 |
| Specialist (MiniRocket, locally fit, 30/30 datasets) | 0.839 | 0.839 |
| Specialist (HIVE-COTE 2.0, published reference, 20/30 datasets) | 0.879 | 0.879 |

Mean test accuracy across all 30 datasets, except the HIVE-COTE 2.0 row (only 20 of the 30
have a published reference). The locally-fit MiniRocket specialist covers all 30 and is
the fairer point of comparison for the full corpus. See `run_all.py --stage figures` for
the figures this table is drawn from.

## Data

*Sensor* category of the [UCR Time Series Classification Archive (2018)](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/):
30 univariate datasets, 2–39 classes, series lengths 24–1,639. The archive zip is
password-protected; the password is published in `BriefingDocument2018.pdf` on the UCR
2018 page. Set it as an environment variable, never hardcode it:

```bash
export UCR2018_ZIP_PASSWORD="<password from BriefingDocument2018.pdf>"
```

```bibtex
@misc{UCRArchive2018,
  title  = {The UCR Time Series Classification Archive},
  author = {Dau, Hoang Anh and Keogh, Eamonn and Kamgar, Kaveh and
            Yeh, Chin-Chia Michael and Zhu, Yan and Gharghabi, Shaghayegh
            and Ratanamahatana, Chotirat Ann and Hu, Bing and Begum, Nurjahan
            and Bagnall, Anthony and Mueen, Abdullah and Batista, Gustavo},
  year   = {2018},
  month  = {October},
  note   = {https://www.cs.ucr.edu/~eamonn/time_series_data_2018/}
}
```

## Authors

Ian Mateos González, Israel Juárez Jiménez, Jesús García-Ramírez, Daniel Sánchez-Ruiz,
Cecilia Reyes Peña, and Eric Ramos Aguilar — Unidad Profesional Interdisciplinaria de
Ingeniería Campus Tlaxcala, Instituto Politécnico Nacional.

## License

MIT — see `LICENSE`.

## How to run

Two modes, same stages:

**Colab** — open `colab.ipynb`, run all cells. It checks for a GPU, clones this repo,
installs from `requirements.txt`, prompts for the UCR password, and calls `run_all.py`.

**Local**:

```bash
pip install -r requirements.txt
python run_all.py --stage all
```

A GPU is strongly recommended for `train-cnn`/`train-lstm` (8 seeds each); everything
else is viable on CPU. Every stage detects its own output artifact and skips re-running
unless `--force` is passed; `train-cnn`/`train-lstm` additionally checkpoint per seed, so
an interrupted run resumes at the next untrained seed rather than starting over.

```bash
python run_all.py --stage all                                   # full pipeline
python run_all.py --stage train-cnn --seeds 7 21                 # one stage, a seed subset
python run_all.py --stage finetune-cnn                           # per-dataset fine-tune, from the joint checkpoint
python run_all.py --stage scratch-cnn,scratch-lstm                # from-scratch control (not in --stage all)
python run_all.py --stage figures
python run_all.py --stage all --dry-run                          # print the plan, run nothing
python run_all.py --stage all --artifacts-dir /content/drive/MyDrive/latam_run
```

`--stage all` runs: `download, eda, geometry, train-cnn, train-lstm, finetune-cnn,
finetune-lstm, random-backbone, baselines, baselines-local, correlation, report, figures`.
`scratch-cnn` and `scratch-lstm` are invoked separately since they roughly double in cost
with both learning-rate variants reported.
