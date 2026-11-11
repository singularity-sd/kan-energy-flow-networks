# KAN inside an Energy Flow Network

Study on simulated light-quark and gluon jets, from the Singularity forecast T3v2-2026-0013.

- `protocol.md`: comparison and threshold fixed before the new training (pilot).
- `experiment.py`: preparation, training, checkpoints, test, bootstrap and numerical checks (pilot).
- `budget_sensitivity.py`: the 16-epoch, one-seed sensitivity check.
- `protocol_full.md` and `experiment_full.py`: the full-scale verification.
- `make_figure.py`: draws `fig_results.png` from the results.
- `results/`: metrics, checkpoints, configuration and logs of the pilot; `results/full/`: the same for the full scale.
- `data/manifest.json` and `data/full/manifest.json`: official URLs and checksums of the data files.

## Pilot

Download the NPZ files listed in `data/manifest.json` into `data/`, or point to their folder:

    python experiment.py --raw PATH
    python budget_sensitivity.py --raw PATH

## Full-scale verification

The protocol is in `protocol_full.md`. The manifest `data/full/manifest.json` lists the 22 files and their Zenodo
checksums. Save the NPZ files under `data/full/` or pass `--raw PATH` to the preparation:

    python experiment_full.py prepare --raw PATH
    python experiment_full.py train --model efn_mlp_wide
    python experiment_full.py train --model efn_kan
    python experiment_full.py train --model efn_mlp
    python experiment_full.py evaluate
    python make_figure.py

The preparation builds a local cache, and every epoch saves a checkpoint from which `train` can resume. Metrics and
predictions are in `results/full/`.

## Fingerprints

`results/sha256_fingerprints.txt` and `results/full/sha256_fingerprints.txt` are the SHA-256 fingerprints taken when
the protocols were frozen, before the runs. They refer to the original files, which had Italian names and comments;
the copies here were later translated into English (names, comments and messages only), so their fingerprints differ.

Sources: Energy Flow Networks (arXiv:1810.05165), KAN (arXiv:2404.19756), Geometric Kolmogorov-Arnold Superposition
Theorem (arXiv:2502.16664), Zenodo 3164691 and 3066475.
