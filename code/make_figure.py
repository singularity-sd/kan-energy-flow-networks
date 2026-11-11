"""Draws fig_results.png (one level up) from results/summary.json and results/full/summary.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
pilot = json.loads((ROOT / "results/summary.json").read_text())["comparisons"]
full = json.loads((ROOT / "results/full/summary.json").read_text())["tests"]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharex=False)
rows = [("QG_jets_18", "Pythia 18 (primary)"), ("QG_jets_19", "Pythia 19"), ("QG_jets_herwig_1", "Herwig 1")]
ax = axes[0]
for k, (shard, label) in enumerate(rows):
    for dy, model, colour, name in [(-0.12, "efn_kan", "#1f5f75", "KAN encoder and head"),
                                    (0.12, "efn_spline_mlp", "#b3542a", "spline encoder + MLP head")]:
        c = pilot[shard][model]
        ax.errorbar(c["delta_ensemble_auc"], k + dy, xerr=[[c["delta_ensemble_auc"] - c["ci95"][0]],
                    [c["ci95"][1] - c["delta_ensemble_auc"]]], fmt="o", color=colour, capsize=0,
                    label=name if k == 0 else None)
ax.set_yticks(range(len(rows)), [r[1] for r in rows]); ax.invert_yaxis()
ax.set_xlim(-0.002, 0.034)
ax.set_title("(a) pilot, 200k training jets: vs compact EFN", fontsize=10)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
rows = [("QG_jets_16", "Pythia 16 (primary)"), ("QG_jets_17", "Pythia 17"), ("QG_jets_herwig_2", "Herwig 2"),
        ("QG_jets_herwig_3", "Herwig 3")]
ax = axes[1]
for k, (shard, label) in enumerate(rows):
    c = full[shard]["efn_mlp_wide"]
    ax.errorbar(c["delta_auc"], k, xerr=[[c["delta_auc"] - c["ci95"][0]], [c["ci95"][1] - c["delta_auc"]]],
                fmt="o", color="#1f5f75", capsize=0)
ax.set_yticks(range(len(rows)), [r[1] for r in rows]); ax.invert_yaxis()
ax.set_title("(b) full scale, 1.6M jets: vs wide EFN", fontsize=10)
for ax in axes:
    ax.axvline(0, color="#444", lw=0.8); ax.axvline(0.003, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("AUC difference (paired 95% interval)")
fig.tight_layout()
fig.savefig(ROOT.parent / "fig_results.png", dpi=180)
