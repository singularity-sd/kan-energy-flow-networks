# Kolmogorov–Arnold layers inside an Energy Flow Network for quark/gluon jet tagging

From a v2 forecast: Kolmogorov–Arnold layers beat a jet tagger six times larger on every test file.

**Paper:** [paper.pdf](paper.pdf) · **Page:** [singularitysd.com/research/quark-gluon.html](https://singularitysd.com/research/quark-gluon.html) · Preprint, not yet peer reviewed.

## Abstract

Energy Flow Networks (EFNs) classify jets by summing a learned per-particle function weighted by momentum fraction. We replace the multilayer perceptrons of a compact EFN with Kolmogorov–Arnold (spline) layers, keeping inputs, summation and parameter count fixed, and test it on public simulated light-quark and gluon jets. In a pilot with 200,000 training jets and three seeds, the KAN version raised the test AUC from 0.8306 to 0.8584 (paired 95% interval for the difference $+0.0266$ to $+0.0293$), with gains on a second Pythia file and on Herwig jets. A full-scale check, fixed in advance, trained on 1.6 million jets and compared the KAN EFN (2,239 parameters) with a wider standard EFN of 13,025 parameters: the KAN version was still better on the new primary file, AUC 0.8665 against 0.8573 (difference $+0.0091$, interval $+0.0081$ to $+0.0100$), and on all three other test files, including two from a different generator. At 80% quark efficiency the gluon mistag rate fell from 24.9% to 22.6%, with a network six times smaller.

## How it was made

This study started from the Singularity forecast **T3v2-2026-0013**: a method predicted to enter a new field. It was then tested once, with the plan fixed before looking at the results.

## What is in this repository

| file | what it is |
|---|---|
| `code/` | full code, protocol and intermediate results |
| `experiment.py` | reproduces the study end to end |
| `fig_results.png` | figure |
| `main.tex` | LaTeX source of the paper |
| `paper.pdf` | the paper |
| `results.json` | every number reported in the paper |

## Reproduce

```bash
pip install -r code/requirements.txt
python experiment.py
```

It rewrites the results and figures.

## Cite

```bibtex
@misc{singularity_kan_energy_flow_networks_2026,
  title  = {Kolmogorov–Arnold layers inside an Energy Flow Network for quark/gluon jet tagging},
  author = {Singularity},
  year   = {2026},
  url    = {https://github.com/singularity-sd/kan-energy-flow-networks}
}
```

## License

Code: MIT (`LICENSE`). Paper, text and figures: CC BY 4.0 (`LICENSE-paper.md`). Third-party data keep their own licence where stated.

---

Part of [Singularity](https://singularitysd.com): forecasting where science goes next, then testing it.
