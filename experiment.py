"""Reproduces the study by running the original scripts in code/ in their historical order.

1. code/experiment.py                      - pilot: three EFN variants, three seeds, 200,000 training jets
2. code/budget_sensitivity.py               - 16-epoch, one-seed sensitivity check (KAN vs perceptron EFN)
3. code/experiment_full.py prepare     - full scale: 2 million Pythia jets split by file, Herwig transfer set
4. code/experiment_full.py train       - wide EFN, KAN EFN and compact EFN (resumable per epoch)
5. code/experiment_full.py evaluate    - test AUC, paired bootstrap, mistag rates
6. code/make_figure.py                      - draws fig_results.png
Protocols fixed before training: code/protocol.md (pilot) and code/protocol_full.md (full scale).

Data: the Pythia and Herwig NPZ files listed with checksums in code/data/manifest.json and
code/data/full/manifest.json (Zenodo 3164691 and 3066475), saved under code/data/ and code/data/full/.
Requires Python 3.12 and the packages in code/requirements.txt (PyTorch on CPU is enough; full scale takes hours).
"""

import os
import runpy
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"

steps = [("experiment.py", []), ("budget_sensitivity.py", []), ("experiment_full.py", ["prepare"])]
steps += [("experiment_full.py", ["train", "--model", m]) for m in ("efn_mlp_wide", "efn_kan", "efn_mlp")]
steps += [("experiment_full.py", ["evaluate"]), ("make_figure.py", [])]

for step, argv in steps:
    script = CODE / step
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script)] + argv
    print(f"== {step} {' '.join(argv)}", flush=True)
    runpy.run_path(str(script), run_name="__main__")
