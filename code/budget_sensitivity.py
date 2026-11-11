"""Budget sensitivity specified while the primary test was still training."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiment import ROOT, NAMES, prepare, train, predict, metrics

def main():
    torch.set_num_threads(3)
    out = ROOT / "results" / "budget16"
    out.mkdir(exist_ok=True)
    entries = {e["key"]: e for e in json.loads(
        (ROOT / "data/manifest.json").read_text(encoding="utf-8-sig"))}
    cache = ROOT / "data/cache"
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    raw = parser.parse_args().raw
    args = SimpleNamespace(epochs=16, batch=512)
    training = [prepare(raw, cache, n, entries) for n in NAMES[:2]]
    valid = prepare(raw, cache, NAMES[2], entries)
    models = {kind: train(kind, 17, training, valid, args, out)[0]
              for kind in ("efn_mlp", "efn_kan")}
    del training, valid
    report = {}
    for name in (NAMES[3], NAMES[5]):
        data = prepare(raw, cache, name, entries)
        scores = {kind: predict(model, data, args.batch) for kind, model in models.items()}
        y = data[-1].numpy()
        report[name] = {kind: metrics(y, s) for kind, s in scores.items()}
        np.savez_compressed(out / (name + "_predictions.npz"), y=y, **scores)
        del data
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(report, flush=True)

if __name__ == "__main__":
    main()
