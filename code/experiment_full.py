"""Full Pythia sample, with independent test shards and a wider EFN reference."""
import argparse
import copy
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiment import ROOT, Tagger, bootstrap, metrics, moments, project

OUT = ROOT / "results/full"
DATA = ROOT / "data/full"
CACHE = DATA / "cache"
TRAIN = ["QG_jets"] + [f"QG_jets_{n}" for n in range(1, 16)]
VALID = ["QG_jets_18", "QG_jets_19"]
TEST = ["QG_jets_16", "QG_jets_17", "QG_jets_herwig_2", "QG_jets_herwig_3"]
MODELS = ("efn_kan", "efn_mlp_wide", "efn_mlp")
EPOCHS = 5
BATCH = 1024
SEED = 17


def source(name, raw):
    file = name + ".npz"
    return next(path for path in (DATA / file, ROOT / "data" / file, raw / file) if path.is_file())


def prepare(name, raw, manifest):
    path = CACHE / (name + ".pt")
    entry = manifest[name + ".npz"]
    if path.exists():
        return
    src = source(name, raw)
    with src.open("rb") as file:
        actual = hashlib.file_digest(file, "md5").hexdigest()
    if "md5:" + actual != entry["checksum"]:
        raise ValueError(f"Checksum mismatch: {src}")
    with np.load(src, allow_pickle=False) as data:
        inputs, labels = data["X"], data["y"]
    if len(labels) != 100_000 or not np.array_equal(np.bincount(labels.astype(int)), [50000, 50000]):
        raise ValueError(f"Invalid label counts: {src}")
    all_q, all_z, all_moments, all_counts = [], [], [], []
    for start in range(0, len(labels), 1024):
        pt, eta, phi = torch.from_numpy(inputs[start:start + 1024, :, :3].astype(np.float32)).unbind(-1)
        z = pt / pt.sum(1, keepdim=True)
        eta0 = (z * eta).sum(1, keepdim=True)
        phi0 = torch.atan2((z * phi.sin()).sum(1, keepdim=True),
                           (z * phi.cos()).sum(1, keepdim=True))
        delta = (phi - phi0 + math.pi).remainder(2 * math.pi) - math.pi
        xy = torch.stack((eta - eta0, delta), -1).div_(0.4)
        q = project(xy)
        real = z > 0
        all_counts.append(real.sum(1))
        all_q.append(q[real])
        all_z.append(z[real])
        all_moments.append(moments(q, z))
    counts = torch.cat(all_counts)
    packed = {"q": torch.cat(all_q), "z": torch.cat(all_z), "moments": torch.cat(all_moments),
              "offset": torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))),
              "y": torch.tensor(labels, dtype=torch.float32)}
    temp = path.with_suffix(".part")
    torch.save(packed, temp)
    temp.replace(path)
    print("PREP", name, len(labels), len(packed["z"]), flush=True)


class EnergyFlow(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name == "efn_kan":
            self.kan = Tagger(name)
        else:
            width, latent = (64, 32) if name.endswith("wide") else (24, 16)
            self.phi = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width),
                                     nn.SiLU(), nn.Linear(width, latent))
            self.norm = nn.BatchNorm1d(latent)
            self.head = nn.Sequential(nn.Linear(latent, width), nn.SiLU(), nn.Linear(width, width),
                                      nn.SiLU(), nn.Linear(width, 1))

    def forward(self, shard, indices):
        if self.name == "efn_kan":
            return self.kan(None, None, shard["moments"][indices])
        offset = shard["offset"]
        starts = offset[indices]
        counts = offset[indices + 1] - starts
        ends = counts.cumsum(0)
        particle_indices = (torch.arange(ends[-1]) +
                            torch.repeat_interleave(starts - (ends - counts), counts,
                                                    output_size=int(ends[-1])))
        transformed = self.phi(shard["q"][particle_indices]) * shard["z"][particle_indices, None]
        pooled = torch.segment_reduce(transformed, "sum", lengths=counts)
        return self.head(self.norm(pooled)).flatten()


def load(name):
    return torch.load(CACHE / (name + ".pt"), mmap=True, weights_only=True)


@torch.inference_mode()
def predict(model, name):
    model.eval()
    shard = load(name)
    batches = torch.arange(len(shard["y"])).split(BATCH)
    scores = torch.cat([model(shard, ids) for ids in batches]).numpy()
    return shard["y"].numpy().copy(), scores


def fit(name):
    torch.manual_seed(SEED)
    model = EnergyFlow(name)
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS, eta_min=.0002)
    checkpoint = OUT / f"{name}_last.pt"
    best = OUT / f"{name}_best.pt"
    history, best_auc, initial = [], -1., 0
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        history, best_auc, initial = state["history"], state["best_auc"], state["epoch"]
    for epoch in range(initial, EPOCHS):
        start = time.perf_counter()
        model.train()
        rng = torch.Generator().manual_seed(SEED + epoch)
        loss_sum, n = 0., 0
        for source_id in torch.randperm(len(TRAIN), generator=rng):
            shard = load(TRAIN[source_id])
            batches = torch.randperm(len(shard["y"]), generator=rng).split(BATCH)
            for ids in batches:
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(shard, ids), shard["y"][ids])
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach()) * len(ids)
                n += len(ids)
            del shard
        sched.step()
        validation = [metrics(*predict(model, shard))["auc"] for shard in VALID]
        auc = sum(validation) / len(validation)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best)
        row = {"epoch": epoch + 1, "jets": n, "loss": loss_sum / n, "validation_auc": auc,
               "validation_by_shard": validation, "seconds": time.perf_counter() - start}
        history.append(row)
        temp = checkpoint.with_suffix(".part")
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch + 1, "history": history,
                    "best_auc": best_auc}, temp)
        temp.replace(checkpoint)
        (OUT / f"{name}_history.json").write_text(json.dumps(history, indent=2))
        print("TRAIN", name, row, flush=True)
    model.load_state_dict(torch.load(best, weights_only=True))
    return model


def evaluate():
    models = {}
    for name in MODELS:
        models[name] = EnergyFlow(name)
        models[name].load_state_dict(torch.load(OUT / f"{name}_best.pt", weights_only=True))
    result = {"models": {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()},
              "tests": {}}
    rows = []
    for shard in TEST:
        predictions = {}
        for name, model in models.items():
            y, score = predict(model, shard)
            predictions[name] = score
            rows.append({"shard": shard, "model": name, **metrics(y, score)})
        np.savez_compressed(OUT / (shard + "_predictions.npz"), y=y, **predictions)
        kan = predictions["efn_kan"]
        comparisons = {}
        for name in MODELS[1:]:
            comparisons[name] = {"delta_auc": metrics(y, kan)["auc"] - metrics(y, predictions[name])["auc"],
                                 "ci95": bootstrap(y.astype(np.int64), kan, predictions[name])}
        result["tests"][shard] = comparisons
        print("TEST", shard, comparisons, flush=True)
    with (OUT / "metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "evaluate"])
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.stage == "prepare":
        CACHE.mkdir(exist_ok=True)
        entries = {e["key"]: e for e in json.loads((DATA / "manifest.json").read_text(encoding="utf-8-sig"))}
        for name in TRAIN + VALID + TEST:
            prepare(name, args.raw, entries)
    elif args.stage == "train":
        fit(args.model)
    else:
        for name in MODELS:
            state = torch.load(OUT / f"{name}_last.pt", weights_only=True)
            if state["epoch"] != EPOCHS:
                raise ValueError(f"Training incomplete for {name}")
        evaluate()


if __name__ == "__main__":
    main()
