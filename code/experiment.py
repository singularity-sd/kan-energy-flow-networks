import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn

ROOT = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
KINDS = ("efn_mlp", "efn_kan", "efn_spline_mlp")
NAMES = ["QG_jets", "QG_jets_2", "QG_jets_1", "QG_jets_18", "QG_jets_19", "QG_jets_herwig_1"]


def basis(x, grid=8):
    # Uniform cubic cardinal B-splines with ghost knots at the boundary.
    u = (x.unsqueeze(-1) + 1) * (grid / 2) - torch.arange(-1, grid + 2)
    a = u.abs()
    return torch.where(a < 1, (4 - 6 * a.square() + 3 * a.pow(3)) / 6,
                       (2 - a).clamp_min(0).pow(3) / 6)


def project(xy):
    x, y = xy.unbind(-1)
    r = (x.square() + y.square()).sqrt()
    return torch.tanh(2.5 * torch.stack((x, y, (x + y) / math.sqrt(2),
                                       (x - y) / math.sqrt(2), r), -1))


def moments(q, z):
    features = torch.cat((basis(q), nn.functional.silu(q).unsqueeze(-1)), -1)
    return (features * z[..., None, None]).sum(1).flatten(1)


def prepare(raw, cache, name, entries):
    file = name + ".npz"
    meta = entries[file]
    src = next(p for p in (ROOT / "data" / file, raw / file) if p.is_file())
    with src.open("rb") as stream:
        actual = hashlib.file_digest(stream, "md5").hexdigest()
    if "md5:" + actual != meta["checksum"]:
        raise ValueError(f"Checksum mismatch for {src}")
    dest = cache / (name + "_" + actual[:10] + "_v1.pt")
    if dest.exists():
        return torch.load(dest, weights_only=True)
    with np.load(src, allow_pickle=False) as data:
        array, labels = data["X"], data["y"]
    parts, weights, summaries = [], [], []
    for start in range(0, len(labels), 1024):
        p = torch.from_numpy(array[start:start + 1024, :, :3].astype(np.float32))
        pt, eta, phi = p.unbind(-1)
        z = pt / pt.sum(1, keepdim=True)
        eta0 = (z * eta).sum(1, keepdim=True)
        phi0 = torch.atan2((z * phi.sin()).sum(1, keepdim=True),
                           (z * phi.cos()).sum(1, keepdim=True))
        dphi = (phi - phi0 + math.pi).remainder(2 * math.pi) - math.pi
        xy = torch.stack((eta - eta0, dphi), -1).div_(0.4)
        xy[z == 0] = 0
        parts.append(xy)
        weights.append(z)
        summaries.append(moments(project(xy), z))
    pack = tuple(torch.cat(x) for x in (parts, weights, summaries)) + (
        torch.tensor(labels, dtype=torch.float32),)
    if len(labels) != 100_000 or not np.array_equal(np.bincount(labels.astype(int)), [50000, 50000]):
        raise ValueError(f"Unexpected labels in {src}")
    torch.save(pack, dest)
    return pack


class Spline(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.linear = nn.Linear(inputs * 12, outputs)
        with torch.no_grad():
            self.linear.weight.mul_(0.25)

    def forward(self, x):
        features = torch.cat((basis(torch.tanh(x)), nn.functional.silu(x).unsqueeze(-1)), -1)
        return self.linear(features.flatten(-2))


class Tagger(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.phi = nn.Sequential(nn.Linear(5, 24), nn.SiLU(), nn.Linear(24, 24), nn.SiLU(),
                                 nn.Linear(24, 16)) if kind == "efn_mlp" else nn.Linear(60, 16)
        self.norm = nn.BatchNorm1d(16)
        self.head = (nn.Sequential(Spline(16, 6), Spline(6, 1)) if kind == "efn_kan" else
                     nn.Sequential(nn.Linear(16, 24), nn.SiLU(), nn.Linear(24, 24),
                                   nn.SiLU(), nn.Linear(24, 1)))

    def forward(self, x, z, summary):
        if self.kind == "efn_mlp":
            pooled = (self.phi(project(x)) * z.unsqueeze(-1)).sum(1)
        else:
            pooled = self.phi(summary)
        return self.head(self.norm(pooled)).flatten()


def metrics(y, scores):
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    return {"auc": float(roc_auc_score(y, scores)),
            **{f"fpr_at_{int(t*100)}": float(fpr[np.searchsorted(tpr, t)]) for t in (.5, .8)}}


@torch.inference_mode()
def predict(model, data, batch):
    model.eval()
    return torch.cat([model(*(v[i:i + batch] for v in data[:3]))
                      for i in range(0, len(data[-1]), batch)]).numpy()


def indexed(data, idx):
    return tuple(v[idx] for v in data)


def train(kind, seed, sources, val, args, out):
    torch.manual_seed(seed)
    model = Tagger(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    file = out / f"{kind}_{seed}.pt"
    history_path = out / f"{kind}_{seed}_history.json"
    if file.exists() and history_path.exists():
        model.load_state_dict(torch.load(file, weights_only=True))
        return model, json.loads(history_path.read_text())
    best_auc, best = -1, None
    history = []
    for epoch in range(args.epochs):
        start = time.perf_counter()
        model.train()
        orders = [torch.randperm(len(d[-1])).split(args.batch) for d in sources]
        for step in range(max(map(len, orders))):
            for d, order in zip(sources, orders):
                if step >= len(order):
                    continue
                x, z, summary, y = indexed(d, order[step])
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.binary_cross_entropy_with_logits(model(x, z, summary), y)
                loss.backward()
                opt.step()
        schedule.step()
        auc = float(roc_auc_score(val[-1].numpy(), predict(model, val, args.batch)))
        duration = time.perf_counter() - start
        history.append({"epoch": epoch + 1, "auc": auc, "seconds": duration})
        if auc > best_auc:
            best_auc = auc
            best = {k: v.clone() for k, v in model.state_dict().items()}
        print(kind, seed, epoch + 1, round(auc, 5), round(duration, 1), flush=True)
    model.load_state_dict(best)
    torch.save(best, file)
    history_path.write_text(json.dumps(history, indent=2))
    return model, history


def bootstrap(y, first, second, repeats=300):
    rng = np.random.default_rng(20261107)
    ordering = [np.argsort(scores, kind="stable") for scores in (first, second)]
    # Tie-aware weighted Mann-Whitney statistic, keeping the same resample for both models.
    blocks = [np.r_[0, np.flatnonzero(np.diff(scores[o])) + 1]
              for scores, o in zip((first, second), ordering)]
    deltas = []
    for _ in range(repeats):
        w = np.bincount(rng.integers(len(y), size=len(y)), minlength=len(y))
        aucs = []
        for order, starts in zip(ordering, blocks):
            pos = np.add.reduceat(w[order] * y[order], starts)
            neg = np.add.reduceat(w[order] * (1 - y[order]), starts)
            aucs.append((pos @ (np.cumsum(neg) - .5 * neg)) / (pos.sum() * neg.sum()))
        deltas.append(aucs[0] - aucs[1])
    return np.quantile(deltas, [.025, .975]).tolist()


@torch.inference_mode()
def invariance(model, data):
    model.eval()
    x, z = (v[:64] for v in data[:2])
    original = model(x, z, moments(project(x), z))
    perm = torch.arange(x.shape[1] - 1, -1, -1)
    xp, zp = x[:, perm], z[:, perm]
    xs, zs = torch.cat((x, x[:, :1]), 1), torch.cat((z, .6 * z[:, :1]), 1)
    zs[:, 0] *= .4
    xe = torch.cat((x, torch.full_like(x[:, :1], .35)), 1)
    ze = torch.cat(((1 - 1e-7) * z, torch.full_like(z[:, :1], 1e-7)), 1)
    return {label: float((model(u, v, moments(project(u), v)) - original).abs().max())
            for label, u, v in (("permutation", xp, zp), ("collinear", xs, zs), ("soft", xe, ze))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "data")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    settings = {"epochs": args.epochs, "batch": args.batch, "seeds": SEEDS}
    config = out / "config.json"
    canonical = json.dumps(settings, sort_keys=True)
    if config.exists() and config.read_text() != canonical:
        raise ValueError("Existing results have a different configuration")
    config.write_text(canonical)
    cache = ROOT / "data" / "cache"
    cache.mkdir(exist_ok=True)
    entries = {e["key"]: e for e in json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8-sig"))}
    sources = [prepare(args.raw, cache, n, entries) for n in NAMES[:2]]
    val = prepare(args.raw, cache, NAMES[2], entries)
    models, checks, parameters = {}, {}, {}
    for seed in SEEDS:
        for kind in KINDS:
            model, _ = train(kind, seed, sources, val, args, out)
            models[kind, seed] = model
            checks[f"{kind}_{seed}"] = invariance(model, val)
            parameters[kind] = sum(p.numel() for p in model.parameters())
    del sources, val
    rows, comparisons = [], {}
    for name in NAMES[3:]:
        data = prepare(args.raw, cache, name, entries)
        y = data[-1].numpy().astype(np.int64)
        scores = {f"{kind}_{seed}": predict(models[kind, seed], data, args.batch)
                  for kind in KINDS for seed in SEEDS}
        for kind in KINDS:
            scores[f"{kind}_ensemble"] = np.mean([scores[f"{kind}_{s}"] for s in SEEDS], axis=0)
        for key, predictions in scores.items():
            kind, seed = key.rsplit("_", 1)
            rows.append({"shard": name, "model": kind, "seed": seed, **metrics(y, predictions)})
        np.savez_compressed(out / f"{name}_predictions.npz", y=y, **scores)
        ref = scores["efn_mlp_ensemble"]
        comparisons[name] = {}
        for kind in KINDS[1:]:
            trial = scores[f"{kind}_ensemble"]
            delta = roc_auc_score(y, trial) - roc_auc_score(y, ref)
            mean = np.mean([roc_auc_score(y, scores[f"{kind}_{s}"]) -
                            roc_auc_score(y, scores[f"efn_mlp_{s}"]) for s in SEEDS])
            comparisons[name][kind] = {"delta_ensemble_auc": float(delta),
                                       "delta_mean_seed_auc": float(mean),
                                       "ci95": bootstrap(y, trial, ref)}
        print(name, comparisons[name], flush=True)
        del data
    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps({"parameters": parameters, "checks": checks,
                                                "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
