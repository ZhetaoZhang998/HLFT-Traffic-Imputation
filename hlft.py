"""Hyperbolic latent factorization of traffic tensors, curvature c=1.

Adapted from the supplied HLFT source. Validation selects the checkpoint;
test values are accessed only for final evaluation. See docs/METHOD.md.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numba import njit

CITIES = ["Berlin", "Guangzhou", "NewYork", "Seattle"]
SPLITS = {"train": "train70", "valid": "valid10", "test": "test20"}


# SECTION: Sparse observations and split checks
def read_observations(path: Path, keep_zero: bool = False):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        text = handle.read()
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        numbers = np.fromstring(text.replace("::", " "), sep=" ")
    if not numbers.size or numbers.size % 4:
        raise ValueError(f"Expected sensor::interval::day::speed records in {path}")
    rows = numbers.reshape(-1, 4)
    ids = rows[:, :3]
    if not np.isfinite(ids).all() or (ids < 0).any() or not np.equal(ids, np.floor(ids)).all():
        raise ValueError(f"Invalid tensor coordinates in {path}")
    finite = np.isfinite(rows[:, 3])
    negative = finite & (rows[:, 3] < 0)
    if negative.any():
        raise ValueError(f"Negative traffic speeds in {path}")
    zero = rows[:, 3] == 0
    mask = finite & (keep_zero | ~zero)
    metadata = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(rows), "used_rows": int(mask.sum()),
        "zero_rows": int(zero.sum()), "nonfinite_rows": int((~finite).sum()),
        "zero_policy": "keep" if keep_zero else "drop_as_in_original_source"}
    rows = np.ascontiguousarray(rows[mask], dtype=np.float64)
    if len(rows) == 0:
        raise ValueError(f"No usable observations in {path}")
    return rows, metadata


def check_splits(arrays: dict[str, np.ndarray]):
    coordinates = {}
    for split, rows in arrays.items():
        coords = np.ascontiguousarray(rows[:, :3], dtype=np.int64)
        keys = coords.view(np.dtype((np.void, coords.dtype.itemsize * 3))).ravel()
        if len(np.unique(keys)) != len(keys):
            raise ValueError(f"Duplicate tensor coordinates within {split}")
        coordinates[split] = keys
    names = list(arrays)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if len(np.intersect1d(coordinates[a], coordinates[b])):
                raise ValueError(f"Tensor entries overlap between {a} and {b}")


def load_city(data_dir: Path, city: str, keep_zero: bool = False):
    arrays, metadata = {}, {}
    for split, suffix in SPLITS.items():
        path = data_dir / city / f"{city}_{suffix}.txt.gz"
        if not path.exists():
            path = path.with_suffix("")
        arrays[split], metadata[split] = read_observations(path, keep_zero)
    check_splits(arrays)
    # This is transductive entry imputation: axis identities from all three
    # splits define the tensor, but held-out speed values do not train factors.
    identities = [np.unique(np.concatenate([a[:, axis] for a in arrays.values()])).astype(np.int64)
                  for axis in range(3)]
    for split in arrays:
        arrays[split] = arrays[split].copy()
        for axis, ids in enumerate(identities):
            arrays[split][:, axis] = np.searchsorted(ids, arrays[split][:, axis])
    metadata["shape"] = [len(ids) for ids in identities]
    metadata["validation_test_axis_ids_not_seen_in_train"] = {
        split: [int(len(np.setdiff1d(arrays[split][:, a], arrays["train"][:, a])))
                for a in range(3)] for split in ("valid", "test")}
    return arrays, identities, metadata


# SECTION: Geometry and Riemannian updates
@njit(cache=True)
def poincare_distance(a, b):
    alpha = max(1.0 - np.dot(a, a), 1e-12)
    beta = max(1.0 - np.dot(b, b), 1e-12)
    diff = a - b
    gamma = max(1.0, 1.0 + 2.0 * np.dot(diff, diff) / (alpha * beta))
    return math.acosh(gamma)


@njit(cache=True)
def distance_gradients(a, b):
    aa, bb = np.dot(a, a), np.dot(b, b)
    ab = np.dot(a, b)
    alpha, beta = max(1.0 - aa, 1e-12), max(1.0 - bb, 1e-12)
    diff = a - b
    sq = np.dot(diff, diff)
    if sq < 1e-28:
        # Distance is nondifferentiable at coincidence; choose zero subgradient.
        return 0.0, np.zeros_like(a), np.zeros_like(b)
    gamma = max(1.0, 1.0 + 2.0 * sq / (alpha * beta))
    factor = 4.0 / (alpha * beta * math.sqrt(max(gamma * gamma - 1.0, 1e-24)))
    ga = factor * (((bb - 2.0 * ab + 1.0) / alpha) * a - b)
    gb = factor * (((aa - 2.0 * ab + 1.0) / beta) * b - a)
    return math.acosh(gamma), ga, gb


@njit(cache=True)
def project_ball(vector, margin=1e-5):
    radius = math.sqrt(np.dot(vector, vector))
    limit = 1.0 - margin
    if radius > limit:
        return vector * (limit / radius)
    return vector


@njit(cache=True)
def predict_transformed(s, d, t, rows, scale):
    predictions = np.empty(len(rows), dtype=np.float64)
    for n in range(len(rows)):
        i, j, k = int(rows[n, 0]), int(rows[n, 1]), int(rows[n, 2])
        dist_sd = poincare_distance(s[i], d[j])
        dist_st = poincare_distance(s[i], t[k])
        dist_dt = poincare_distance(d[j], t[k])
        predictions[n] = scale * (1.0 - math.tanh(dist_sd + dist_st + dist_dt))
    return predictions


@njit(cache=True)
def train_epoch(s, d, t, rows, scale, learning_rate, order):
    for n in order:
        i, j, k = int(rows[n, 0]), int(rows[n, 1]), int(rows[n, 2])
        a, b, c = s[i].copy(), d[j].copy(), t[k].copy()
        ab, ga_ab, gb_ab = distance_gradients(a, b)
        ac, ga_ac, gc_ac = distance_gradients(a, c)
        bc, gb_bc, gc_bc = distance_gradients(b, c)
        tangent = math.tanh(ab + ac + bc)
        prediction = scale * (1.0 - tangent)
        loss_distance_grad = 2.0 * (rows[n, 3] - prediction) * scale * (1.0 - tangent * tangent)
        aa, bb, cc = np.dot(a, a), np.dot(b, b), np.dot(c, c)
        step_a = learning_rate * math.exp(-math.sqrt(aa)) * (1.0 - aa) ** 2 / 4.0
        step_b = learning_rate * math.exp(-math.sqrt(bb)) * (1.0 - bb) ** 2 / 4.0
        step_c = learning_rate * math.exp(-math.sqrt(cc)) * (1.0 - cc) ** 2 / 4.0
        s[i] = project_ball(a - step_a * loss_distance_grad * (ga_ab + ga_ac))
        d[j] = project_ball(b - step_b * loss_distance_grad * (gb_ab + gb_bc))
        t[k] = project_ball(c - step_c * loss_distance_grad * (gc_ac + gc_bc))


def transform_target(y, mode):
    return np.log10(y + 1.0) if mode == "log10" else np.asarray(y).copy()


def inverse_target(y, mode):
    return np.power(10.0, y) - 1.0 if mode == "log10" else np.asarray(y).copy()


def regression_metrics(y, prediction):
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Non-finite predictions; training is numerically unstable")
    error = prediction - y
    positive = y > 0
    return {"rmse": float(np.sqrt(np.mean(error * error))),
        "mae": float(np.mean(np.abs(error))),
        "mape_percent": float(100 * np.mean(np.abs(error[positive] / y[positive]))) if positive.any() else None,
        "mape_excluded_zero_targets": int((~positive).sum()), "n_entries": len(y)}


@dataclass
class Config:
    rank: int = 20
    max_epochs: int = 100
    learning_rate: float = 0.001
    init_scale: float = 0.001
    patience: int = 10
    min_delta: float = 1e-5
    seed: int = 42
    target_transform: str = "log10"
    shuffle: bool = False
    keep_zero: bool = False


# SECTION: Validation-selected fitting and final evaluation
def fit(train, valid, shape, config: Config):
    rng = np.random.default_rng(config.seed)
    factors = [rng.uniform(-config.init_scale, config.init_scale,
                           size=(size, config.rank)) for size in shape]
    transformed = train.copy()
    transformed[:, 3] = transform_target(train[:, 3], config.target_transform)
    scale = float(transformed[:, 3].max())
    if scale <= 0:
        raise ValueError("Training scale must be positive")
    order = np.arange(len(train), dtype=np.int64)
    # Compile kernels before timing training; restore initial factors afterwards.
    warmup = [x.copy() for x in factors]
    start_jit = time.perf_counter()
    train_epoch(*warmup, transformed[:1], scale, config.learning_rate, np.array([0], dtype=np.int64))
    predict_transformed(*warmup, valid[:1], scale)
    jit_seconds = time.perf_counter() - start_jit
    start = time.perf_counter()
    best, best_epoch, waited = math.inf, 0, 0
    best_factors = [x.copy() for x in factors]
    history = []
    for epoch in range(config.max_epochs + 1):
        if epoch:
            if config.shuffle:
                rng.shuffle(order)
            train_epoch(*factors, transformed, scale, config.learning_rate, order)
        prediction = inverse_target(predict_transformed(*factors, valid, scale), config.target_transform)
        measured = regression_metrics(valid[:, 3], prediction)
        value = measured["rmse"]
        history.append({"epoch": epoch, "valid_rmse": value,
                        "valid_mae": measured["mae"], "elapsed_seconds": time.perf_counter() - start})
        if value < best - config.min_delta:
            best, best_epoch, waited = value, epoch, 0
            best_factors = [x.copy() for x in factors]
        else:
            waited += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} validation_RMSE={value:.6f}", flush=True)
        if epoch and waited >= config.patience:
            break
    metadata = {"best_epoch": best_epoch, "best_validation_rmse": best,
        "epochs_trained": epoch, "training_and_validation_seconds": time.perf_counter() - start,
        "jit_warmup_seconds": jit_seconds, "training_target_scale": scale,
        "stopped_early": epoch < config.max_epochs}
    return best_factors, scale, history, metadata


def baseline_predictions(train, heldout, n_sensors):
    mean = float(train[:, 3].mean())
    ids = train[:, 0].astype(np.int64)
    sums = np.bincount(ids, weights=train[:, 3], minlength=n_sensors)
    counts = np.bincount(ids, minlength=n_sensors)
    by_sensor = np.full(n_sensors, mean)
    np.divide(sums, counts, out=by_sensor, where=counts > 0)
    return {"global_mean": np.full(len(heldout), mean),
            "sensor_mean": by_sensor[heldout[:, 0].astype(np.int64)]}


def write_csv(path, rows):
    import csv
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_city(data_dir: Path, output: Path, city: str, config: Config):
    output.mkdir(parents=True, exist_ok=True)
    arrays, identities, data_metadata = load_city(data_dir, city, config.keep_zero)
    train, valid, test = arrays["train"], arrays["valid"], arrays["test"]
    factors, scale, history, metadata = fit(train, valid, data_metadata["shape"], config)
    start = time.perf_counter()
    prediction = inverse_target(predict_transformed(*factors, test, scale), config.target_transform)
    metadata["test_prediction_seconds"] = time.perf_counter() - start
    all_predictions = {"HLFT": prediction, **baseline_predictions(train, test, data_metadata["shape"][0])}
    metrics = [{"city": city, "model": name, **regression_metrics(test[:, 3], p)}
               for name, p in all_predictions.items()]
    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "history.csv", history)
    np.savez_compressed(output / "model.npz", S=factors[0], D=factors[1], T=factors[2],
        sensor_ids=identities[0], interval_ids=identities[1], day_ids=identities[2],
        scale=scale, target_transform=config.target_transform)
    # Plot examples are a deterministic subset, not the easiest predictions.
    sample = np.random.default_rng(config.seed).choice(len(test), min(500, len(test)), replace=False)
    write_csv(output / "prediction_sample.csv", ({"sensor_id": int(identities[0][int(test[i, 0])]),
        "interval_id": int(identities[1][int(test[i, 1])]),
        "day_id": int(identities[2][int(test[i, 2])]),
        "observed": float(test[i, 3]), "predicted": float(prediction[i])} for i in sample))
    package_versions = {name: importlib.metadata.version(name) for name in ("numpy", "numba", "llvmlite", "matplotlib")}
    run_metadata = {"city": city, "config": asdict(config), "data": data_metadata,
        "fit": metadata, "python": platform.python_version(), "packages": package_versions,
        "platform": platform.platform(), "metric_units": "supplied dataset speed units",
        "protocol": "supplied random-entry split; validation-RMSE checkpoint; single final test evaluation"}
    (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print(f"{city}: best_epoch={metadata['best_epoch']}; test_RMSE={metrics[0]['rmse']:.6f}", flush=True)
    return metrics


# SECTION: Command line
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--dataset", choices=CITIES + ["all"], default="Guangzhou")
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--init-scale", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-transform", choices=["log10", "identity"], default="log10")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--keep-zero", action="store_true")
    args = parser.parse_args()
    if args.rank < 1 or args.max_epochs < 1 or args.patience < 1 or args.learning_rate <= 0:
        parser.error("rank, epochs, patience, and learning rate must be positive")
    if args.init_scale <= 0 or args.init_scale * math.sqrt(args.rank) >= 1 - 1e-5:
        parser.error("Initialization must remain inside the Poincare ball")
    config = Config(rank=args.rank, max_epochs=args.max_epochs, learning_rate=args.learning_rate,
        init_scale=args.init_scale, patience=args.patience, seed=args.seed,
        target_transform=args.target_transform, shuffle=args.shuffle, keep_zero=args.keep_zero)
    metrics = []
    for city in CITIES if args.dataset == "all" else [args.dataset]:
        metrics.extend(run_city(args.data_dir, args.output / city, city, config))
    write_csv(args.output / "summary.csv", metrics)


if __name__ == "__main__":
    main()
