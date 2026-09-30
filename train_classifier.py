"""
train_classifier.py

Trains a lightweight (logistic-regression) HPC/HTC task classifier on real
cluster-trace features, and exports it as plain JSON coefficients so the Java
side (HPCClassifierLoader.java) needs zero ML runtime dependency.

Replaces the manuscript's `Class(t_i) = HPC if i mod 5 == 0` rule with a
classifier derived from real workload structure.

Data sources (pick one):
  - Google Cluster Trace 2019: https://github.com/google/cluster-data
    (instance_events / instance_usage tables; access via BigQuery public
    dataset or the gsutil download instructions in that repo)
  - Alibaba Cluster Trace 2018: https://github.com/alibaba/clusterdata
    (batch_task.csv / batch_instance.csv)

This script expects a pre-extracted CSV with the columns below — do the
BigQuery/gsutil pull separately (it's multi-GB; sample it, don't load it all).
A synthetic fallback generator is included so you can validate the pipeline
end-to-end before the real trace is ready.

Usage:
    python train_classifier.py --input alibaba_sample.csv --output hpc_classifier.json
    python train_classifier.py --synthetic --output hpc_classifier.json   # pipeline smoke test
"""
import argparse
import json
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

FEATURE_COLUMNS = [
    "cpu_request",      # normalized [0,1], fraction of machine CPU requested
    "mem_request",      # normalized [0,1], fraction of machine memory requested
    "duration_norm",    # normalized estimated/actual duration
    "io_intensity",     # normalized disk/network IO indicator (0 if unavailable)
]


ALIBABA_BATCH_TASK_COLUMNS = [
    "task_name", "instance_num", "job_name", "task_type",
    "status", "start_time", "end_time", "plan_cpu", "plan_mem",
]


def load_alibaba_batch_task(path: str, nrows: int = None) -> "pd.DataFrame":
    """Alibaba batch_task.csv (2018 trace) ships with NO header row -- columns
    are positional per the official schema (trace_2018.md in alibaba/clusterdata).
    nrows: pass e.g. 1_000_000 to sample instead of loading the whole file
    (batch_task.csv is several GB uncompressed)."""
    df = pd.read_csv(path, header=None, names=ALIBABA_BATCH_TASK_COLUMNS, nrows=nrows)
    df = df.dropna(subset=["plan_cpu", "plan_mem", "start_time", "end_time"])
    df = df[(df["plan_cpu"] > 0) & (df["plan_mem"] > 0)]  # drop malformed/placeholder rows
    df = df[df["end_time"] > df["start_time"]]  # drop negative/zero duration rows
    
    df["cpu_request"] = df["plan_cpu"] / df["plan_cpu"].max()
    df["mem_request"] = df["plan_mem"] / df["plan_mem"].max()
    
    # Log-transform duration to handle long-tail distributions [0, 1]
    duration = df["end_time"] - df["start_time"]
    df["duration_norm"] = np.log1p(duration) / np.log1p(duration.max())
    df["io_intensity"] = 0.0  # not present in this table; wire up instance_usage if you join it in
    return df


def synthetic_dataset(n=20000, seed=0) -> "pd.DataFrame":
    """Smoke-test data with a similar structure to real traces: a minority of
    tasks are genuinely high-cpu/short/latency-sensitive (HPC-like)."""
    rng = np.random.default_rng(seed)
    is_hpc = rng.random(n) < 0.20
    cpu = np.where(is_hpc, rng.beta(5, 2, n), rng.beta(2, 5, n))
    mem = np.where(is_hpc, rng.beta(4, 2, n), rng.beta(2, 4, n))
    duration = np.where(is_hpc, rng.beta(2, 6, n), rng.beta(3, 3, n))  # HPC tends shorter/bursty here
    io = rng.beta(2, 5, n)
    df = pd.DataFrame({
        "cpu_request": cpu, "mem_request": mem,
        "duration_norm": duration, "io_intensity": io,
        "_true_label": is_hpc.astype(int),
    })
    return df


def derive_labels(df: "pd.DataFrame") -> np.ndarray:
    """Rule-based weak label from REAL trace fields (replaces i % 5 == 0):
    HPC = high CPU request AND short-to-moderate duration (latency-sensitive,
    compute-bound). Tune thresholds against your trace's actual distribution
    (e.g. top quartile) rather than hardcoding 0.7/0.4 blindly."""
    if "_true_label" in df.columns:
        return df["_true_label"].to_numpy()
    
    print("\n--- Trace Feature Distributions ---")
    print("cpu_request summary:\n", df["cpu_request"].describe())
    print("duration_norm summary:\n", df["duration_norm"].describe())
    
    cpu_thresh = df["cpu_request"].quantile(0.75)
    dur_thresh = df["duration_norm"].quantile(0.30)
    print(f"\nTuning thresholds: cpu_thresh (75th percentile)={cpu_thresh:.4f}, dur_thresh (30th percentile)={dur_thresh:.4f}")
    
    label = ((df["cpu_request"] >= cpu_thresh) & (df["duration_norm"] <= dur_thresh)).astype(int)
    pos_ratio = label.mean() * 100
    print(f"HPC Positive Label Ratio: {pos_ratio:.2f}% ({label.sum()}/{len(label)})\n")
    return label.to_numpy()


def train_logistic_regression(X: np.ndarray, y: np.ndarray, l2=0.01, lr=2.0, epochs=1000):
    """Hand-rolled logistic regression (no sklearn dependency required) so the
    whole pipeline stays inspectable. Swap in sklearn.linear_model.LogisticRegression
    freely if it's already in your environment — same exported JSON shape."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))
        grad_w = X.T @ (p - y) / n + l2 * w / n
        grad_b = np.mean(p - y)
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def evaluate(X, y, w, b):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    pred = (p >= 0.5).astype(int)
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"accuracy": float(acc), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="path to extracted trace CSV")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic smoke-test data")
    ap.add_argument("--output", type=str, default="hpc_classifier.json")
    ap.add_argument("--sample-rows", type=int, default=1_000_000,
                     help="rows to read from the real trace file (it's multi-GB; don't load it all)")
    args = ap.parse_args()

    if pd is None:
        raise SystemExit("pandas is required: pip install pandas numpy")

    if args.synthetic or not args.input:
        print("Using synthetic smoke-test dataset (pass --input for the real trace).")
        df = synthetic_dataset()
    else:
        df = load_alibaba_batch_task(args.input, nrows=args.sample_rows)
        print(f"Loaded {len(df)} real trace rows from {args.input}")

    y = derive_labels(df)
    X = df[FEATURE_COLUMNS].to_numpy()

    # simple 80/20 split
    n = len(X)
    idx = np.random.RandomState(0).permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]

    w, b = train_logistic_regression(X[train_idx], y[train_idx])
    metrics = evaluate(X[test_idx], y[test_idx], w, b)
    print("Held-out metrics:", json.dumps(metrics, indent=2))

    export = {
        "feature_order": FEATURE_COLUMNS,
        "weights": w.tolist(),
        "bias": float(b),
        "threshold": 0.5,
        "held_out_metrics": metrics,
        "note": "HPC=1 if sigmoid(w.x + b) >= threshold, else HTC=0",
    }
    with open(args.output, "w") as f:
        json.dump(export, f, indent=2)
    print(f"Wrote classifier to {args.output}")


if __name__ == "__main__":
    main()
