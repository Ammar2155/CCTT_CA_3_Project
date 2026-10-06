
import argparse
import json
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

FEATURE_COLUMNS = [
    "cpu_request",      
    "mem_request",      
    "duration_norm",    
    "io_intensity",     
]


ALIBABA_BATCH_TASK_COLUMNS = [
    "task_name", "instance_num", "job_name", "task_type",
    "status", "start_time", "end_time", "plan_cpu", "plan_mem",
]


def load_alibaba_batch_task(path: str, nrows: int = None) -> "pd.DataFrame":

    df = pd.read_csv(path, header=None, names=ALIBABA_BATCH_TASK_COLUMNS, nrows=nrows)
    df = df.dropna(subset=["plan_cpu", "plan_mem", "start_time", "end_time"])
    df = df[(df["plan_cpu"] > 0) & (df["plan_mem"] > 0)] 
    df = df[df["end_time"] > df["start_time"]]  
    
    df["cpu_request"] = df["plan_cpu"] / df["plan_cpu"].max()
    df["mem_request"] = df["plan_mem"] / df["plan_mem"].max()
    
    duration = df["end_time"] - df["start_time"]
    df["duration_norm"] = np.log1p(duration) / np.log1p(duration.max())
    df["io_intensity"] = 0.0  
    return df


def synthetic_dataset(n=20000, seed=0) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    is_hpc = rng.random(n) < 0.20
    cpu = np.where(is_hpc, rng.beta(5, 2, n), rng.beta(2, 5, n))
    mem = np.where(is_hpc, rng.beta(4, 2, n), rng.beta(2, 4, n))
    duration = np.where(is_hpc, rng.beta(2, 6, n), rng.beta(3, 3, n)) 
    io = rng.beta(2, 5, n)
    df = pd.DataFrame({
        "cpu_request": cpu, "mem_request": mem,
        "duration_norm": duration, "io_intensity": io,
        "_true_label": is_hpc.astype(int),
    })
    return df


def derive_labels(df: "pd.DataFrame") -> np.ndarray:
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
