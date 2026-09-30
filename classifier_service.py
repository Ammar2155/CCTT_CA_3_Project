"""
classifier_service.py

Tiny Flask wrapper around the exported logistic-regression classifier
(hpc_classifier.json from classifier/train_classifier.py), deployed on the
`classifier_service` EC2 instance. The orchestration harness calls this once
per job before routing it to the flat or nested tier.

Run: python3 classifier_service.py --model hpc_classifier.json --port 5000
"""
import argparse
import json
import math
from flask import Flask, request, jsonify

app = Flask(__name__)
MODEL = {}


def load_model(path):
    with open(path) as f:
        return json.load(f)


def classify(features: dict) -> dict:
    x = [features[name] for name in MODEL["feature_order"]]
    z = MODEL["bias"] + sum(w * xi for w, xi in zip(MODEL["weights"], x))
    p = 1 / (1 + math.exp(-z))
    is_hpc = p >= MODEL["threshold"]
    return {"class": "HPC" if is_hpc else "HTC", "probability": p}


@app.route("/classify", methods=["POST"])
def classify_endpoint():
    features = request.get_json(force=True)
    missing = [f for f in MODEL["feature_order"] if f not in features]
    if missing:
        return jsonify({"error": f"missing features: {missing}"}), 400
    return jsonify(classify(features))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "feature_order": MODEL.get("feature_order")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hpc_classifier.json")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    MODEL = load_model(args.model)
    app.run(host="0.0.0.0", port=args.port, threaded=True)
