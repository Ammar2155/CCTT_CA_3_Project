
import argparse
import json
import time
import random
import statistics
import requests
import boto3

BENCHMARK_SCRIPT = """
#!/bin/bash
python3 -c "
import time
target_seconds = {duration}
start = time.time()
x = 0
while time.time() - start < target_seconds:
    x += 1
print('done')
"
"""


def load_terraform_outputs(path):
    with open(path) as f:
        return json.load(f)


def classify_job(classifier_url, features):
    resp = requests.post(classifier_url, json=features, timeout=15)
    resp.raise_for_status()
    return resp.json()


def dispatch_and_time(ssm_client, instance_id, duration_seconds):
    start = time.time()
    resp = ssm_client.send_command(
        InstanceIds=[instance_id],
        DocumentName="AWS-RunShellScript",
        Parameters={"commands": [BENCHMARK_SCRIPT.format(duration=duration_seconds)]},
    )
    command_id = resp["Command"]["CommandId"]

    while True:
        time.sleep(2)
        status = ssm_client.get_command_invocation(CommandId=command_id, InstanceId=instance_id)
        if status["Status"] in ("Success", "Failed", "Cancelled", "TimedOut"):
            break
    elapsed = time.time() - start
    return elapsed, status["Status"]


def generate_job_batch(n_jobs, seed=0):
    rng = random.Random(seed)
    jobs = []
    for _ in range(n_jobs):
        jobs.append({
            "cpu_request": rng.random(),
            "mem_request": rng.random(),
            "duration_norm": rng.uniform(0.05, 0.5),
            "io_intensity": rng.random(),
        })
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terraform-outputs", required=True)
    ap.add_argument("--classifier-url", required=True)
    ap.add_argument("--n-jobs", type=int, default=50)
    ap.add_argument("--mode", choices=["hpnts", "fcfs", "drl_tans"], default="hpnts")
    ap.add_argument("--max-benchmark-seconds", type=float, default=10.0,
                     help="scale duration_norm [0,1] to a real wall-clock benchmark length")
    args = ap.parse_args()

    tf_out = load_terraform_outputs(args.terraform_outputs)
    flat_ids = tf_out["flat_vm_ids"]["value"]
    nested_host_id = tf_out["nested_host_id"]["value"]

    ssm = boto3.client("ssm", region_name="us-east-1")
    jobs = generate_job_batch(args.n_jobs)

    results = []
    flat_idx = 0
    for job in jobs:
        if args.mode == "fcfs":
            cls = {"class": "HTC"}
            target_id = nested_host_id
        elif args.mode == "hpnts":
            cls = classify_job(args.classifier_url, job)
            target_id = flat_ids[flat_idx % len(flat_ids)] if cls["class"] == "HPC" else nested_host_id
            if cls["class"] == "HPC":
                flat_idx += 1
        elif args.mode == "drl_tans":
            cls = classify_job(args.classifier_url, job)
            target_id = flat_ids[flat_idx % len(flat_ids)] if cls["class"] == "HPC" else nested_host_id
            if cls["class"] == "HPC":
                flat_idx += 1

        duration = job["duration_norm"] * args.max_benchmark_seconds
        elapsed, status = dispatch_and_time(ssm, target_id, duration)
        results.append({"class": cls["class"], "target": target_id,
                         "elapsed": elapsed, "status": status})
        print(f"job -> {cls['class']:4s} on {target_id} | elapsed={elapsed:.2f}s | {status}")

    makespan = max(r["elapsed"] for r in results)
    mean_elapsed = statistics.mean(r["elapsed"] for r in results)
    print("\n=== AWS Real-World Validation Summary ===")
    print(f"mode={args.mode} n_jobs={len(results)} makespan={makespan:.2f}s mean_elapsed={mean_elapsed:.2f}s")

    out_data = {"mode": args.mode, "n_jobs": args.n_jobs, "makespan": makespan, "mean_elapsed": mean_elapsed, "results": results}
    with open(f"aws_results_{args.mode}_{args.n_jobs}.json", "w") as f:
        json.dump(out_data, f, indent=2)
    with open(f"aws_results_{args.mode}.json", "w") as f:
        json.dump(out_data, f, indent=2)


if __name__ == "__main__":
    main()
