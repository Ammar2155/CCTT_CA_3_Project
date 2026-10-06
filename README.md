# HPNTS & DRL-TANS: Adaptive Hybrid Cloud Task Scheduling Framework

[![AWS Cloud](https://img.shields.io/badge/AWS-EC2%20%7C%20SSM%20%7C%20Terraform-orange.svg)](https://aws.amazon.com/)
[![CloudSim](https://img.shields.io/badge/Simulation-CloudSim-blue.svg)](http://www.cloudbus.org/cloudsim/)
[![PyTorch](https://img.shields.io/badge/DRL-PyTorch%20Residual--CNN-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end framework implementing **HPNTS (High-Performance Nested Tier Scheduling)** and **DRL-TANS (Deep Reinforcement Learning - Task Adaptive Nested Scheduling)**. This project provides both a discrete-event simulation harness via **CloudSim** (Java) and a real-world cloud deployment harness on **Amazon Web Services (AWS)** using **Terraform** and **AWS Systems Manager (SSM)**.

---

## 🌟 Executive Summary

In heterogeneous multi-tier cloud environments, workloads vary significantly between compute-intensive **High-Performance Computing (HPC)** tasks and I/O-bound or lightweight **High-Throughput Computing (HTC)** tasks. Placing compute-bound HPC tasks into containerized nested environments can introduce high virtual interference penalties ($\Omega_{\text{virtual}}$), while scheduling HTC tasks on dedicated flat VMs wastes expensive compute capacity.

This framework solves resource allocation bottlenecks by introducing:
1. **Machine Learning Classification**: Real-time logistic regression classifier that dynamically categorizes incoming cloudlets into HPC vs. HTC based on resource requests, duration, and I/O intensity.
2. **Multi-Tier Resource Mapping**:
   - **Flat Tier**: High-capacity bare-metal / dedicated EC2 instances for compute-bound HPC tasks.
   - **Nested Tier**: Containerized host environments for lightweight HTC tasks with tracked interference penalties.
3. **DRL-TANS Agent**: A PyTorch Residual-CNN Actor-Critic reinforcement learning model that dynamically routes overflow tasks and adapts to non-stationary workload shifts (ratio drift, arrival rate bursts, and container interference spikes).
4. **AWS Infrastructure & Orchestration**: Automated Terraform provisioning of multi-tier AWS EC2 topology and Python SSM harness (`orchestrate.py`) to benchmark real wall-clock execution times.

---

## 🛠️ System Architecture

```
                                +---------------------------+
                                |  Incoming Workload Batch  |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |  ML Classifier Service    |
                                |  (HPC vs HTC Prediction)  |
                                +-------------+-------------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
                       v                                             v
        +------------------------------+              +------------------------------+
        |     HPC Compute Workloads    |              |     HTC Container Tasks      |
        +--------------+---------------+              +--------------+---------------+
                       |                                             |
                       v                                             v
        +------------------------------+              +------------------------------+
        |   Flat Dedicated VM Tier     |              |     Nested Container Tier    |
        |   (High Capacity, Low Penalty) |              |   (Shared Host + Ω_virtual)  |
        +------------------------------+              +------------------------------+
                       ^                                             ^
                       |                                             |
                       +----------------------+----------------------+
                                              |
                                              v
                                +---------------------------+
                                |   PyTorch DRL-TANS Agent  |
                                | (Residual CNN Scheduler)  |
                                +---------------------------+
```

---

## 📂 Repository Structure

```
├── README.md                           # Comprehensive Project Documentation
├── .gitignore                          # Excludes large trace binaries (>100MB) & security keys
├── main.tf                             # Terraform infrastructure manifest for AWS EC2 deployment
├── orchestrate.py                      # Real-world AWS SSM benchmark & orchestration harness
├── teardown.sh                         # Cleanup shell script for AWS cloud resources
│
├── classifier_service.py               # Flask REST microservice serving HPC classifier on EC2
├── train_classifier.py                 # Logistic regression classifier training script
│
├── tans_agent.py                       # PyTorch Residual-CNN Actor-Critic TCP Server
├── train_standalone.py                 # Standalone offline DRL agent training script
├── simulate_non_stationary.py          # Synthetic & trace non-stationary drift simulator
├── generate_drl_results_and_plots.py   # Plotting & statistical validation harness
│
├── HPCClassifierLoader.java            # Java ML classifier loader for CloudSim
├── HPNTS_Comparative_Project.java      # Main CloudSim simulation benchmark suite
├── TANSBridgeClient.java               # Java TCP socket bridge connecting CloudSim to DRL agent
├── T_Test_Validation.java              # Statistical significance validator (Paired t-test)
│
├── src/main/java/org/cloudbus/cloudsim/examples/
│   ├── HPCClassifierLoader.java        # Package-structured CloudSim Java components
│   ├── HPNTS_Comparative_Project.java
│   ├── TANSBridgeClient.java
│   └── T_Test_Validation.java
│
└── AWS Results/                        # Experimental benchmark datasets & AWS outputs
    ├── aws_results_fcfs*.json          # FCFS Baseline execution metrics
    ├── aws_results_hpnts*.json         # Deterministic HPNTS metrics
    ├── aws_results_drl_tans*.json      # DRL-TANS adaptive scheduling metrics
    ├── hpc_classifier.json             # Exported classifier weights & threshold
    └── outputs.json                    # Provisioned AWS Terraform instance output mappings
```

---

## 💻 Prerequisites & Setup

### 1. Requirements
- **Java**: JDK 11+ & CloudSim 3.0.3+
- **Python**: Python 3.8+
- **PyTorch**: `torch >= 2.0`
- **AWS CLI & Boto3**: AWS profile configured with permissions for EC2 and Systems Manager (SSM).
- **Terraform**: `terraform >= 1.0`

### 2. Python Dependencies Installation
```bash
pip install torch numpy scipy matplotlib seaborn flask requests boto3
```

---

## 🚀 Step-by-Step Usage Guide

### Phase 1: Train & Export ML Classifier
Train the logistic regression classifier to distinguish HPC and HTC tasks:
```bash
python train_classifier.py --output hpc_classifier.json
```
Launch the Flask microservice for real-time API inference:
```bash
python classifier_service.py --model hpc_classifier.json --port 5000
```

---

### Phase 2: PyTorch DRL-TANS Agent
Start the PyTorch Residual-CNN TCP server in training or evaluation mode:

**Training Mode:**
```bash
python tans_agent.py --port 8765 --train --checkpoint-path tans_checkpoint.pt
```

**Evaluation Mode (Frozen Policy):**
```bash
python tans_agent.py --port 8765 --eval tans_checkpoint_standalone.pt
```

---

### Phase 3: CloudSim Java Simulation Harness
Compile and run the CloudSim comparative evaluation:

```bash
javac -cp ".:lib/*" src/main/java/org/cloudbus/cloudsim/examples/*.java

java -cp ".:lib/*:src/main/java" org.cloudbus/cloudsim.examples.HPNTS_Comparative_Project

java -cp ".:lib/*:src/main/java" org.cloudbus.cloudsim.examples.T_Test_Validation
```

---

### Phase 4: AWS Infrastructure Provisioning & Real-World Orchestration

1. **Deploy EC2 Instances with Terraform:**
   ```bash
   terraform init
   terraform apply -auto-approve
   terraform output -json > outputs.json
   ```

2. **Run Real-World AWS SSM Workload Orchestration:**
   Execute benchmark workloads across provisioned EC2 instances:

   ```bash
   python orchestrate.py --terraform-outputs outputs.json --classifier-url http://<CLASSIFIER_IP>:5000/classify --n-jobs 50 --mode fcfs

   python orchestrate.py --terraform-outputs outputs.json --classifier-url http://<CLASSIFIER_IP>:5000/classify --n-jobs 50 --mode hpnts

   python orchestrate.py --terraform-outputs outputs.json --classifier-url http://<CLASSIFIER_IP>:5000/classify --n-jobs 50 --mode drl_tans
   ```

3. **Teardown Cloud Resources:**
   ```bash
   ./teardown.sh
   ```

---

### Phase 5: Non-Stationary Workload Drift & Plot Generation
Generate adaptation regret curves, makespan benchmarks, and publication figures:
```bash
python generate_drl_results_and_plots.py
```
This produces `drl_tans_adaptation_regret.png` showcasing performance under:
- **Stationary Workloads**
- **HPC Ratio Drift**
- **Arrival Rate Bursts**
- **$\Omega_{\text{virtual}}$ Interference Drift**
- **Combined Non-Stationary Drift**

---

## 📊 Experimental Results & Benchmarks

Real-world AWS benchmarking results (stored under `AWS Results/`) demonstrate significant performance gains:

| Strategy | Makespan Improvement vs. FCFS | Mean Latency Reduction | Adaptation Recovery Lag |
| :--- | :---: | :---: | :---: |
| **FCFS Baseline** | Reference (0%) | High | N/A (Static) |
| **Deterministic HPNTS** | ~15% - 25% | Moderate | ~38 Tasks |
| **DRL-TANS (Ours)** | **~35% - 48%** | **Optimal** | **~8 Tasks** |

Statistical significance confirmed via **Paired Student's t-test** ($p < 0.001$).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
