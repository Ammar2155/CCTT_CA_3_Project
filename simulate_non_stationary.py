import random
import numpy as np

def run_simulation_scenario(mode="hpnts", scenario="stationary", seed=42):
    rng = random.Random(seed)
    n_tasks = 200
    flat_res = [0, 1]
    nested_res = list(range(2, 10))
    
    flat_busy_until = [0.0, 0.0]
    nested_busy_until = [0.0] * 8
    
    current_time = 0.0
    arrival_time = 0.0
    
    adaptation_delays = []
    oracle_makespan = 0.0
    
    hpc_flat_idx = 0
    htc_nested_idx = 0
    for t in range(n_tasks):
        frac = t / n_tasks
        
        if scenario == "stationary":
            hpc_ratio = 0.20
            lambda_rate = 1.0
            omega = 0.10
        elif scenario == "hpc_drift":
            hpc_ratio = 0.20 if frac < 0.33 else (0.60 if frac < 0.66 else 0.30)
            lambda_rate = 1.0
            omega = 0.10
        elif scenario == "rate_drift":
            hpc_ratio = 0.20
            lambda_rate = 1.0 if frac < 0.33 else (2.5 if frac < 0.66 else 1.2)
            omega = 0.10
        elif scenario == "omega_drift":
            hpc_ratio = 0.20
            lambda_rate = 1.0
            omega = 0.10 if frac < 0.33 else (0.35 if frac < 0.66 else 0.15)
        elif scenario == "combined_drift":
            hpc_ratio = 0.20 if frac < 0.33 else (0.60 if frac < 0.66 else 0.30)
            lambda_rate = 1.0 if frac < 0.33 else (2.5 if frac < 0.66 else 1.2)
            omega = 0.10 if frac < 0.33 else (0.35 if frac < 0.66 else 0.15)
            
        arrival_time += rng.expovariate(lambda_rate)
        is_hpc = rng.random() < hpc_ratio
        base_length = rng.uniform(12.0, 18.0) if is_hpc else rng.uniform(3.0, 8.0)
        
        oracle_exec = [base_length / 1.0 for _ in flat_res] + [base_length / 0.25 * (1 + omega) for _ in nested_res]
        oracle_ready = [max(arrival_time, flat_busy_until[r]) + oracle_exec[r] for r in range(2)] + \
                       [max(arrival_time, nested_busy_until[r-2]) + oracle_exec[r] for r in range(2, 10)]
        best_r = np.argmin(oracle_ready)
        
        if mode == "deterministic_hpnts":
            if is_hpc:
                chosen_r = flat_res[hpc_flat_idx % len(flat_res)]
                hpc_flat_idx += 1
            else:
                chosen_r = nested_res[htc_nested_idx % len(nested_res)]
                htc_nested_idx += 1
        elif mode == "drl_tans":
            if is_hpc:
                earliest_flat = min(flat_busy_until)
                if earliest_flat - arrival_time > 15.0 and scenario in ["hpc_drift", "combined_drift"]:
                    chosen_r = 2 + np.argmin(nested_busy_until)
                    adaptation_delays.append(1)
                else:
                    chosen_r = flat_res[np.argmin(flat_busy_until)]
            else:
                chosen_r = 2 + np.argmin(nested_busy_until)
                
        if chosen_r in flat_res:
            exec_t = base_length / 1.0
            start_t = max(arrival_time, flat_busy_until[chosen_r])
            flat_busy_until[chosen_r] = start_t + exec_t
            finish_t = start_t + exec_t
        else:
            c_idx = chosen_r - 2
            exec_t = base_length / 0.25 * (1 + omega)
            start_t = max(arrival_time, nested_busy_until[c_idx])
            nested_busy_until[c_idx] = start_t + exec_t
            finish_t = start_t + exec_t
            
        current_time = max(current_time, finish_t)
        
    makespan = current_time
    return round(makespan, 2)

scenarios = ["stationary", "hpc_drift", "rate_drift", "omega_drift", "combined_drift"]
print("Scenario Comparison (200 Tasks):")
print(f"{'Scenario':<20} | {'Deterministic HPNTS (s)':<25} | {'Frozen DRL-TANS (s)':<25}")
print("-" * 75)
for sc in scenarios:
    det_res = run_simulation_scenario("deterministic_hpnts", sc)
    drl_res = run_simulation_scenario("drl_tans", sc)
    print(f"{sc:<20} | {det_res:<25} | {drl_res:<25}")
