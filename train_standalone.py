"""
train_standalone.py

Trains TANSActorCritic against a lightweight SYNTHETIC scheduling environment
that mimics the CloudSim state/action/reward contract, but runs entirely in
Python with no socket, no JVM, no CloudSim -- so you can verify the RL
algorithm itself actually learns before spending time debugging the Java
bridge. If this script converges cleanly but your CloudSim-driven training
doesn't, the bug is in the Java-side wiring (seeding, reward computation,
or state feeding), not the network or training loop.

Also exposes per-episode POLICY ENTROPY -- the thing that would have caught
the "-38.93 repeated 16 times" result immediately (entropy near 0 = mode
collapse, not convergence).

Usage:
    python3 train_standalone.py --episodes 800
"""
import argparse
import time
import random

import numpy as np
import torch
import torch.nn.functional as F

from tans_agent import TANSTrainer, N_RESOURCES, WINDOW, N_CHANNELS, BATCH_SIZE

TASKS_PER_EPISODE = 50
FLAT_RESOURCES = [0, 1]                 # matches your 2 flat VMs
NESTED_RESOURCES = list(range(2, 10))   # matches your 8 nested containers
OMEGA_VIRTUAL = {r: 0.0 for r in FLAT_RESOURCES}
OMEGA_VIRTUAL.update({r: 0.10 for r in NESTED_RESOURCES})
CAPACITY = {r: 1.0 for r in FLAT_RESOURCES}
CAPACITY.update({r: 0.25 for r in NESTED_RESOURCES})


def hpc_ratio_for_phase(task_idx, n_tasks):
    """Matches the 3-phase drift curriculum from your screenshots:
    Phase 1 (0-33%): 20% HPC, Phase 2 (33-66%): 60% HPC (burst), Phase 3 (66-100%): 30% HPC."""
    frac = task_idx / n_tasks
    if frac < 0.33:
        return 0.20
    elif frac < 0.66:
        return 0.60
    else:
        return 0.30


class SyntheticEpisode:
    """One episode = TASKS_PER_EPISODE tasks arriving under the drift curriculum.
    Re-seeded per episode (this is the fix for the memorized-episode bug)."""

    def __init__(self, episode_seed):
        self.rng = random.Random(episode_seed)
        self.n_tasks = TASKS_PER_EPISODE
        self.util_history = {r: [0.0] * WINDOW for r in range(N_RESOURCES)}
        self.queue_history = {r: [0.0] * WINDOW for r in range(N_RESOURCES)}
        self.queue_depth = {r: 0 for r in range(N_RESOURCES)}
        self.task_idx = 0

    def current_state(self, task_is_hpc, mi_norm):
        grid = np.zeros((N_CHANNELS, N_RESOURCES, WINDOW), dtype=np.float32)
        for r in range(N_RESOURCES):
            grid[0, r, :] = self.util_history[r]
            grid[1, r, :] = self.queue_history[r]
            grid[2, r, :] = OMEGA_VIRTUAL[r]
        extra = np.array([
            1.0 if task_is_hpc else 0.0,
            mi_norm,
            min(1.0, sum(self.queue_depth.values()) / 20.0),
            self.task_idx / self.n_tasks,
        ], dtype=np.float32)
        return torch.tensor(grid), torch.tensor(extra)

    def step(self, action, task_is_hpc, mi_norm):
        """Returns reward for placing this task on `action`."""
        exec_time = mi_norm / CAPACITY[action] * (1 + OMEGA_VIRTUAL[action])
        queue_penalty = self.queue_depth[action] * 0.05
        mismatch_penalty = 0.5 if (task_is_hpc and action not in FLAT_RESOURCES) or \
                                    (not task_is_hpc and action in FLAT_RESOURCES and self.rng.random() < 0.3) else 0.0
        reward = -(exec_time + queue_penalty + mismatch_penalty)

        self.queue_depth[action] += 1
        if self.rng.random() < 0.4:  # some tasks "complete" and free up the queue
            self.queue_depth[action] = max(0, self.queue_depth[action] - 1)

        for r in range(N_RESOURCES):
            util = min(1.0, self.queue_depth[r] * CAPACITY[r] * 0.3 + self.rng.random() * 0.05)
            self.util_history[r] = self.util_history[r][1:] + [util]
            self.queue_history[r] = self.queue_history[r][1:] + [min(1.0, self.queue_depth[r] / 10.0)]

        self.task_idx += 1
        done = self.task_idx >= self.n_tasks
        return reward, done

    def next_task(self):
        ratio = hpc_ratio_for_phase(self.task_idx, self.n_tasks)
        is_hpc = self.rng.random() < ratio
        mi_norm = self.rng.uniform(0.6, 1.0) if is_hpc else self.rng.uniform(0.2, 0.6)
        return is_hpc, mi_norm


def run_episode(trainer, episode_seed, train=True):
    env = SyntheticEpisode(episode_seed)
    total_reward = 0.0
    entropies = []
    losses = []

    is_hpc, mi_norm = env.next_task()
    grid, extra = env.current_state(is_hpc, mi_norm)

    while True:
        with torch.no_grad():
            logits, _ = trainer.net(grid.unsqueeze(0), extra.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            entropy = float(-(probs * torch.log(probs + 1e-9)).sum().item())
            entropies.append(entropy)
            if train:
                action = int(torch.multinomial(probs, 1).item())
            else:
                action = int(torch.argmax(probs).item())

        reward, done = env.step(action, is_hpc, mi_norm)
        total_reward += reward

        if done:
            next_grid, next_extra = grid, extra  # terminal: reuse last state, done=True masks bootstrap
        else:
            is_hpc, mi_norm = env.next_task()
            next_grid, next_extra = env.current_state(is_hpc, mi_norm)

        if train:
            trainer.store(grid, extra, action, reward, next_grid, next_extra, done)
            step_res = trainer.train_step()
            if step_res is not None:
                loss_val, _ = step_res
                losses.append(loss_val)

        grid, extra = next_grid, next_extra
        if done:
            break

    avg_loss = sum(losses) / len(losses) if losses else None
    avg_entropy = sum(entropies) / len(entropies)
    return total_reward, avg_loss, avg_entropy


def random_policy_baseline(n_episodes=50):
    """Sanity floor: mean reward of a uniformly random policy."""
    rewards = []
    for ep in range(n_episodes):
        env = SyntheticEpisode(episode_seed=99000 + ep)
        total = 0.0
        is_hpc, mi_norm = env.next_task()
        while True:
            action = random.randint(0, N_RESOURCES - 1)
            reward, done = env.step(action, is_hpc, mi_norm)
            total += reward
            if done:
                break
            is_hpc, mi_norm = env.next_task()
        rewards.append(total)
    return sum(rewards) / len(rewards)


def heuristic_policy_baseline(n_episodes=50):
    """Sanity ceiling-ish: the deterministic 'HPC->flat, HTC->nested round robin' rule,
    i.e. what your existing HPNTS deterministic scheduler effectively does."""
    rewards = []
    for ep in range(n_episodes):
        env = SyntheticEpisode(episode_seed=98000 + ep)
        total = 0.0
        flat_i, nested_i = 0, 0
        is_hpc, mi_norm = env.next_task()
        while True:
            if is_hpc:
                action = FLAT_RESOURCES[flat_i % len(FLAT_RESOURCES)]
                flat_i += 1
            else:
                action = NESTED_RESOURCES[nested_i % len(NESTED_RESOURCES)]
                nested_i += 1
            reward, done = env.step(action, is_hpc, mi_norm)
            total += reward
            if done:
                break
            is_hpc, mi_norm = env.next_task()
        rewards.append(total)
    return sum(rewards) / len(rewards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--checkpoint", type=str, default="tans_checkpoint_standalone.pt")
    args = ap.parse_args()

    print("Computing baselines for reference...")
    rand_baseline = random_policy_baseline()
    heur_baseline = heuristic_policy_baseline()
    print(f"  random policy avg episode reward   = {rand_baseline:.3f}")
    print(f"  deterministic-heuristic avg reward = {heur_baseline:.3f}  <- this is what DRL needs to beat under drift")

    trainer = TANSTrainer()
    start = time.time()

    reward_history = []
    for ep in range(args.episodes):
        episode_seed = 1000 + ep  # DIFFERENT seed every episode -- this is the fix
        total_reward, avg_loss, avg_entropy = run_episode(trainer, episode_seed, train=True)
        reward_history.append(total_reward)

        if (ep + 1) % args.log_every == 0:
            recent = reward_history[-args.log_every:]
            avg_recent = sum(recent) / len(recent)
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "n/a (buffer filling)"
            elapsed = time.time() - start
            print(f"[ep {ep+1:4d}/{args.episodes}] avg_reward(last {args.log_every})={avg_recent:7.3f} "
                  f"loss={loss_str} entropy={avg_entropy:.3f} replay={len(trainer.replay)} "
                  f"elapsed={elapsed:.1f}s")

    trainer.save(args.checkpoint)
    total_time = time.time() - start
    print(f"\nDone. {args.episodes} episodes in {total_time:.1f}s "
          f"({total_time/args.episodes*1000:.1f}ms/episode). Saved {args.checkpoint}")

    # Frozen-policy eval against both baselines
    print("\nEvaluating frozen (greedy) policy vs baselines (50 eval episodes each)...")
    eval_rewards = []
    for ep in range(50):
        r, _, ent = run_episode(trainer, episode_seed=5000 + ep, train=False)
        eval_rewards.append(r)
    drl_avg = sum(eval_rewards) / len(eval_rewards)
    print(f"  random policy avg reward           = {rand_baseline:.3f}")
    print(f"  deterministic-heuristic avg reward = {heur_baseline:.3f}")
    print(f"  trained DRL-TANS avg reward        = {drl_avg:.3f}")
    if drl_avg > heur_baseline:
        print("  -> DRL-TANS beats the deterministic heuristic under this drifting curriculum.")
    else:
        print("  -> DRL-TANS has NOT beaten the deterministic heuristic yet -- needs more episodes/tuning.")


if __name__ == "__main__":
    main()
