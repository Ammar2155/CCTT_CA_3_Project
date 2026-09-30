"""
TANS Agent: Residual-CNN + Actor-Critic scheduler for HPNTS-DRL.

Runs as a local server. CloudSim (Java) connects over a TCP socket, sends the
current system state once per scheduling decision as a JSON line, and receives
back the chosen resource index. Training happens asynchronously from replayed
(state, action, reward, next_state) tuples that CloudSim also reports.

This is a SKELETON: state/action shapes match the HPNTS_Comparative_Project.java
setup (2 flat VMs + 8 nested containers = 10 resources). Tune reward shaping,
network depth, and hyperparameters against your own experiments before trusting
any numbers for the paper.

Usage:
    python tans_agent.py --port 8765 --train    # training mode, saves checkpoints
    python tans_agent.py --port 8765 --eval ckpt.pt   # frozen policy for comparison runs
"""
import argparse
import json
import socket
import socketserver
import threading
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    torch.set_num_threads(4)
except Exception:
    pass

N_RESOURCES = 10          # 2 flat VMs + 8 nested containers, matches the .java files
WINDOW = 8                 # rolling ticks of history per resource
N_CHANNELS = 3             # [utilization, queue_length, omega_virtual] per (resource, tick)
GAMMA = 0.98
LR = 1e-4
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
ENTROPY_BONUS = 0.02


class ResidualBlock(nn.Module):
    """Residual block with (1x3) kernels: convolves along TIME for each resource
    independently, so weights are shared across resources."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.bn1 = nn.GroupNorm(8, channels)  # GroupNorm: identical in train/eval, fine at batch size 1
        self.bn2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TANSActorCritic(nn.Module):
    """Fully-convolutional residual CNN over the [C, N_RESOURCES, WINDOW] state grid.
    Task features are broadcast as extra input channels and a resource-coordinate
    channel is added. Every resource is scored by the SAME weights (permutation-
    friendly inductive bias), then global context (mean over resources) is mixed in
    before the per-resource policy logit (actor) and a pooled value estimate (critic)."""

    def __init__(self, n_resources=N_RESOURCES, n_channels=N_CHANNELS, window=WINDOW,
                 hidden=64, n_res_blocks=3, extra_features=4):
        super().__init__()
        self.n_resources = n_resources
        in_ch = n_channels + extra_features + 1          # grid + task features + coordinate
        self.stem = nn.Conv2d(in_ch, hidden, (1, 3), padding=(0, 1))
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_res_blocks)])
        self.mix = nn.Conv1d(hidden * 2, hidden, 1)      # per-resource feature + global context
        self.actor_head = nn.Conv1d(hidden, 1, 1)
        self.critic_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

        # Register coordinate buffer [1, 1, N, W] to avoid recreating on every step
        coord = torch.linspace(0, 1, n_resources).view(1, 1, n_resources, 1).expand(1, 1, n_resources, window).clone().contiguous()
        self.register_buffer("coord_base", coord, persistent=False)

        # Initialize actor head weights small so initial logits are near 0 and entropy is high (~2.3)
        nn.init.normal_(self.actor_head.weight, std=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)

    def forward(self, grid, extra):
        B, _, N, W = grid.shape
        ex = extra[:, :, None, None].expand(B, -1, N, W)
        coord = self.coord_base.expand(B, -1, -1, -1)
        x = torch.cat([grid, ex, coord], dim=1)
        x = F.relu(self.stem(x))
        x = self.res_blocks(x)                            # [B, H, N, W]
        x = x[:, :, :, -1]                                # latest step (receptive field covers the window)
        g = x.mean(dim=2, keepdim=True).expand_as(x)      # global context across resources
        h = F.relu(self.mix(torch.cat([x, g], dim=1)))    # [B, H, N]
        logits = self.actor_head(h).squeeze(1)            # [B, N]
        logits = torch.clamp(logits, min=-5.0, max=5.0)   # prevent logit explosion and entropy collapse
        value = self.critic_head(h.mean(dim=2))           # [B, 1]
        return logits, value


Transition = collections.namedtuple("Transition", "grid extra action reward next_grid next_extra done")


class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buf, min(batch_size, len(self.buf)))

    def __len__(self):
        return len(self.buf)


class TANSTrainer:
    def __init__(self, device="cpu"):
        self.device = device
        self.net = TANSActorCritic().to(device)
        self.opt = optim.Adam(self.net.parameters(), lr=LR)
        self.replay = ReplayBuffer()

    def act(self, grid, extra, greedy=False):
        """grid: [N_CHANNELS, N_RESOURCES, WINDOW] float tensor (unbatched)
           extra: [extra_features] float tensor (unbatched)"""
        with torch.no_grad():
            logits, value = self.net(grid.unsqueeze(0), extra.unsqueeze(0))
            probs = F.softmax(logits, dim=-1).squeeze(0)
            if greedy:
                action = int(torch.argmax(probs).item())
            else:
                action = int(torch.multinomial(probs, 1).item())
        return action, float(value.item())

    def store(self, grid, extra, action, reward, next_grid, next_extra, done):
        self.replay.push(grid, extra, action, reward, next_grid, next_extra, done)

    def train_step(self):
        if len(self.replay) < BATCH_SIZE:
            return None
        batch = self.replay.sample(BATCH_SIZE)
        grids = torch.stack([t.grid for t in batch]).to(self.device)
        extras = torch.stack([t.extra for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_grids = torch.stack([t.next_grid for t in batch]).to(self.device)
        next_extras = torch.stack([t.next_extra for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        logits, values = self.net(grids, extras)
        with torch.no_grad():
            _, next_values = self.net(next_grids, next_extras)
            targets = rewards + GAMMA * next_values.squeeze(-1) * (1 - dones)
        
        raw_advantage = targets - values.squeeze(-1)
        adv_std = raw_advantage.std()
        norm_advantage = (raw_advantage - raw_advantage.mean()) / (adv_std + 1e-8) if adv_std > 1e-5 else raw_advantage

        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        actor_loss = -(chosen_log_probs * norm_advantage.detach()).mean()
        critic_loss = F.mse_loss(values.squeeze(-1), targets.detach())
        entropy = -(log_probs.exp() * log_probs).sum(dim=1).mean()
        loss = actor_loss + 0.01 * critic_loss - ENTROPY_BONUS * entropy

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()
        return float(loss.item()), float(entropy.item())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True), strict=False)


def parse_state(msg: dict):
    grid = torch.tensor(msg["grid"], dtype=torch.float32)
    extra = torch.tensor(msg["extra"], dtype=torch.float32)
    return grid, extra


class TCPHandler(socketserver.StreamRequestHandler):
    def handle(self):
        trainer: TANSTrainer = self.server.trainer
        train_mode: bool = self.server.train_mode
        action_history = collections.Counter()
        prev = {}  # per-connection last (grid, extra, action) for reward bookkeeping
        print(f"[eval-check] Handler started connection | train_mode={train_mode} | greedy={not train_mode}", flush=True)

        step_count = 0
        for line in self.rfile:
            try:
                msg = json.loads(line.decode("utf-8").strip())
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "step":
                grid, extra = parse_state(msg)
                greedy_flag = not train_mode
                action, value = trainer.act(grid, extra, greedy=greedy_flag)
                action_history[action] += 1
                step_count += 1

                if step_count <= 5 and not train_mode:
                    print(f"  [eval-step {step_count}] greedy={greedy_flag} chosen_action={action} value={value:.4f}", flush=True)

                if train_mode and "prev" in prev:
                    p_grid, p_extra, p_action = prev["prev"]
                    reward = float(msg.get("reward", 0.0))
                    done = bool(msg.get("done", False))
                    trainer.store(p_grid, p_extra, p_action, reward, grid, extra, done)
                    step_res = trainer.train_step()

                    if step_res is not None:
                        loss, ent = step_res
                        self.server.train_step_count += 1
                        self.server.recent_rewards.append(reward)
                        self.server.recent_entropies.append(ent)
                        n = self.server.train_step_count
                        log_every = self.server.log_every
                        ckpt_every = self.server.checkpoint_every
                        if log_every and n % log_every == 0:
                            recent_r = self.server.recent_rewards[-log_every:]
                            recent_e = self.server.recent_entropies[-log_every:]
                            avg_r = sum(recent_r) / max(len(recent_r), 1)
                            avg_e = sum(recent_e) / max(len(recent_e), 1)
                            print(f"[train] step={n} loss={loss:.4f} "
                                  f"avg_reward(last {log_every})={avg_r:.4f} "
                                  f"entropy={avg_e:.3f} "
                                  f"replay_size={len(trainer.replay)}", flush=True)
                        if ckpt_every and n % ckpt_every == 0:
                            trainer.save(self.server.checkpoint_path)
                            print(f"[checkpoint] saved to {self.server.checkpoint_path} at step={n}", flush=True)

                prev["prev"] = (grid, extra, action)
                self.wfile.write((json.dumps({"action": action, "value": value}) + "\n").encode("utf-8"))

            elif msg.get("type") == "checkpoint":
                trainer.save(msg.get("path", "tans_checkpoint.pt"))
                self.wfile.write((json.dumps({"ok": True}) + "\n").encode("utf-8"))

        if not train_mode and step_count > 0:
            print(f"[eval-histogram] Total actions={step_count} | Histogram={dict(sorted(action_history.items()))}", flush=True)


class TANSServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", type=str, default=None, help="checkpoint path to load for frozen-policy eval")
    ap.add_argument("--log-every", type=int, default=100, help="print loss/avg-reward every N train steps (0=off)")
    ap.add_argument("--checkpoint-every", type=int, default=500, help="auto-save every N train steps (0=off)")
    ap.add_argument("--checkpoint-path", type=str, default="tans_checkpoint.pt")
    args = ap.parse_args()

    trainer = TANSTrainer()
    if args.eval:
        trainer.load(args.eval)

    server = TANSServer(("localhost", args.port), TCPHandler)
    server.trainer = trainer
    server.train_mode = args.train and not args.eval
    server.train_step_count = 0
    server.recent_rewards = []
    server.recent_entropies = []
    server.log_every = args.log_every
    server.checkpoint_every = args.checkpoint_every
    server.checkpoint_path = args.checkpoint_path

    mode_desc = f"EVAL (frozen policy from {args.eval})" if args.eval else f"TRAIN (checkpoints -> {args.checkpoint_path})"
    print(f"TANS agent listening on localhost:{args.port} | mode={mode_desc}")
    server.serve_forever()


if __name__ == "__main__":
    main()
