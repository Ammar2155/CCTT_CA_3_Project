
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

N_RESOURCES = 10          
WINDOW = 8                 
N_CHANNELS = 3             
GAMMA = 0.98
LR = 1e-4
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
ENTROPY_BONUS = 0.02


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.bn1 = nn.GroupNorm(8, channels)  
        self.bn2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class TANSActorCritic(nn.Module):


    def __init__(self, n_resources=N_RESOURCES, n_channels=N_CHANNELS, window=WINDOW,
                 hidden=64, n_res_blocks=3, extra_features=4):
        super().__init__()
        self.n_resources = n_resources
        in_ch = n_channels + extra_features + 1          
        self.stem = nn.Conv2d(in_ch, hidden, (1, 3), padding=(0, 1))
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_res_blocks)])
        self.mix = nn.Conv1d(hidden * 2, hidden, 1)     
        self.actor_head = nn.Conv1d(hidden, 1, 1)
        self.critic_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))

        
        coord = torch.linspace(0, 1, n_resources).view(1, 1, n_resources, 1).expand(1, 1, n_resources, window).clone().contiguous()
        self.register_buffer("coord_base", coord, persistent=False)

        nn.init.normal_(self.actor_head.weight, std=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)

    def forward(self, grid, extra):
        B, _, N, W = grid.shape
        ex = extra[:, :, None, None].expand(B, -1, N, W)
        coord = self.coord_base.expand(B, -1, -1, -1)
        x = torch.cat([grid, ex, coord], dim=1)
        x = F.relu(self.stem(x))
        x = self.res_blocks(x)                            
        x = x[:, :, :, -1]                                
        g = x.mean(dim=2, keepdim=True).expand_as(x)      
        h = F.relu(self.mix(torch.cat([x, g], dim=1)))    
        logits = self.actor_head(h).squeeze(1)            
        logits = torch.clamp(logits, min=-5.0, max=5.0)   
        value = self.critic_head(h.mean(dim=2))           
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
        prev = {}  
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
