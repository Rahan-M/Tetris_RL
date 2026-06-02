import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time

# Ensure we can import the environment
# Adjust this path based on where you run the script from!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tetris.tetris_env import TetrisGymEnv

# --- 1. SumTree for Prioritized Experience Replay (PER) ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write_ptr = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write_ptr + self.capacity - 1
        self.data[self.write_ptr] = data
        self.update(idx, p)
        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

# --- 2. Prioritized Replay Buffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, idxs, torch.FloatTensor(is_weight)

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (np.abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)

    def size(self):
        return self.tree.n_entries

# --- 3. Dueling Q-Network Architecture ---
class DuelingQNetwork(nn.Module):
    def __init__(self, num_actions=5):
        super(DuelingQNetwork, self).__init__()
        
        # Board CNN (2 channels: board + piece) -> 20x10 input
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Flattened CNN output = 64 * 20 * 10 = 12800
        
        # Heuristics MLP (3 features)
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined feature representation
        self.fc = nn.Sequential(
            nn.Linear(12800 + 64, 512),
            nn.ReLU()
        )
        
        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage Stream A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, board, features):
        cnn_out = self.cnn(board)
        mlp_out = self.mlp(features)
        
        combined = torch.cat([cnn_out, mlp_out], dim=1)
        x = self.fc(combined)
        
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# --- 4. Main Training Loop ---
def train():
    print("Initializing environment...")
    env = TetrisGymEnv(observation_type="board_and_heuristics")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Hyperparameters ---
    num_episodes = 50000
    batch_size = 64
    gamma = 0.99
    learning_rate = 1e-4
    target_update_freq = 2500  # Sync target network every N frames
    buffer_capacity = 100_000
    learning_starts = 10_000
    
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_frames = 200_000
    
    alpha = 0.6  # PER priority exponent
    beta_start = 0.4 # PER importance sampling starting beta
    beta_frames = 200_000
    
    # Init networks
    model = DuelingQNetwork(env.action_space.n).to(device)
    target_model = DuelingQNetwork(env.action_space.n).to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity, alpha=alpha)
    
    frame_idx = 0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Starting Training: Dueling Double DQN with PER...")
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        lines_cleared = 0
        
        while not (done or truncated):
            # Epsilon Decay
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * frame_idx / epsilon_decay_frames)
            epsilon = max(epsilon_end, epsilon)
            
            # Epsilon-greedy action
            if random.random() < epsilon or frame_idx < learning_starts:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    b_t = torch.FloatTensor(obs["board"]).unsqueeze(0).to(device)
                    f_t = torch.FloatTensor(obs["features"]).unsqueeze(0).to(device)
                    q_values = model(b_t, f_t)
                    action = q_values.argmax(dim=1).item()
                    
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Save Experience to Prioritized Buffer
            replay_buffer.add(obs, action, reward, next_obs, float(done))
            
            obs = next_obs
            episode_reward += reward
            if "lines_cleared" in info:
                lines_cleared += info["lines_cleared"]
                
            frame_idx += 1
            
            # --- Network Update ---
            if replay_buffer.size() > learning_starts and frame_idx % 4 == 0:
                beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
                
                batch = replay_buffer.sample(batch_size, beta)
                b_states, b_actions, b_rewards, b_next_states, b_dones, b_idxs, b_weights = batch
                b_weights = b_weights.unsqueeze(1).to(device)
                
                # Convert dict states to tensors
                b_boards = torch.FloatTensor(np.array([s["board"] for s in b_states])).to(device)
                b_feats = torch.FloatTensor(np.array([s["features"] for s in b_states])).to(device)
                b_next_boards = torch.FloatTensor(np.array([s["board"] for s in b_next_states])).to(device)
                b_next_feats = torch.FloatTensor(np.array([s["features"] for s in b_next_states])).to(device)
                
                b_actions = torch.LongTensor(b_actions).unsqueeze(1).to(device)
                b_rewards = torch.FloatTensor(b_rewards).unsqueeze(1).to(device)
                b_dones = torch.FloatTensor(b_dones).unsqueeze(1).to(device)
                
                # Double DQN Logic
                with torch.no_grad():
                    # Main network selects the best action
                    next_actions = model(b_next_boards, b_next_feats).argmax(dim=1, keepdim=True)
                    # Target network evaluates the action
                    next_q_values = target_model(b_next_boards, b_next_feats).gather(1, next_actions)
                    target_q = b_rewards + gamma * next_q_values * (1 - b_dones)
                    
                # Current Q values
                current_q = model(b_boards, b_feats).gather(1, b_actions)
                
                # Update Priorities based on TD Error
                td_errors = (current_q - target_q).abs().detach().cpu().numpy().flatten()
                replay_buffer.update_priorities(b_idxs, td_errors)
                
                # Weighted Loss for PER
                loss = (b_weights * nn.MSELoss(reduction='none')(current_q, target_q)).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Update Target Network
            if frame_idx % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())
                
        # --- Logging and Checkpointing ---
        if episode % 10 == 0:
            print(f"Ep: {episode:5d} | Frames: {frame_idx:7d} | Reward: {episode_reward:7.2f} | Lines: {lines_cleared:4d} | Eps: {epsilon:.3f}")
            
        if episode % 1000 == 0:
            ckpt_path = os.path.join(save_dir, f"dueling_ddqn_per_ep_{episode}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    train()
