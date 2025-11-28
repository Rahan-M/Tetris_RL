import torch
import numpy as np

from tetris.tetris_env import TetrisGymEnv
from agents.q_learning.q_network import QNetwork

MODEL_PATH = "models/q_learning/model.pth"

def evaluate(episodes=5):
    env = TetrisGymEnv(observation_type="heuristics")

    q_network = QNetwork(input_dim=3, output_dim=4)
    q_network.load_state_dict(torch.load(MODEL_PATH))
    q_network.eval()

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_lines = 0

        print(f"Starting episode {ep}")
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            action = torch.argmax(q_values).item()

            next_state, reward, done, info = env.step(action)

            total_reward += reward
            print(f"Action {action} total reward {total_reward}")
            total_lines += info.get("lines_cleared", 0)
            # print(state)
            state = next_state

        print(f"Episode {ep+1}: Reward={total_reward:.3f}, Lines={total_lines}")

if __name__ == "__main__":
    evaluate()
