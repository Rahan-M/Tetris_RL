# test_random_agent.py
import time
import numpy as np
from tetris_env import TetrisGymEnv

env = TetrisGymEnv(observation_type="heuristics")
obs = env.reset()
done = False
total_reward = 0.0
steps = 0

print("Starting random episode (heuristics obs).")
while not done and steps < 500:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    total_reward += reward
    steps += 1
    if steps % 20 == 0:
        print(f"steps={steps}, last_info={info}, total_reward={total_reward:.2f}")
    # render every hard drop (info.lines_cleared > 0) for visibility
    if info.get("lines_cleared", 0) > 0:
        print("LINES CLEARED!")
        env.render()
        time.sleep(0.2)

print("Episode finished. steps=", steps, "total_reward=", total_reward)
env.close()
