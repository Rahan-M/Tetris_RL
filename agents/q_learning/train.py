import torch, random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from tetris.tetris_env import TetrisGymEnv
from agents.q_learning.q_network import QNetwork
from agents.q_learning.replay_buffer import ReplayBuffer

def train_q_learning(
    episodes=500,
    batch_size=64,
    gamma=0.99, # controls how much the agent cares about the long term rewards, we set this high(close to 1) as tetris is a long horizon game
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.995,
    lr=1e-3, #Controls how fast the Q-network updates during gradient descent. 1e-3 is the standard default (step size of weight updates)
    buffer_capacity=50000
):
    env=TetrisGymEnv(observation_type="heuristics")
    input_dim=3
    output_dim=env.action_space.n

    q_network = QNetwork(input_dim, output_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start

    for episode in range(episodes):
        state=env.reset()
        episode_reward=0
        done=False

        while not done:
            if random.random()<epsilon:
                action=env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor=torch.tensor(state, dtype=torch.float32).unsqueeze(0) # (batch_size, features) here (1,3) without unsqueeze batch size will be (3,)
                    q_values=q_network(state_tensor) 
                    # When you call q_network(x)
                    # PyTorch automatically calls the forward() function.
                    action=torch.argmax(q_values).item() # returns index of largest value, item() converts tensor to integer

            # take an action and record new state
            next_state, reward, done, info = env.step(action)

            # store in replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            state=next_state
            episode_reward+=reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states=torch.tensor(states, dtype=torch.float32)
                actions=torch.tensor(actions, dtype=torch.long)
                rewards=torch.tensor(rewards, dtype=torch.float32)
                next_states=torch.tensor(next_states, dtype=torch.float32)
                dones=torch.tensor(dones, dtype=torch.float32)

                q_values=q_network(states)
                q_s_a=q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # q values of the actions taken are stored here shape is (64,)
                
                with torch.no_grad():
                    next_q = q_network(next_states)
                    max_next_q = next_q.max(1)[0] # max q for each action, max_next_q is shape (batch,)
                    target = rewards + gamma * max_next_q * (1 - dones) # target coming from the Bellman equation.

                # target is what q network should output ideally

                loss = loss_fn(q_s_a, target) # loss = mean((predicted - target)^2)

                # back prop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epsilon = max(epsilon_end, epsilon * epsilon_decay)

        print(f"Episode {episode+1}/{episodes} | Reward: {episode_reward:.2f} | Epsilon: {epsilon:.3f}")

    print("Training complete.")
    torch.save(q_network.state_dict(), "models/q_learning/model.pth")
    print("Model saved to models/q_learning/model.pth")

if __name__ == "__main__":
    train_q_learning()