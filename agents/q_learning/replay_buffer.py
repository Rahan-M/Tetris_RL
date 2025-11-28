import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        """
        :param capacity: maximum  number of transitions we can store in buffer
        """
        self.capacity=capacity
        self.buffer=[]
        self.position=0

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition into the buffer
        A circular queue
        """
        transition=(state, action, reward, next_state, done)
        if(len(self.buffer)<self.capacity):
            self.buffer.append(transition)
        else:
            self.buffer[self.position]=transition
        self.position=(self.position+1)%self.capacity
            
    def sample(self, batch_size):
        """
        Return a random batch of transitons
        """
        batch=random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
# states      → np.ndarray of shape (batch, obs_dim)
# actions     → np.ndarray of shape (batch,)
# rewards     → np.ndarray of shape (batch,)
# next_states → np.ndarray of shape (batch, obs_dim)
# dones       → np.ndarray of shape (batch,)

    
    def __len__(self):
        return len(self.buffer)