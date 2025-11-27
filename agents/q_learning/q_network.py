import torch
import torch.nn as nn

class QNetwork(nn.Module):
    """
        a simple mlp that maps observation to q values for each action
        input dim: 3, here for the manual training we are going to take 3 dimensions, the heuristics to be exact
        output dim: 4, beacuse there are only 4 possible actions
    """

    def __init__(self, input_dim=3, output_dim=4):
        super(QNetwork, self).__init__()
        
        # define a single two layer mlp
        self.network=nn.Sequential(
            nn.Linear(input_dim, 64), # take the 3 heuristics and expand into 64 neurons
            nn.ReLU(), # Nonlinear activation so the network can learn complex patterns.
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )


    def forward(self, x):
        """
            Forward pass.
            x: tensor of shape (batch_size, input_dim)
            returns: Q-values for each action -> (batch_size, output_dim)
        """
        return self.network(x)