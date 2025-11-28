# dqn_sb3

Observation space:
1. 2 Boards Tetris board Empty board with just the falling piece (2,20,10)
3. Heuristcs (3,)

heuristic normalization matters a LOT when you mix:
CNN input (board: 0/1)
MLP input (heuristics: big numbers like 80 height, 15 holes, 40 bumpiness)
If you don’t normalize, the MLP features will be on a much larger scale than the CNN output, and the network will give them disproportionate weight.