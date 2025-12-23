# Tetris_RL
A reinforcement learning project where I train an AI to play tetris

## Tetris Engine
We have created a tetris engine (with gravity) where the agent has the options at each state
1. Move left
2. Move right
3. Rotate piece
4. Do Nothing 
5. Hard drop

For all options except hard drop the piece drops by one row.

These actions are sufficient for strategic piece placement while avoiding unnecessary complexity.
Real Tetris includes soft drops, lock delays, hold mechanics, wall kicks, combo scoring, and multi-frame movement.
These features make gameplay richer for humans but dramatically increase the complexity for RL agents.

This simplification results in a deterministic, single-step control scheme where each action immediately affects the board state.
The agent focuses solely on where to place the current piece, not when.

We also use 3 heuristics:
1. Aggregate Height :  How high the stack is, the lower the better
2. Bumpiness : The jaggedness of the surface, the flatter the better
3. Holes : Empty spaces with filled blocks above them, the fewer the better

We calculate bumpiness by adding the sum of absolute differences between two adjacent columns, 1&2, 2&3, 3&4 so on

We track number of steps and stop at 500 because an intermediate model might cause an infinite loop otherwise by never using hard drop

The observation space fed to our dqn_sb3 model is (2, 20, 10) + (3,) where

We feed the board via two channels, one containing the board and fixed pieces and the other one contianing just the falling piece
We also feed it the three heuristics mentioned above