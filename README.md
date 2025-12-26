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


I have tried training a model on several times, 
at first there was no gravity, so you move left or right and rotate the piece as many times as you want and then hard drop
version 1: Death penalty was a 1000 times greater than step penalty (10 and 0.01). So the model will never drop and keep playing with the piece infinitely
result 1: training will get stuck at evaluation
version 2: Added max steps, to avoid infinity. 
result 2: But model will make 500 steps with no drop
version 3: Started changing step penalty
result 3: if it was too great, model will die asap to avoid it, if it was too little model will never drop a piece. Couldn't find a suitable middle ground
version4: Introduced penalties for changing the heuristics in a negative manner, 
result 4: still stuck at the dilemma from previous version

version 5: Added gravity, so pieces fall one level after each move. Avoids infinite loops by guranteeing death. Removed step penalty and penalty for incresing aggregate height. (V3)
result 5: Model survives for around 250-300 steps but doesn't clear any line
version 6: Reintroduced aggregate height penalty. Introduced a survival reward of 0.01 (V4)
result 6: Same as result 5, but model seems close to clearing lines
version 7: incrase exploration fraction from 0.3 to 0.5 and exploration_final_eps to 0.05 from 0.02 (V5)
result 7: same as 6
version 8: tuned many of the parameters of the model and train on 3M timesteps (V6)