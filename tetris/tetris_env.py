import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any

from tetris_engine import TetrisEngine, BOARD_HEIGHT, BOARD_WIDTH

class TetrisGymEnv(gym.Env): # it inherits from gym.Env so we must implement __init__ reset and step, optaionally render and close as well
    metadata={"render.modes": ["human"]}

    def __init__(
        self,
        observation_type: str="heuristics", # Controls what state the agent receives 
        step_penalty: float=-0.01, # a small negative reward for every action to prevent infinite random moves
        lines_cleared_reward: float = 1.0, 
        death_penalty: float=-10.0,
        normalize: bool=True # whether to scale heuristics to [0, 1]
    ):
        super().__init__() # run parent class's init before continuing
        assert observation_type in ("heuristics", "board", "board_and_heuristics"), \
            "observation_type must be one of 'heuristics', 'board', 'board_and_heuristics'"

        # assert stops program execution if condition is false and shows the following error

        self.engine = TetrisEngine()
        self.observation_type = observation_type
        self.step_penalty = step_penalty
        self.lines_cleared_reward = lines_cleared_reward
        self.death_penalty = death_penalty
        self.normalize = normalize

        # action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)

        max_agg = BOARD_HEIGHT*BOARD_WIDTH # max value for aggregate height
        max_holes = BOARD_HEIGHT*BOARD_WIDTH # max value for number of holes
        max_bump = BOARD_HEIGHT*BOARD_WIDTH # A value greater than max value of bumpiness

        if observation_type=="heuristics":
            low=np.array([0.0,0.0,0.0], dtype=np.float32)
            high=np.array([max_agg, max_holes, max_bump], dtype=np.float32)

            if self.normalize:
                high=np.ones(3, dtype=np.float32)
            self.observation_space=spaces.Box(low=low, high=high, dtype=np.float32)
            # the observation space would be Box(low=[0,0,0], high=[1,1,1], shape=(3,))

        elif observation_type=="board":
            # Board is 20x10; we'll send it as float32 0/1
            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32
            )
            # spaces.Box(low=0.0, high=1.0, shape=(20,10), dtype=float32)

        else:  # board_and_heuristics
            # Flattened board + 3 heuristics
            board_shape = BOARD_HEIGHT * BOARD_WIDTH
            if self.normalize:
                low_board = np.zeros(board_shape + 3, dtype=np.float32) # a zero vector of size 203
                high_board = np.ones(board_shape + 3, dtype=np.float32) # a vector contianing all 1s of size 203
                self.observation_space = spaces.Box(low=low_board, high=high_board, dtype=np.float32)
            else:
                low = np.zeros(board_shape + 3, dtype=np.float32)
                high = np.concatenate([
                    np.ones(board_shape, dtype=np.float32) * 1.0,
                    np.array([max_agg, max_holes, max_bump], dtype=np.float32)
                ])
                self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.last_info={} # This is just for debugging — Gym allows returning metadata.
        self.reset() # This initializes the board and spawns the first piece. Makes environment valid immediately

    def reset(self, *, seed:int =None, return_info:bool = False, options: Dict=None):
        self.engine.reset_board()
        alive=self.engine.spawn_piece()
        obs=self._get_obs() # the underscore in the beginning indicates the function is a private method, just a convention

        self.last_info={"alive":alive, "lines cleared":0}
        if return_info:
            return obs, self.last_info
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
            Perform an action:
                0 -> move_left()
                1 -> move_right()
                2 -> rotate_piece()
                3 -> hard_drop() (locks piece, returns  (lines_cleared, alive) )
            Returns: obs, reward, done, info
        """

        assert self.action_space.contains(action), f"Invalid action {action}"
       
        lines_cleared=0
        alive=True

        if action == 0:
            _ = self.engine.move_left()
        elif action == 1:
            _ = self.engine.move_right()
        elif action == 2:
            _ = self.engine.rotate_piece()
        elif action == 3:
            lines_cleared, alive = self.engine.hard_drop()
        else:
            raise ValueError("Unknown action")

        reward=self._compute_reward(lines_cleared, action)
        obs=self._get_obs()
        done=not alive

        self.last_info = {"alive": alive, "lines_cleared": int(lines_cleared)}
        info = self.last_info.copy()

        return obs, float(reward), bool(done), info
    
    def _get_obs(self) -> np.ndarray:
        """
            Returns observation according to observation type. Returns state of the system you could say
        """
        if self.observation_type=="heuristics":
            heur=self.engine.get_heuristics().astype(np.float32)
            if self.normalize:
                heur=self._normalize_heuristics(heur)
            return heur

        elif self.observation_type=="board_and_heuristics":
            board_with_piece=self._board_with_piece()
            return board_with_piece.astype(np.float32)
        else:
            board_with_piece=self._board_with_piece().flatten().astype(np.float32)
            heur=self.engine.get_heuristics().astype(np.float32)
            if self.normalize:
                heur=self._normalize_heuristics(heur)
            return np.concatenate([board_with_piece, heur], axis=1)
        
    def _normalize_heuristics(self, heur:np.ndarray) -> np.ndarray:
        max_agg = BOARD_HEIGHT*BOARD_WIDTH # max value for aggregate height
        max_holes = BOARD_HEIGHT*BOARD_WIDTH # max value for number of holes
        max_bump = BOARD_HEIGHT*BOARD_WIDTH # A value greater than max value of bumpiness

        heur=np.array([heur[0]/max_agg, heur[1]/max_holes, heur[2]/max_bump], dtype=np.float32)
        return heur
    
    def _board_with_piece(self)->np.ndarray:
        board=self.engine.board.copy()
        mat=self.engine.get_current_matrix()
        h, w=mat.shape
        x, y=self.engine.piece_x, self.engine.piece_y

        # board indexing from x0, y0 to x1, y1 (opp corners of a rectangle)
        x0=max(0, x)
        y0=max(0, y)

        x1=min(BOARD_WIDTH, x+w)
        y1=min(BOARD_HEIGHT, y+h)
        # we consider invalid states because when the game fails, the piece maybe partially out of bounds
        # we have to still train the agent on this data

        # board indexing from mat_x0, mat_y0 to mat_x1, mat_y1 (opp corners of a rectangle)
        mat_x0=x0-x
        mat_y0=y0-y

        mat_x1=mat_x0 + x1-x0
        mat_y1=mat_y0 + y1-y0

        if y0<y1 and x0<x1: 
            board[y0:y1, x0:x1]|= mat[mat_y0:mat_y1, mat_x0:mat_x1]

        return board
    
    def _compute_reward(self, lines_cleared:int, alive:bool) ->float:
        reward=0.0
        if(lines_cleared>0):
            reward+=self.lines_cleared_reward*(2**lines_cleared) 
            # exponential reward for clearing multiple lines at once
        reward+=self.step_penalty
        if not alive:
            reward+=self.death_penalty
        return float(reward)
    
    def render(self, mode:str="human"):
        board=self._board_with_piece()
        
        print("-"*BOARD_WIDTH)
        print("\n")
        for row in board:
            for entry in row:
                if entry==0:
                    print("# ", end="")
                else:
                    print(". ", end="")
            print("\n")
        print("-"*BOARD_WIDTH)
        print("\n")

    def close(self):
        pass

    def get_last_info(self)->Dict[str, Any]:
        return self.last_info
    