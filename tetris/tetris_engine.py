import numpy as np
BOARD_HEIGHT=20
BOARD_WIDTH=10

TETROMINOES={
    "I":[
        np.array([1,1,1,1]), 
        np.array([1],[1],[1],[1])
    ],
    "O":[
        np.array([1,1],[1,1])
    ],
    "T":[
        np.array([1,1,1],[0,1,0]),
        np.arrray([0,1],[1,1],[0,1]),
        np.array([0,1,0],[1,1,1]),
        np.array([1,0],[1,1],[1,0])
    ],
    "L":[
        np.array([1,0],[1,0],[1,1]),
        np.array([1,1,1],[1,0,0]),
        np.array([1,1],[0,1],[0,1]),
        np.array([0,0,1],[1,1,1])
    ],
    "J":[
        np.array([0,1],[0,1],[1,1]),
        np.array([1,0,0],[1,1,1]),
        np.array([1,1],[1,0],[1,0]),
        np.array([1,1,1],[0,0,1])
    ],
    "S":[
        np.array([0,1,1],[1,1,0]),
        np.array([1,0],[1,1],[0,1])
    ],
    "Z":[
        np.array([1,1,0],[0,1,1]),
        np.array([0,1],[1,1],[1,0])
    ],
}

class TetrisEngine:
    def __init__(self):
        self.board=np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int) # initialize the board
        self.current_piece=None
        self.current_rotation=0
        self.piece_x=0
        self.piece_y=0
    
    def reset_board(self):
        self.board=np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int) # initialize the board
        
    def check_collision(self, matrix, x, y):
        piece_h, piece_w=matrix.shape

        # check boundaries
        if x<0 or x+piece_w>BOARD_WIDTH:
            return True
        if y<0 or y+piece_h>BOARD_HEIGHT:
            return True
        
        board_section=self.board[y:y+piece_h, x:x+piece_w]
        if(matrix and board_section).any():
            return True
        
        return False
        

    def spawn_piece(self):
        shape_key = np.random.choice(list(TETROMINOES.keys()))
        rotations = TETROMINOES[shape_key]
        
        self.current_piece=rotations
        self.current_rotation=0
        matrix=rotations[0]

        piece_h, piece_w=matrix.shape
        self.piece_x=(10-piece_w)//2
        self.piece_y=0

        if self.check_collision(matrix, self.piece_x, self.piece_y):
            return False # game over

        return True