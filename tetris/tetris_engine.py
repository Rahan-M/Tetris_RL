import numpy as np
BOARD_HEIGHT=20
BOARD_WIDTH=10

TETROMINOES={
    "I":[
        np.array([[1,1,1,1]]), 
        np.array([[1],[1],[1],[1]])
    ],
    "O":[
        np.array([[1,1],[1,1]])
    ],
    "T":[
        np.array([[1,1,1],[0,1,0]]),
        np.array([[0,1],[1,1],[0,1]]),
        np.array([[0,1,0],[1,1,1]]),
        np.array([[1,0],[1,1],[1,0]])
    ],
    "L":[
        np.array([[1,0],[1,0],[1,1]]),
        np.array([[1,1,1],[1,0,0]]),
        np.array([[1,1],[0,1],[0,1]]),
        np.array([[0,0,1],[1,1,1]])
    ],
    "J":[
        np.array([[0,1],[0,1],[1,1]]),
        np.array([[1,0,0],[1,1,1]]),
        np.array([[1,1],[1,0],[1,0]]),
        np.array([[1,1,1],[0,0,1]])
    ],
    "S":[
        np.array([[0,1,1],[1,1,0]]),
        np.array([[1,0],[1,1],[0,1]])
    ],
    "Z":[
        np.array([[1,1,0],[0,1,1]]),
        np.array([[0,1],[1,1],[1,0]])
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
        if(matrix & board_section).any(): # checks if overlap exists at any point
            # this is equivalent to matrix[0][0] and board[0][0] or matrix[0][1] and board[0][1] or ... matrix[2][1] and board[2][1] so on
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
    
    def get_current_matrix(self):
        return self.current_piece[self.current_rotation]
    
    def move_left(self):
        matrix=self.get_current_matrix()
        new_x=self.piece_x-1
        
        if self.check_collision(matrix, new_x, self.piece_y):
            return False
        
        self.piece_x=new_x
        return True

    def move_right(self):
        matrix=self.get_current_matrix()
        new_x=self.piece_x+1
        
        if self.check_collision(matrix, new_x, self.piece_y):
            return False
        
        self.piece_x=new_x
        return True
    
    def rotate_piece(self):
        new_rot=(self.current_rotation+1) % len(self.current_piece)
        new_mat=self.current_piece[new_rot]

        if self.check_collision(new_mat, self.piece_x, self.piece_y):
            return False
        
        self.current_rotation=new_rot
        return True

    def go_down(self):
        matrix=self.get_current_matrix()

        if not self.check_collision(matrix, self.piece_x, self.piece_y+1): # if next drop will cause a collision, stop
            self.piece_y+=1 # update position
            return 0, True
        
        lines_cleared=self.lock_piece()
        alive=self.spawn_piece()
        return lines_cleared, alive

    def hard_drop(self):
        matrix=self.get_current_matrix()
        new_y=self.piece_y

        while True:
            if self.check_collision(matrix, self.piece_x, new_y+1): # if next drop will cause a collision, stop
                break
            new_y+=1
        
        self.piece_y=new_y # update position

        lines_cleared=self.lock_piece()
        alive=self.spawn_piece()

        return lines_cleared, alive
    
    def lock_piece(self):
        matrix = self.get_current_matrix()
        h, w = matrix.shape
        x, y = self.piece_x, self.piece_y

        # Optimized: Vectorized update using OR operator (add 1s to board)
        # We slice board[y:y+h, x:x+w] and OR it with the piece matrix
        self.board[y:y+h, x:x+w] |= matrix # same as += or -=

        return self.clear_lines()
    
    def clear_lines(self):
        # Check which rows are all ones
        full_rows = np.all(self.board == 1, axis=1) # gives one output per row
        # output would be something like [False, False, True, False, True, ...]
        lines_cleared = np.sum(full_rows)
        # number of true values is full rows

        if lines_cleared > 0: # if no lines are cleared no change
            # Keep only rows that are NOT full
            non_full_board = self.board[~full_rows] 
            # Pad with zeros on top
            padding = np.zeros((lines_cleared, BOARD_WIDTH), dtype=int)
            self.board = np.vstack((padding, non_full_board))
        
        return lines_cleared
    
    def get_column_heights(self):
        # Create a boolean mask where the board has blocks (1)
        mask = self.board != 0
        
        # argmax returns the index of the first 'True' value along axis 0 (rows). 
        # If no True value exists, it returns 0.
        # We need to handle columns that are completely empty.
        first_block_indices = np.argmax(mask, axis=0)
        
        # Check if the columns actually have blocks
        has_blocks = np.any(mask, axis=0)
        
        # Calculate heights: Board Height - Index of first block
        heights = np.where(has_blocks, BOARD_HEIGHT - first_block_indices, 0)
        return heights

    def get_heuristics(self):
        heights = self.get_column_heights()
        
        # 1. Aggregate Height
        aggregate_height = np.sum(heights)
        
        # 2. Bumpiness (Sum of absolute differences between adjacent columns)
        # We calculate diff between col 0&1, 1&2, etc.
        bumpiness = np.sum(np.abs(heights[:-1] - heights[1:]))
        
        # 3. Holes
        # A hole is a 0 that has a 1 somewhere above it in the same column.
        # We can iterate columns or use cumsum tricks. Here is a readable iterative approach:
        holes = 0
        for x in range(BOARD_WIDTH):
            if heights[x] == 0:
                continue # No blocks, no holes
            
            # Look at the segment of the column from the first block down to the bottom
            # Any 0 in this segment is a hole.
            column_segment = self.board[BOARD_HEIGHT-heights[x]:, x] # sekects column with index x and the rows from the first 1 to end
            holes += np.count_nonzero(column_segment == 0)

        return np.array([aggregate_height, holes, bumpiness])
