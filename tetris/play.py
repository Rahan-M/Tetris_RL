from tetris_engine import TetrisEngine, BOARD_HEIGHT, BOARD_WIDTH
import numpy as np

def board_with_piece(engine: TetrisEngine)->np.ndarray:
    board=engine.board.copy()
    mat=engine.get_current_matrix()
    h, w=mat.shape
    x, y=engine.piece_x, engine.piece_y

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

def displayBoard(engine: TetrisEngine):
    obs=board_with_piece(engine)
    for i in range(BOARD_HEIGHT):
        for j in range(BOARD_WIDTH):
            if(obs[i][j]==1):
                print(".", end="")
            else:
                print("#", end="")
        print("\n", end="")
    # print(len(obs))
    # print(BOARD_WIDTH)

def displayMenu():
    print("1. Move Left\n2. Move Right\n3. Rotate Piece\n4. Continue\n5. Hard Drop\n6. Controls Drop Piece\n7. Exit")

engine=TetrisEngine()
engine.reset_board()
alive=engine.spawn_piece()

total_steps=0
total_lines_cleared=0
total_score=0

print("Welcome to Tetris")
displayMenu()

while(True):
    lines_cleared=0
    displayBoard(engine)
    ch=int(input(""))
    valid=True
    total_steps+=1

    if(ch==5):
        lines_cleared, alive=engine.hard_drop()
    elif ch==6:
        displayMenu()
    elif(ch==7):
        break
    else:
        if ch==1:
            valid=engine.move_left()
        elif (ch==2):
            valid=engine.move_right()
        elif ch==3:
            valid=engine.rotate_piece()
        
        lines_cleared, alive=engine.go_down()

    if not valid:
        print("Invalid Move")
        total_steps-=1

    
    
    if not alive or total_steps>100:
        print("GAME OVER")
        break
    
    if lines_cleared!=0:
        print(f"{lines_cleared} Lines Cleared!!!")
        print("+", 2**lines_cleared)
        total_score=2**lines_cleared
        total_lines_cleared+=lines_cleared

print("Great Job")
print(f"Total Lines Cleared = {total_lines_cleared}")
print(f"Total Score = {total_score}")