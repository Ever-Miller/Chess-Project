import numpy as np
import cv2 as cv
from problems import Board 
from algorithms import *
from dataclasses import dataclass, field
import random
import time

start_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"

# handles the base global flags for ui purposes
@dataclass
class UI:
    pos_idx: int
    white_idx: int
    black_idx: int
    depth: int
    msg: str
    nodes: int
    time: int

    positions: list[str] = field(default_factory=lambda : [
        "start position",
        "random position",
        "puzzle position"
    ])
    agents: list[str] = field(default_factory=lambda: [
        "Human", "MiniMaxV1", "MiniMaxV2", "MiniMaxV3", "MiniMaxV4", "MiniMaxV5", "MiniMaxV6", "MiniMaxV7",
        "MonteCarloV1", "MonteCarloV2", "RL"
    ])


ui = UI(0, 0, 0, 1, None, 0, 0)

# enums
white = 0
black = 1
both = 2

P, N, B, R, Q, K, p, n, b, r, q, k = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

# gets index of LSB
LS1B_IDX = {1<<i:i for i in range(64)}
# lookup table for 1 << n patterns n = {0 - 63}
BIT = tuple(1 << sq for sq in range(64))

PIECE_IMAGES = {
    P: "pieces-basic-png/white-pawn.png",
    N: "pieces-basic-png/white-knight.png",
    B: "pieces-basic-png/white-bishop.png",
    R: "pieces-basic-png/white-rook.png",
    Q: "pieces-basic-png/white-queen.png",
    K: "pieces-basic-png/white-king.png",
    p: "pieces-basic-png/black-pawn.png",
    n: "pieces-basic-png/black-knight.png",
    b: "pieces-basic-png/black-bishop.png",
    r: "pieces-basic-png/black-rook.png",
    q: "pieces-basic-png/black-queen.png",
    k: "pieces-basic-png/black-king.png",
}

img = np.zeros((1200, 1300, 3), np.uint8)
window = 'board'
cv.namedWindow(window)

# Global vars
CELL_SIZE = 100
AVAILABLE_MOVES = set()
SELECTED_MOVE = None
SELECTED_PIECE = -1
human_turn = -1
LOCKED = False
STATES = []

# get a random puzzle position from database
def get_random_puzzle() -> tuple[str]:
    n = random.randint(1, 5000)
    with open(r"puzzles\puzzle_subset.csv") as f:
        for _ in range(n):
            line = f.readline()

    _, fen, moves, _, _, _, _, _, _, _ = line.split(",")
    fen, side, castle, _, _, _ = fen.split()

    fen = " ".join([fen, side, castle])
    return fen, moves

# get a random opening positon from database
def get_random_position() -> str:
    n = random.randint(1, 5000)
    with open(r"openings\eco_fens.txt") as f:
        for _ in range(n):
            line = f.readline()

    return line

# Loads the base board
def load_board() -> None:
    font = cv.FONT_HERSHEY_TRIPLEX

    points = np.array([[100, 100], [100, 900], [900, 900], [900, 100]], np.int32)
    cv.fillPoly(img, pts=[points], color=(154, 203, 227))

    # Checker pattern
    for i in range(0, 800, 100):
        for j in range(0, 800, 100):
            points = np.array([
                [100 + j,     100 + i],
                [100 + j,     200 + i],
                [200 + j,     200 + i],
                [200 + j,     100 + i]
            ], np.int32)

            # switches back and forth throught green and tan
            if ((i / 100) + (j / 100)) % 2 == 0:
                color = (154, 203, 227)
            else:
                color = (59, 138, 76)

            cv.fillPoly(img, pts=[points], color=color)

            # Text for column letters and row numbers
            if j == 0:
                num = str((800 - i) // 100)
                pos = (101, 120 + i)
                color = (59, 138, 76) if (i / 100) % 2 == 0 else (154, 203, 227)
                cv.putText(img, num, pos, font, 0.7, color, 1, cv.LINE_AA, False)
            
            if i == 700:
                letter = chr( 97 + (j // 100) )
                pos = (184 + j, 894)
                color = (59, 138, 76) if (j / 100) % 2 == 1 else (154, 203, 227)
                cv.putText(img, letter, pos, font, 0.7, color, 1, cv.LINE_AA, False)

    cv.rectangle(img, (90, 90), (910, 910), (154, 203, 227), 5) 

# draw the pieces on the board or captured
def draw_pieces(board: Board) -> None:

    # loop thorugh all pieces
    for piece_type in range(12):
        bb = board.bitboards[piece_type]
        while bb:
            # extract lsb
            lsb = bb & -bb
            idx = LS1B_IDX[lsb]
            row, col = divmod(idx, 8)
            # get image for that piece 
            image = PIECE_IMAGES[piece_type]
    
            piece = cv.imread(image, cv.IMREAD_UNCHANGED)
            draw_piece(piece, row, col)

            bb ^= lsb

# draws the piece on the screeen
def draw_piece(image, row: int, col: int, captured=False) -> None:
    if captured:
        y = row
        x = col
        image = cv.resize(image, (50,50))
        size = 50
    else:
        y = 99 + (100 * row)
        x = 99 + (100 * col)
        image = cv.resize(image, (100,100))
        size = 100

    # Separate the color and alpha channels
    bgr = image[:, :, :3]
    alpha = image[:, :, 3] / 255.0

    # Get the region of interest on the board
    out = img[y:y+size, x:x+size]

    # Blend piece image with the board region for each color channel
    # blending formula below from https://www.geeksforgeeks.org/addition-blending-images-using-opencv-python/?ref=ml_lbp
    for c in range(3):
        out[:, :, c] = (alpha * bgr[:, :, c] + (1 - alpha) * out[:, :, c])
    img[y:y+size, x:x+size] = out

# draws the board bases
def draw_bases(board: Board) -> None:
    load_board()
    draw_pieces(board)
    draw_captured(board)


def draw_moves(moves: list[int]) -> None:
    # draws available moves for selected piece
    for move in moves:
        tgt = (move & 0xfc0) >>  6
        y, x = tgt // 8, tgt % 8
        center = (
            100 + (x * CELL_SIZE) + CELL_SIZE//2,
            100 + (y * CELL_SIZE) + CELL_SIZE//2
        )
        cv.circle(img, center, CELL_SIZE//8, (49, 56, 59), thickness=-1)
        
    cv.imshow(window, img)

# logic for mouse clicks
def mouse_click(event, x: int, y: int, flags, param: Board) -> None:
    global AVAILABLE_MOVES, SELECTED_MOVE, SELECTED_PIECE, human_turn, STATES

    # clinking the slider for depth
    if 1210 >= x >= 1040 and 770 <= y <= 800 and event == cv.EVENT_LBUTTONDOWN:
        ui.depth = x_to_depth(x)
        redraw(param)
        return
    
    # clicking on a piece shows its available moves
    if event == cv.EVENT_LBUTTONUP:
        # selector panel
        if x >= 925:
            y_pos = (y // 25) - 1

            # poition type
            if 1 <= y_pos <= len(ui.positions):
                ui.pos_idx = y_pos - 1
            
            # white agent
            base = 6
            if base <= y_pos < base + len(ui.agents):
                ui.white_idx = y_pos - base

            # black agent 
            base += len(ui.agents) + 1
            if base <= y_pos < base+len(ui.agents):
                ui.black_idx = y_pos - base

            redraw(param)
            return 

        # board / piece clicks
        if 100 <= x < 900 and 100 <= y < 900:
            row = (y - 100) // CELL_SIZE
            col = (x - 100) // CELL_SIZE
            idx = row * 8 + col
            board = param
            board.generate_moves()
            if board.side == white: pieces = range(0, 6)
            else: pieces = range(6, 12)
            piece = -1

            # if there is no selected piece
            if SELECTED_PIECE == -1:
                
                # check if the sqaure you clicked has a piece
                for bb in pieces: 
                    if board.bitboards[bb] & BIT[idx]:
                        # set piece
                        piece = bb
                        break
            
            if piece != -1:
                # find available moves for selected piece and add
                for mv in board.ML.moves[:board.ML.count]:
                    # same source and piece
                    if idx == (mv & 0x3f) and piece == ((mv & 0xf000) >> 12):
                        AVAILABLE_MOVES.add(mv)
                SELECTED_PIECE = piece

            # if there is a selected piece
            elif SELECTED_PIECE != -1:
                # calculate available moves
                for mv in AVAILABLE_MOVES:
                    # same target and piece
                    if idx == ((mv & 0xfc0) >> 6) and SELECTED_PIECE == ((mv & 0xf000) >> 12):
                        # promotion
                        promo = (mv & 0xf0000) >> 16
                        if promo:
                            # defult queen for now
                            if (promo != Q) if board.side == white else (promo != q):
                                continue

                        SELECTED_MOVE = mv
                        break
                
                AVAILABLE_MOVES.clear()
                SELECTED_PIECE = -1
                
                # making the move logic
                if (board.side == human_turn or human_turn == 2) and SELECTED_MOVE:
                    board.make_move(SELECTED_MOVE, all_moves)
                    STATES.append(board.copy_board())
                    SELECTED_MOVE = None
                    AVAILABLE_MOVES.clear()
                    draw_bases(board)
                    cv.imshow(window, img)

            # clear globals
            else:
                AVAILABLE_MOVES.clear()
                SELECTED_PIECE = -1

            # draw the moves on the board
            redraw(board, moves=AVAILABLE_MOVES)

# redraws all dynamic parts of the screen
def redraw(board: Board, moves=None) -> None:
    load_board()
    draw_pieces(board)
    draw_captured(board)
    if moves:
        draw_moves(moves)
    draw_selector_panel(board)
    cv.imshow(window, img)

# draw the captured piece on the top and bottom
def draw_captured(board: Board) -> None:
    # Puts the captured pices in lower boxes

    # draw the rectangles 
    cv.rectangle(img, (100,10), (900, 75), (33,33,33), thickness=-1)
    cv.rectangle(img, (100,925), (900, 990), (33,33,33), thickness=-1)
    white_pieces = []
    black_pieces = []

    # put the relaive pieces into their arrays
    for piece in board.capped_pieces:
        image = PIECE_IMAGES[piece]

        im = cv.imread(image, cv.IMREAD_UNCHANGED)

        color = white if piece < 6 else black
        if color == white:
            white_pieces.append((piece, im))
        else:
            black_pieces.append((piece, im))
    
    # sort based on piece value 
    white_pieces.sort(key=lambda x: x[0])
    black_pieces.sort(key=lambda x: x[0])

    # draw the pieces
    for i in range(len(white_pieces)):
        y = 20
        x = 110 + (25 * i)
        draw_piece(white_pieces[i][1], y, x, captured=True)

    for i in range(len(black_pieces)):
        y = 930
        x = 110 + (25 * i)
        draw_piece(black_pieces[i][1], y, x, captured=True)


# initialize the base windows
def initialize_window(board: Board) -> None:
    points = np.array([[950,  60], [950, 100], [1210, 100], [1210,  60]], np.int32)
    cv.fillPoly(img, pts=[points], color=(33,33,33))

    # instructions
    cv.rectangle(img, (925, 920), (1250,985), (55,55,55), -1)
    cv.putText(img, "Use arrow keys to move forward and back", (935, 945), cv.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1, cv.LINE_AA, False)
    cv.putText(img, "Press escape to exit", (1010, 970), cv.FONT_HERSHEY_TRIPLEX, 0.4, (0,0,0), 1, cv.LINE_AA, False)

    draw_selector_panel(board)


# draws the selector panel for different agents and options
def draw_selector_panel(board: Board) -> None:
    # setup the base
    cv.rectangle(img, (925, 10), (1250, 910), (33,33,33), -1)
    font = cv.FONT_HERSHEY_TRIPLEX
    line = cv.LINE_AA
    y = 35

    # chose position
    cv.putText(img, "Position", (935, y), font, 0.6, (200,200,200), 1, line)
    for i, type in enumerate(ui.positions):
        y += 25
        label = f"[{'x' if i==ui.pos_idx else ' '}] {type}"
        cv.putText(img, label, (945, y), font, 0.5, (0,0,0), 1, line)

    # agent choice
    y += 50
    cv.putText(img, "White Agent", (935, y), font, 0.6, (255,255,255), 1, line)
    for i, agent in enumerate(ui.agents):
        y += 25
        label = f"[{'x' if i==ui.white_idx else ' '}]  {agent}"
        cv.putText(img, label, (945, y), font, .5, (255,255,255), 1, line)

    y += 25
    cv.putText(img, "Black Agent", (935, y), font, 0.6, (0,0,0), 1, line)
    for i, agent in enumerate(ui.agents):
        y += 25
        label = f"[{'x' if i==ui.black_idx else ' '}]  {agent}"
        cv.putText(img, label, (945, y), font, .5, (0,0,0), 1, line)

    # depth slider
    y += 50
    cv.putText(img,  f"Depth: {ui.depth}", (935, y), font, .6, (200,200,200), 1, line)
    if ui.depth > 5:
        cv.putText(img,  "Might Take a While!", (935, y + 20), font, .6, (200,200,200), 1, line)

    cv.rectangle(img,
                 (1205, 770),
                 (1050,  790),
                 (70,70,70), -1)
    
    knobx = int(depth_to_x(ui.depth))
    cv.circle(img, (knobx, 780), 10, (180,180,180), -1)

    # print the moves to make if from puzzle
    if ui.msg:
        chunks = ui.msg.split()
        if len(chunks) > 5:
            first_half = " ".join(chunks[:5])
            second_half = " ".join(chunks[5:])

            cv.putText(img, f"Winning moves: {first_half}", (935, y + 40), font, .4, (200,200,200), 1, line)
            cv.putText(img, second_half , (935, y + 55), font, .4, (200,200,200), 1, line)
        else:
            cv.putText(img, f"Winning moves: {ui.msg}", (935, y + 40), font, .4, (200,200,200), 1, line)

    # print the amount of nodes visited and in what amount of time
    if ui.nodes != 0:
        color = (200,200,200) if board.side == black else (0,0,0)
        cv.putText(img, f"Traversed {ui.nodes} nodes in {ui.time} seconds", (935, y + 115), font, .4, color, 1, line)
 
    # helper messages to navigate 
    if not LOCKED:
        cv.putText(img, "Press space to lock in!", (975, 850), cv.FONT_HERSHEY_TRIPLEX, 0.5, (200,200,200), 1, cv.LINE_AA)

    if LOCKED:
        cv.putText(img, "Press r go back!", (1100, 30), cv.FONT_HERSHEY_TRIPLEX, 0.5, (200,200,200), 1, cv.LINE_AA)


# turn the depth of seach to x cord for slider
def depth_to_x(depth: int) -> int:
    return 1045 + (depth * 15)

# turn x cord to seach depth
def x_to_depth(x: int) -> int:
    d = round((x - 1050) / 15)
    return max(1, min(10, d))


# retreive the position string for a given option
def get_starting_pos(pos_type: str) -> str:
    if pos_type == "start position":
        return start_position
    elif pos_type == "random position":
        # return random midgame position
        return get_random_position()
    else:
        # return random puzzle position
        fen, msg = get_random_puzzle()
        # set msg to the right moves
        ui.msg = msg
        return fen
    

def build_agent(agent: str, color: int) -> Agent:
    # return agent and flag associated with the string 
    agents = {
        "MiniMaxV1": (MiniMaxAgentV1(color), "mm"),
        "MiniMaxV2": (MiniMaxAgentV2(color), "mm"),
        "MiniMaxV3": (MiniMaxAgentV3(color), "mm"),
        "MiniMaxV4": (MiniMaxAgentV4(color), "mm"),
        "MiniMaxV5": (MiniMaxAgentV5(color), "mm"),
        "MiniMaxV6": (MiniMaxAgentV6(color), "mm"),
        "MiniMaxV7": (MiniMaxAgentV7(color), "mm"),
        "MonteCarloV1": (MonteCarloAgentV1(color), "mc"),
        "MonteCarloV2": (MonteCarloAgentV2(color), "mc"),
        "RL": (RLAgentV2(color), "rl"),
        "Human": (0, "h")}
    
    return agents[agent]


def get_end_card(game_board: Board):
    # print the text based on the winner
    if game_board.winner == white:
        text = "White Wins"
    elif game_board.winner == black:
        text = "Black Wins"
    else:
        text = "Draw"

    # get a copy of the current screen
    overlay = img.copy()

    # calculate muffled overlay 
    cv.rectangle(overlay, (100,100), (900,900), (154,203,220), thickness=-1)
    blended = cv.addWeighted(overlay, 0.8, img, 0.2, 0)
    font = cv.FONT_HERSHEY_TRIPLEX
    font_scale = 2
    thickness_text = 2

    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness_text)

    # center font on screen
    center_x = 100 + (800 - text_width) // 2
    center_y = 100 + (800 + text_height) // 2

    # put the text on the screen 
    cv.putText(blended, text, (center_x, center_y), cv.FONT_HERSHEY_TRIPLEX, 2, (0,0,0), 2, cv.LINE_AA)
    cv.putText(blended, "press r to retry", (360, 550), font, 1, (0,0,0), 2, cv.LINE_AA)

    return blended

        

def main() -> None:
    global img, SELECTED_MOVE, human_turn, LOCKED, STATES
    all_vecs = PositionWeightVectors.load_all(model="V2")

    restart = False

    # set up original board
    board = Board()
    st = board.copy_board()

    # initialize windows
    cv.setMouseCallback('board', mouse_click, param=board)    
    initialize_window(board)
    draw_bases(board)
    cv.imshow(window, img)

    # main progam loop
    while True:

        key = cv.waitKeyEx(50)

        # escape key
        if key == 27:
            cv.destroyAllWindows()
            return

        # space key to start
        if key == ord(" "):
            LOCKED = True
            # fech the position and agents
            fen = get_starting_pos(ui.positions[ui.pos_idx])
            white_agent, wtype = build_agent(ui.agents[ui.white_idx], white)
            black_agent, btype = build_agent(ui.agents[ui.black_idx], black)

            # init board
            game_board = Board(fen=fen)

            # set turn that human can play
            if wtype == "h" and btype == "h":
                human_turn = 2
            elif wtype == "h":
                human_turn = white
            elif btype == "h":
                human_turn = black
                
            cv.setMouseCallback('board', mouse_click, param=game_board)  
            redraw(game_board)

            # game loop
            while True:
                
                # get agent and agent type
                agent, atype = (white_agent, wtype) if game_board.side == white else (black_agent, btype)
                game_key = cv.waitKeyEx(50)

                # if the game is at a terminal state
                if game_board.winner != -1:
                    break

                # right arrow, move forward
                if game_key == 2555904:
                    # time the move
                    start = time.time()

                    # search the position
                    if atype == "mm":
                        val = agent.search_position(ui.depth, game_board)
                        ui.nodes = agent.nodes
                    elif atype == "mc":
                        val = agent.search_position(game_board)
                        ui.nodes = agent.nodes
                    elif atype == "rl":
                        val = agent.search_position(ui.depth, game_board, all_vecs.weights)
                        ui.nodes = agent.nodes
                    else:
                        continue
                    end = time.time()
                    ui.time = round((end - start), 2)

                    # make the move on the board, if illegal, break out of game loop
                    if not game_board.make_move(agent.best_move, all_moves):
                        break 
                    
                    
                    # append new board state to states
                    STATES.append(game_board.copy_board())

                    # reset windows and attributes
                    redraw(game_board)
                    ui.nodes, ui.time = 0, 0

                # left arrow go back a move
                if key == 2424832:
                    # if there is no moves keep looping
                    if len(STATES) == 0:
                        continue

                    # get the most recent state and go back to it
                    last = STATES.pop()
                    game_board.restore_board(last)
                    redraw(game_board)
                
                # secape key
                if game_key == 27:
                    cv.destroyAllWindows()
                    return
                
                # r key will refresh the option 
                if game_key == ord("r"):
                    # wipe all globals clean
                    restart = True
                    STATES = []
                    LOCKED = False
                    ui.msg = None
                    game_board.restore_board(st)
                    initialize_window(board)
                    draw_bases(board)
                    cv.imshow(window, img)
                    break
                
                # check if the human just made a move that results in a terminal state
                if human_turn != -1:
                    game_board.make_move(0, 10)

            # if simply restarting jut go back to the top of the program loop
            if restart:
                restart = False

            else:
                
                # in terminal state
                while True:
                    # get the new screen image
                    blended = get_end_card(game_board)

                    cv.imshow(window, blended)

                    # press r to go back to program loop
                    key = cv.waitKeyEx(50)
                    if key == ord("r"):
                        STATES = []
                        LOCKED = False
                        ui.msg = None
                        game_board.restore_board(st)
                        initialize_window(board)
                        draw_bases(board)
                        cv.imshow(window, img)
                        break
                    
                    # escape key
                    if key == 27:
                        cv.destroyAllWindows()
                        return

main()