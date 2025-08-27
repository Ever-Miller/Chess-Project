from problems import Board, CORD_TO_SQUARE, get_bishop_attacks, get_queen_attacks, \
king_attacks, knight_attacks, get_rook_attacks, print_bitboard, get_move_capture, get_move_target, get_move_piece
import numpy as np
import random
import sys
import time
np.set_printoptions(threshold=sys.maxsize)

# POSITION MASKS FOR BOARD
LEFT = 0xe0e0e0e0e0e0e0e0
RIGHT = 0x0707070707070707
MIDDLE = 0x1818181818181818

# Enums
MAX_PLY = 64
white = 0
black = 1
both = 2
P, N, B, R, Q, K, p, n, b, r, q, k = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
all_moves, only_captures = 0, 1
hfe, hfa, hfb = 0, 1, 2
mate_score, mate_value = 98000, 99000

# gets index of LSB
LS1B_IDX = {1<<i:i for i in range(64)}
# lookup table for 1 << n patterns n = {0 - 63}
BIT = tuple(1 << sq for sq in range(64))


material_score = [
    100, # P
    300, # N
    325, # B
    500, # R
    900, # Q
  10000, # K
   -100, # p
   -300, # n
   -350, # b
   -500, # r
   -900, # q
 -10000, # k
]

# positional scores based on squares
pawn_score = [
     90,  90,  90,  90,  90,  90,  90,  90,
     40,  30,  30,  40,  40,  30,  30,  40,
     30,  20,  20,  30,  30,  30,  20,  30,
     20,  10,  10,  20,  20,  10,  10,  20,
     10,  10,   5,  20,  20,   5,  10,  10,
      5,   5,   0,   5,   5,   0,   5,   5,
      0,   0,   0, -15, -15,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0
]

knight_score = [
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   5,  10,  10,   5,   0,  -5,
     -5,   5,  20,  20,  20,  20,   5,  -5,
     -5,  25,  20,  30,  30,  20,  25,  -5,
     -5,  10,  20,  30,  30,  20,  10,  -5,
     -5,   5,  20,  10,  10,  20,   5,  -5,
     -5,   0,   0,   5,   5,   0,   0,  -5,
     -5, -15,   0,   0,   0,   0, -15,  -5
]

bishop_score = [
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,  10,  10,   0,   0,  0,
      0,  10,  10,  20,  20,  10,  10,   0,
      0,   0,  10,  20,  20,  10,   0,   0,
      0,  10,   0,   0,   0,   0,  10,   0,
      0,  30,   0,   5,   5,   0,  30,   0,
      10,  0, -15,   0,   0, -15,   0,   10
]

rook_score = [
     50,  50,  50,  50,  50,  50,  50,  50,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,  10,  20,  20,  10,   0,   0,
      0,   0,  10,  20,  20,  10,   0,   0,
      0,   0,  10,  20,  20,  10,   0,   0,
      5,   0,  10,  20,  20,  10,   0,   5,
      0,   0,  10,  20,  20,  10,   0,   0,
      0,   0,   0,  20,  20,   0,   0,   0
]

queen_score = [
     0,  0,   0,   0,   0,   0,   0,   0,
     0,  0,   5,   5,   5,   5,   0,   0,
     0, 10,   5,  10,  10,  10,  10,   0,
     0,  0,  10,  20,  20,  10,   0,   0,
     0,  0,  10,  20,  20,  10,   0,  -5,
     0, 10,   0,  20,  20,   0,  10,  -5,
     0, 30,   0,   0,   0,   0,  30, -10,
    10,  0, -10,   0,   0, -10, -10, -10
]

king_score = [
      0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   5,   5,   5,   5,   0,   0,
      0,   5,   5,  10,  10,   5,   5,   0,
      0,   5,  10,  20,  20,  10,   5,   0,
      0,   5,  10,  20,  20,  10,   5,   0,
      0,   0,   5,  10,  10,   5,   0,   0,
      0,   5,   5,  -5,  -5,   0,   5,   0,
      0,   0,   5,  -5, -15,  -5,  10,   0
]

mirror_score = [
    56, 57, 58, 59, 60, 61, 62, 63,
    48, 49, 50, 51, 52, 53, 54, 55,
    40, 41, 42, 43, 44, 45, 46, 47,
    32, 33, 34, 35, 36, 37, 38, 39,
    24, 25, 26, 27, 28, 29, 30, 31,
    16, 17, 18, 19, 20, 21, 22, 23,
     8,  9, 10, 11, 12, 13, 14, 15,
     0,  1,  2,  3,  4,  5,  6,  7 
]

# most valuable victim least valuable attacker
# https://www.open-chess.org/viewtopic.php?f=5&t=3058
# [attacker][victim]
mvv_lva = [

# attackers                Victims (p,n,b,r,q,k)
#    v                          
            [600, 2022, 2025, 2040, 2080, 2690,   600, 2022, 2025, 2040, 2080, 2690],
            [477,  600, 2002, 2017, 2057, 2667,   477,  600, 2002, 2017, 2057, 2667],
            [475,  497,  600, 2015, 2055, 2665,   475,  497,  600, 2015, 2055, 2665],
            [460,  482,  485,  600, 2040, 2650,   460,  482,  485,  600, 2040, 2650],
            [420,  442,  445,  460,  601, 2610,   420,  442,  445,  460,  601, 2610],
            [310,  332,  335,  350,  390, 2600,   310,  332,  335,  350,  390, 2600],
            
            [600, 2022, 2025, 2040, 2080, 2690,   600, 2022, 2025, 2040, 2080, 2690],
            [477,  600, 2002, 2017, 2057, 2667,   477,  600, 2002, 2017, 2057, 2667],
            [475,  497,  600, 2015, 2055, 2665,   475,  497,  600, 2015, 2055, 2665],
            [460,  482,  485,  600, 2040, 2650,   460,  482,  485,  600, 2040, 2650],
            [420,  442,  445,  460,  601, 2610,   420,  442,  445,  460,  601, 2610],
            [310,  332,  335,  350,  390, 2600,   310,  332,  335,  350,  390, 2600],
]

ranks = [
    7, 7, 7, 7, 7, 7, 7, 7,
    6, 6, 6, 6, 6, 6, 6, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0, 0, 0
]

# file/ rank for each square
file_masks = [0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080,
              0x101010101010101,
              0x202020202020202,
              0x404040404040404,
              0x808080808080808,
              0x1010101010101010,
              0x2020202020202020,
              0x4040404040404040,
              0x8080808080808080]

rank_masks = [0xff,
              0xff,
              0xff,
              0xff,
              0xff,
              0xff,
              0xff,
              0xff,
              0xff00,
              0xff00,
              0xff00,
              0xff00,
              0xff00,
              0xff00,
              0xff00,
              0xff00,
              0xff0000,
              0xff0000,
              0xff0000,
              0xff0000,
              0xff0000,
              0xff0000,
              0xff0000,
              0xff0000,
              0xff000000,
              0xff000000,
              0xff000000,
              0xff000000,
              0xff000000,
              0xff000000,
              0xff000000,
              0xff000000,
              0xff00000000,
              0xff00000000,
              0xff00000000,
              0xff00000000,
              0xff00000000,
              0xff00000000,
              0xff00000000,
              0xff00000000,
              0xff0000000000,
              0xff0000000000,
              0xff0000000000,
              0xff0000000000,
              0xff0000000000,
              0xff0000000000,
              0xff0000000000,
              0xff0000000000,
              0xff000000000000,
              0xff000000000000,
              0xff000000000000,
              0xff000000000000,
              0xff000000000000,
              0xff000000000000,
              0xff000000000000,
              0xff000000000000,
              0xff00000000000000,
              0xff00000000000000,
              0xff00000000000000,
              0xff00000000000000,
              0xff00000000000000,
              0xff00000000000000,
              0xff00000000000000,
              0xff00000000000000]

# isolated pawn pasks
isolated_pawns = [0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040,
                  0x202020202020202,
                  0x505050505050505,
                  0xa0a0a0a0a0a0a0a,
                  0x1414141414141414,
                  0x2828282828282828,
                  0x5050505050505050,
                  0xa0a0a0a0a0a0a0a0,
                  0x4040404040404040]

# passed pawn masks
passed_pawns_w = [0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x3,
                  0x7,
                  0xe,
                  0x1c,
                  0x38,
                  0x70,
                  0xe0,
                  0xc0,
                  0x303,
                  0x707,
                  0xe0e,
                  0x1c1c,
                  0x3838,
                  0x7070,
                  0xe0e0,
                  0xc0c0,
                  0x30303,
                  0x70707,
                  0xe0e0e,
                  0x1c1c1c,
                  0x383838,
                  0x707070,
                  0xe0e0e0,
                  0xc0c0c0,
                  0x3030303,
                  0x7070707,
                  0xe0e0e0e,
                  0x1c1c1c1c,
                  0x38383838,
                  0x70707070,
                  0xe0e0e0e0,
                  0xc0c0c0c0,
                  0x303030303,
                  0x707070707,
                  0xe0e0e0e0e,
                  0x1c1c1c1c1c,
                  0x3838383838,
                  0x7070707070,
                  0xe0e0e0e0e0,
                  0xc0c0c0c0c0,
                  0x30303030303,
                  0x70707070707,
                  0xe0e0e0e0e0e,
                  0x1c1c1c1c1c1c,
                  0x383838383838,
                  0x707070707070,
                  0xe0e0e0e0e0e0,
                  0xc0c0c0c0c0c0,
                  0x3030303030303,
                  0x7070707070707,
                  0xe0e0e0e0e0e0e,
                  0x1c1c1c1c1c1c1c,
                  0x38383838383838,
                  0x70707070707070,
                  0xe0e0e0e0e0e0e0,
                  0xc0c0c0c0c0c0c0]

passed_pawns_b = [0x303030303030300,
                  0x707070707070700,
                  0xe0e0e0e0e0e0e00,
                  0x1c1c1c1c1c1c1c00,
                  0x3838383838383800,
                  0x7070707070707000,
                  0xe0e0e0e0e0e0e000,
                  0xc0c0c0c0c0c0c000,
                  0x303030303030000,
                  0x707070707070000,
                  0xe0e0e0e0e0e0000,
                  0x1c1c1c1c1c1c0000,
                  0x3838383838380000,
                  0x7070707070700000,
                  0xe0e0e0e0e0e00000,
                  0xc0c0c0c0c0c00000,
                  0x303030303000000,
                  0x707070707000000,
                  0xe0e0e0e0e000000,
                  0x1c1c1c1c1c000000,
                  0x3838383838000000,
                  0x7070707070000000,
                  0xe0e0e0e0e0000000,
                  0xc0c0c0c0c0000000,
                  0x303030300000000,
                  0x707070700000000,
                  0xe0e0e0e00000000,
                  0x1c1c1c1c00000000,
                  0x3838383800000000,
                  0x7070707000000000,
                  0xe0e0e0e000000000,
                  0xc0c0c0c000000000,
                  0x303030000000000,
                  0x707070000000000,
                  0xe0e0e0000000000,
                  0x1c1c1c0000000000,
                  0x3838380000000000,
                  0x7070700000000000,
                  0xe0e0e00000000000,
                  0xc0c0c00000000000,
                  0x303000000000000,
                  0x707000000000000,
                  0xe0e000000000000,
                  0x1c1c000000000000,
                  0x3838000000000000,
                  0x7070000000000000,
                  0xe0e0000000000000,
                  0xc0c0000000000000,
                  0x300000000000000,
                  0x700000000000000,
                  0xe00000000000000,
                  0x1c00000000000000,
                  0x3800000000000000,
                  0x7000000000000000,
                  0xe000000000000000,
                  0xc000000000000000,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0,
                  0x0]

# bonus based on rank
passed_pawn_bonus = [0, 0, 5, 10, 20, 40, 100, 200]
double_pawn_penalty = -10
isolated_pawns_penalty = -10

# open files for rooks
semi_open_file_score = 5
open_file_score = 10

# how many protectors the king has
king_sheild_bonus = 5

# sets file and rank masks for each square
def set_file_and_rank(file_num: int, rank_num: int) -> int:
    mask = 0
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file

            if file_num != -1:
               if file == file_num:
                # set bit
                 mask |= (1 << square)
            elif rank_num != -1:
                if rank == rank_num:
                  # set bit
                  mask |= (1 << square)
 
    return mask

# sets relivant pawn masks
def init_eval_maps() -> int:
    # file masks
    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file

            # rank and file masks
            file_masks[square] |= set_file_and_rank(file, -1)
            rank_masks[square]  |= set_file_and_rank(-1, rank)

            # isolated pawn mask for the square
            isolated_pawns[square] |= set_file_and_rank(file + 1, -1)
            isolated_pawns[square] |= set_file_and_rank(file - 1, -1)

    for rank in range(8):
        for file in range(8):
            square = rank * 8 + file

            # passed pawn mask
            passed_pawns_w[square] |= set_file_and_rank(file + 1, -1)
            passed_pawns_w[square] |= set_file_and_rank(file, -1)
            passed_pawns_w[square] |= set_file_and_rank(file - 1, -1)

            # mask out squares behind pawn
            for i in range(8 - rank):
                passed_pawns_w[square] &= ~rank_masks[(7-i) * 8 + file]

            passed_pawns_b[square] |= set_file_and_rank(file + 1, -1)
            passed_pawns_b[square] |= set_file_and_rank(file, -1)
            passed_pawns_b[square] |= set_file_and_rank(file - 1, -1)

            for i in range(rank + 1):
                passed_pawns_b[square] &= ~rank_masks[i * 8 + file]


# genaric fucntions for agents
class Agent:
    def __init__(self, color: int):
        self.color = color
        self.ply = 0  # move count 1 for each sides move
        self.nodes = 0  # nodes traversed during search
        self.best_move = 0  # agents selected move

    def sort_moves(self, board: Board) -> list[int]:
        # sorts the available moves based on the score_move function given by the agent
        return sorted(board.ML.moves[:board.ML.count], key=lambda x: self.score_move(x, board), reverse=True)


# Basic negamax agent - no optimizations
class MiniMaxAgentV1(Agent):
    def __init__(self, color: int):
        super().__init__(color)

    # Heuristic function for the cur state
    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if   piece == P: score += pawn_score[square]
                elif piece == N: score += knight_score[square]
                elif piece == B: score += bishop_score[square]
                elif piece == R: score += rook_score[square]
                elif piece == Q: score += queen_score[square]
                elif piece == K: score += king_score[square]

                # positional scores for black pieces
                if   piece == p: score -= pawn_score[mirror_score[square]]
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: score -= bishop_score[mirror_score[square]]
                elif piece == r: score -= rook_score[mirror_score[square]]
                elif piece == q: score -= queen_score[mirror_score[square]]
                elif piece == k: score -= king_score[mirror_score[square]]

                # pop bit
                bitboard ^= lsb

        # white side will be positive for good vals and black will be negative for good vals
        return score if board.side == white else -score

    def score_move(self, move: int, board: Board) -> int:
        # score capture moves only
        if get_move_capture(move):
            # mvv lva 
            target = P
            move_target = get_move_target(move)

            # range in enemy pieces
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break
            
            # return value in mvv_lva table
            return mvv_lva[get_move_piece(move)][target]
            
        return 0
    
    # main move finding function
    def search_position(self, depth: int, board: Board) -> int:
        self.ply = 0
        self.nodes = 0
        self.best_move = 0
        # score will be negamax with alpha = -100000, beta = 100000
        score = self.negamax(-100000, 100000, depth, board)
        return score


    # https://www.chessprogramming.org/Alpha-Beta
    def negamax(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        
        # if the board is a repetion or move count based tie
        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # base case for recursion
        if depth == 0:
            return self.evaluate(board)
        
        # incr nodes
        self.nodes += 1

        # if the king is in check for this call, increase the depth
        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        # hold onto alpha value to check if we found a move
        cur_best = 0
        old_alpha = alpha
        legal_moves = 0

        # generate all moves for board, stored in board.ML object
        board.generate_moves()

        # loop over all available moves
        for mv in board.ML.moves[:board.ML.count]:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            # increase location in repitition table
            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                # decriment ply and rep index
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse negamax style 
            score = -self.negamax(-beta, -alpha, depth - 1, board)

            # restore the old board
            board.restore_board(st)

            # decriment ply and repetition index back to normal
            self.ply -= 1
            board.repetition_index -= 1

            # fail hard
            if score >= beta:
                return beta
            
            # better move, alpha acts like max move in negamax
            if score > alpha:
                alpha = score
                if self.ply == 0:
                    cur_best = mv

            # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -98000 + self.ply
            else:
                # stalemate score
                return 0
        
        # if there was a better move found
        if old_alpha != alpha:
            self.best_move = cur_best
        
        # else return last score
        return score
    



# Basic negamax agent + quiescent search
class MiniMaxAgentV2(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)

    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   score += pawn_score[square]
                elif piece == N: score += knight_score[square]
                elif piece == B: score += bishop_score[square]
                elif piece == R: score += rook_score[square]
                elif piece == Q: score += queen_score[square]
                elif piece == K: score += king_score[square]

                # positional scores for black pieces
                if piece == p:   score -= pawn_score[mirror_score[square]]
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: score -= bishop_score[mirror_score[square]]
                elif piece == r: score -= rook_score[mirror_score[square]]
                elif piece == q: score -= queen_score[mirror_score[square]]
                elif piece == k: score -= king_score[mirror_score[square]]

                # pop bit
                bitboard ^= lsb

        return score if board.side == white else -score

    def score_move(self, move: int, board: Board) -> int:
        # capture moves
        if get_move_capture(move):
            # mvv lva 
            target = P
            move_target = get_move_target(move)
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[get_move_piece(move)][target] + 10000
            
        return 0
    

    def search_position(self, depth: int, board: Board) -> int:
        self.ply = 0
        self.nodes = 0
        self.best_move = 0
        score = self.negamax_with_quiescence(-100000, 100000, depth, board)
        return score

    
    # nagamax search with quiescent search addeed
    def negamax_with_quiescence(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        
        # repetition or move based tie
        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # base case, switch to quiescent search (only captures)
        if depth == 0:
            return self.quiescence(alpha, beta, board)
        
        # incr nodes
        self.nodes += 1

        # give an extra depth of search if king is in check
        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        # hold on to best values
        cur_best = 0
        old_alpha = alpha
        legal_moves = 0

        # generate moves for current state
        board.generate_moves()

        # loop over all available moves
        for mv in board.ML.moves[:board.ML.count]:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            score = -self.negamax_with_quiescence(-beta, -alpha, depth - 1, board)

            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1

            if score >= beta:
                return beta
            
            if score > alpha:
                alpha = score
                if self.ply == 0:
                    cur_best = mv

            # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -98000 + self.ply
            else:
                # stalemate score
                return 0
            
        if old_alpha != alpha:
            self.best_move = cur_best
        
        return alpha
    

    # https://www.chessprogramming.org/Quiescence_Search
    # negamax search but only with captures
    def quiescence(self, alpha: int, beta: int, board: Board) -> int:
        self.nodes += 1

        score = self.evaluate(board)

        # fail hard
        if score >= beta:
            return beta
        
        # set new alpha if possible
        if score > alpha:
            alpha = score

        # genberate board moves
        board.generate_moves()

        # loop over all available moves
        for mv in board.ML.moves[:board.ML.count]:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence(-beta, -alpha, board)
            board.restore_board(st)

            self.ply -= 1
            board.repetition_index -= 1

            # prune tree
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha
    



# Basic negamax agent + quiescent search + move ordering
class MiniMaxAgentV3(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)

    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   score += pawn_score[square]
                elif piece == N: score += knight_score[square]
                elif piece == B: score += bishop_score[square]
                elif piece == R: score += rook_score[square]
                elif piece == Q: score += queen_score[square]
                elif piece == K: score += king_score[square]

                # positional scores for black pieces
                if piece == p:   score -= pawn_score[mirror_score[square]]
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: score -= bishop_score[mirror_score[square]]
                elif piece == r: score -= rook_score[mirror_score[square]]
                elif piece == q: score -= queen_score[mirror_score[square]]
                elif piece == k: score -= king_score[mirror_score[square]]

                # pop bit
                bitboard ^= lsb

        return score if board.side == white else -score

    def score_move(self, move: int, board: Board) -> int:
        # capture moves
        if get_move_capture(move):
            # mvv lva 
            target = P
            move_target = get_move_target(move)
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[get_move_piece(move)][target]
            
        return 0
    

    def search_position(self, depth: int, board: Board) -> int:
        self.ply = 0
        self.nodes = 0
        self.best_move = 0
        score = self.negamax_with_move_ordering(-100000, 100000, depth, board)
        return score

    # same as algorithms above, but order movees based on mvv_lva table
    def negamax_with_move_ordering(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0
        
        # base case
        if depth == 0:
            # run quiescent search
            return self.quiescence_with_move_ordering(alpha, beta, board)

        self.nodes += 1

        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        cur_best = 0
        old_alpha = alpha
        legal_moves = 0

        board.generate_moves()

        # sort moves
        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse
            score = -self.negamax_with_move_ordering(-beta, -alpha, depth - 1, board)
            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1

            # fail hard
            if score >= beta:
                return beta
            
            # found better move
            if score > alpha:
                alpha = score
                if self.ply == 0:
                    cur_best = mv
        
        # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -98000 + self.ply
            else:
                # stalemate score
                return 0
            
        if old_alpha != alpha:
            self.best_move = cur_best

        return alpha
    

    def quiescence_with_move_ordering(self, alpha: int, beta: int, board: Board) -> int:
        self.nodes += 1

        score = self.evaluate(board)

        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence_with_move_ordering(-beta, -alpha, board)

            self.ply -= 1
            board.repetition_index -= 1

            board.restore_board(st)

            # fail hard
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha
    


# Basic negamax agent + quiescent search + move ordering + killer move detection
class MiniMaxAgentV4(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)

    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   score += pawn_score[square]
                elif piece == N: score += knight_score[square]
                elif piece == B: score += bishop_score[square]
                elif piece == R: score += rook_score[square]
                elif piece == Q: score += queen_score[square]
                elif piece == K: score += king_score[square]

                # positional scores for black pieces
                if piece == p:   score -= pawn_score[mirror_score[square]]
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: score -= bishop_score[mirror_score[square]]
                elif piece == r: score -= rook_score[mirror_score[square]]
                elif piece == q: score -= queen_score[mirror_score[square]]
                elif piece == k: score -= king_score[mirror_score[square]]

                # pop bit
                bitboard ^= lsb

        return score if board.side == white else -score


    def score_move(self, move: int, board: Board) -> int:
        # capture moves
        if get_move_capture(move):
            # mvv lva 
            target = P
            move_target = get_move_target(move)
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[get_move_piece(move)][target] + 10000
        else:
            # killer move heuristic (moves that were also good in last position or search)
            # info from https://rustic-chess.org/search/ordering/killers.html
            if board.killer_moves[0][self.ply] == move:
                return 9000
            
            elif board.killer_moves[1][self.ply] == move:
                return 8000
            
            else:
                # moves that caused a cutoff in past searches, the base value will be 0
                # https://www.chessprogramming.org/History_Heuristic
                return board.history_moves[get_move_piece(move)][get_move_target(move)]
            

    

    def search_position(self, depth: int, board: Board) -> int:
        self.ply = 0
        self.nodes = 0
        self.best_move = 0
        score = self.negamax_with_move_ordering(-100000, 100000, depth, board)
        return score

        
    def negamax_with_move_ordering(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # base case
        if depth == 0:
            # run quiescent search
            return self.quiescence_with_move_ordering(alpha, beta, board)

        self.nodes += 1

        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        cur_best = 0
        old_alpha = alpha
        legal_moves = 0

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse
            score = -self.negamax_with_move_ordering(-beta, -alpha, depth - 1, board)
            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1

            # fail hard, store killer mvoes
            if score >= beta:
                # only quiet moves
                if not get_move_capture(mv):
                    # store killer moves
                    board.killer_moves[1][self.ply] = board.killer_moves[0][self.ply]
                    board.killer_moves[0][self.ply] = mv
                return beta
            
            # found better move
            if score > alpha:
                # only quiet moves
                if not get_move_capture(mv):
                    # store history
                    board.history_moves[get_move_piece(mv)][get_move_target(mv)] += depth
                alpha = score
                if self.ply == 0:
                    cur_best = mv
        
        # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -98000 + self.ply
            else:
                # stalemate score
                return 0
            
        if old_alpha != alpha:
            self.best_move = cur_best

        return alpha
    

    def quiescence_with_move_ordering(self, alpha: int, beta: int, board: Board) -> int:
        self.nodes += 1

        score = self.evaluate(board)

        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence_with_move_ordering(-beta, -alpha, board)

            self.ply -= 1
            board.repetition_index -= 1

            board.restore_board(st)

            # fail hard
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha
    

# Iterative deepening negamax agent + quiescent search + move ordering + killer move detection + PV
# Quadratic pv table implemetation from https://sites.google.com/site/tscpchess/principal-variation
class MiniMaxAgentV5(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)
        
        # follow the pv moves and score of the pv moves
        self.score_pv = 0
        self.follow_pv = 0

    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   score += pawn_score[square]
                elif piece == N: score += knight_score[square]
                elif piece == B: score += bishop_score[square]
                elif piece == R: score += rook_score[square]
                elif piece == Q: score += queen_score[square]
                elif piece == K: score += king_score[square]

                # positional scores for black pieces
                if piece == p:   score -= pawn_score[mirror_score[square]]
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: score -= bishop_score[mirror_score[square]]
                elif piece == r: score -= rook_score[mirror_score[square]]
                elif piece == q: score -= queen_score[mirror_score[square]]
                elif piece == k: score -= king_score[mirror_score[square]]

                # pop bit
                bitboard ^= lsb

        return score if board.side == white else -score

    def score_move(self, move: int, board: Board) -> int:
        # if PV scoring
        if self.score_pv:
            # if move in PV
            if board.pv_table[0][self.ply] == move:
                # disable scorepv
                self.score_pv = 0 
                # give PV move the highest score
                return 50000

        # capture moves
        if (move >> 20) & 1:
            # mvv lva 
            target = P if board.side == black else p
            move_target = (move & 0xfc0) >> 6
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[(move & 0xf000) >> 12][target] + 10000
        else:
            # killer move
            if board.killer_moves[0][self.ply] == move:
                return 9000
            
            elif board.killer_moves[1][self.ply] == move:
                return 8000
            
            else:
                return board.history_moves[(move & 0xf000) >> 12][(move & 0xfc0) >> 6]

    # enable a move to be scored by principle variation
    def enable_pv_scoring(self, board):
        self.follow_pv = 0

        for mv in board.ML.moves[:board.ML.count]:
            # hit pv move
            if board.pv_table[0][self.ply] == mv:
                # enable scoring and following
                self.score_pv = self.follow_pv = 1
    
    # resets tables and flags
    def clear_vars(self, board):
        self.ply = self.nodes = self.best_move = self.follow_pv = self.score_pv = 0
        board.killer_moves = [[0] * MAX_PLY for _ in range(2)]
        board.history_moves = [[0] * 64 for _ in range(12)]
        board.pv_len = [0] * MAX_PLY
        board.pv_table = [[0] * MAX_PLY for _ in range(MAX_PLY)]
    

    def search_position(self, depth: int, board: Board) -> int:
        # reset all vars
        self.clear_vars(board)
        

        # iterative deepening 
        for d in range(1, depth + 1):
            self.nodes = 0

            # enable follow pv
            self.follow_pv = 1
            score = self.negamax_with_move_ordering(-100000, 100000, d, board)
            # best move will be the first value in the pv table
            self.best_move = board.pv_table[0][0]
        
        return score
        
    def negamax_with_move_ordering(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        self.nodes
        # init PV len
        board.pv_len[self.ply] = self.ply 

        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # base case
        if depth == 0:
            # run quiescent search
            return self.quiescence_with_move_ordering(alpha, beta, board)
        
        # too deep of a search
        if self.ply >= MAX_PLY:
            return self.evaluate(board)

        self.nodes += 1

        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        legal_moves = 0

        board.generate_moves()
        
        # following PV line
        if self.follow_pv:
            self.enable_pv_scoring(board)

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse
            score = -self.negamax_with_move_ordering(-beta, -alpha, depth - 1, board)
            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1

            # fail hard, store killer mvoes
            if score >= beta:
                # only quiet moves
                if not (mv >> 20) & 1:
                    # store killer moves
                    board.killer_moves[1][self.ply] = board.killer_moves[0][self.ply]
                    board.killer_moves[0][self.ply] = mv
                return beta
            
            # found better move
            if score > alpha:
                # only quiet moves
                if not (mv >> 20) & 1:
                    # store history
                    board.history_moves[(mv & 0xf000) >> 12][(mv & 0xfc0) >> 6] += depth

                alpha = score

                # write PV move
                board.pv_table[self.ply][self.ply] = mv

                # loop through all moves of pv
                for next_ply in range(self.ply + 1, board.pv_len[self.ply + 1]):
                    # copy move from deeper ply
                    board.pv_table[self.ply][next_ply] = board.pv_table[self.ply + 1][next_ply]

                # adjust pv length
                board.pv_len[self.ply] = board.pv_len[self.ply + 1]

        # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -98000 + self.ply
            else:
                # stalemate score
                return 0

        return alpha
    

    def quiescence_with_move_ordering(self, alpha: int, beta: int, board: Board) -> int:
        self.nodes += 1

        score = self.evaluate(board)

        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence_with_move_ordering(-beta, -alpha, board)

            self.ply -= 1
            board.repetition_index -= 1

            board.restore_board(st)

            # fail hard
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha
    

# Iterative deepening negamax agent + quiescent search + move ordering + killer move detection + PV search
# zobrist hashing
class MiniMaxAgentV6(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)
        
        # follow the pv moves and score of the pv moves
        self.score_pv = 0
        self.follow_pv = 0

    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   score += pawn_score[square]
                elif piece == N: score += knight_score[square]
                elif piece == B: score += bishop_score[square]
                elif piece == R: score += rook_score[square]
                elif piece == Q: score += queen_score[square]
                elif piece == K: score += king_score[square]

                # positional scores for black pieces
                if piece == p:   score -= pawn_score[mirror_score[square]]
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: score -= bishop_score[mirror_score[square]]
                elif piece == r: score -= rook_score[mirror_score[square]]
                elif piece == q: score -= queen_score[mirror_score[square]]
                elif piece == k: score -= king_score[mirror_score[square]]

                # pop bit
                bitboard ^= lsb

        return score if board.side == white else -score

    def score_move(self, move: int, board: Board) -> int:
        # if PV scoring
        if self.score_pv:
            # if move in PV
            if board.pv_table[0][self.ply] == move:
                # disable scorepv
                self.score_pv = 0 
                # give PV move the highest score
                return 50000

        # capture moves
        if (move >> 20) & 1:
            # mvv lva 
            target = P if board.side == black else p
            move_target = (move & 0xfc0) >> 6
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[(move & 0xf000) >> 12][target] + 10000
        else:
            # killer move
            if board.killer_moves[0][self.ply] == move:
                return 9000
            
            elif board.killer_moves[1][self.ply] == move:
                return 8000
            
            else:
                return board.history_moves[(move & 0xf000) >> 12][(move & 0xfc0) >> 6]

    
    def enable_pv_scoring(self, board: Board) -> None:
        self.follow_pv = 0

        for mv in board.ML.moves[:board.ML.count]:
            # hit pv move
            if board.pv_table[0][self.ply] == mv:
                # enable scoring and following
                self.score_pv = self.follow_pv = 1

    def clear_vars(self, board: Board) -> None:
        self.ply = self.nodes = self.best_move = self.follow_pv = self.score_pv = 0
        board.killer_moves = [[0] * MAX_PLY for _ in range(2)]
        board.history_moves = [[0] * 64 for _ in range(12)]
        board.pv_len = [0] * MAX_PLY
        board.pv_table = [[0] * MAX_PLY for _ in range(MAX_PLY)]
    

    def search_position(self, depth: int, board: Board) -> int:
        # reset all vars
        self.clear_vars(board)
        

        # iterative deepening 
        for d in range(1, depth + 1):
            self.nodes = 0

            # enable follow pv
            self.follow_pv = 1
            score = self.negamax_with_move_ordering(-100000, 100000, d, board)
            self.best_move = board.pv_table[0][0]

        return score
      
    # https://web.archive.org/web/20071031100051/http://www.brucemo.com/compchess/programming/hashing.htm
    def negamax_with_move_ordering(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        score = 0
        # hash flag = hash flag alpha
        hashf = hfa

        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # read hash table
        val = board.read_trans_table(alpha, beta, depth, self.ply)
        # if we have seen the move before
        if self.ply and val != 500000:
            return val

        # init PV len
        board.pv_len[self.ply] = self.ply 

        # base case
        if depth == 0:
            # run quiescent search
            return self.quiescence_with_move_ordering(alpha, beta, board)
        
        # too deep of a search
        if self.ply >= MAX_PLY:
            return self.evaluate(board)

        self.nodes += 1

        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        legal_moves = 0

        board.generate_moves()
        
        # following PV line
        if self.follow_pv:
            self.enable_pv_scoring(board)

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse
            score = -self.negamax_with_move_ordering(-beta, -alpha, depth - 1, board)
            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1
            
            # found better move
            if score > alpha:
                # update hash flag
                hashf = hfe

                # only quiet moves
                if not (mv >> 20) & 1:
                    # store history
                    board.history_moves[(mv & 0xf000) >> 12][(mv & 0xfc0) >> 6] += depth

                alpha = score

                # write PV move
                board.pv_table[self.ply][self.ply] = mv

                for next_ply in range(self.ply + 1, board.pv_len[self.ply + 1]):
                    # copy move from deeper ply
                    board.pv_table[self.ply][next_ply] = board.pv_table[self.ply + 1][next_ply]

                # adjust pv length
                board.pv_len[self.ply] = board.pv_len[self.ply + 1]

                # fail hard, store killer mvoes
                if score >= beta:
                    # store position
                    board.write_trans_table(depth, beta, hfb, self.ply)
                    # only quiet moves
                    if not (mv >> 20) & 1:
                        # store killer moves
                        board.killer_moves[1][self.ply] = board.killer_moves[0][self.ply]
                        board.killer_moves[0][self.ply] = mv
                    return beta

        # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -mate_value + self.ply
            else:
                # stalemate score
                return 0
        
        # write move to hash table
        board.write_trans_table(depth, alpha, hashf, self.ply)

        return alpha
    

    def quiescence_with_move_ordering(self, alpha: int, beta: int, board: Board) -> int:
        self.nodes += 1

        score = self.evaluate(board)

        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence_with_move_ordering(-beta, -alpha, board)

            self.ply -= 1
            board.repetition_index -= 1

            board.restore_board(st)

            # fail hard
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha
    

# Iterative deepening negamax agent + quiescent search + move ordering + killer move detection + PV search +
# zobrist hashing + new evaluation heuristics
class MiniMaxAgentV7(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)
        
        # follow the pv moves and score of the pv moves
        self.score_pv = 0
        self.follow_pv = 0

    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   
                    score += pawn_score[square]
                    score += double_pawn_penalty * (bin(board.bitboards[P] & file_masks[square]).count("1") - 1)
                    score += isolated_pawns_penalty if (board.bitboards[P] & isolated_pawns[square]) == 0 else 0
                    score += passed_pawn_bonus[ranks[square]] if (board.bitboards[p] & passed_pawns_w[square]) == 0 else 0
                elif piece == N: score += knight_score[square]
                elif piece == B: 
                    score += bishop_score[square]
                    score += bin(get_bishop_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == R: 
                    score += rook_score[square]
                    score += semi_open_file_score if (board.bitboards[P] & file_masks[square]) == 0 else 0
                    score += open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                elif piece == Q: 
                    score += queen_score[square]
                    score += bin(get_queen_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == K: 
                    score += king_score[square]
                    score -= semi_open_file_score if (board.bitboards[P] & file_masks[square]) == 0 else 0
                    score -= open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                    score += king_sheild_bonus * bin(king_attacks[square] & board.occupancies[white]).count('1')

                # positional scores for black pieces
                if piece == p:
                    score -= pawn_score[mirror_score[square]]
                    score += -double_pawn_penalty * (bin(board.bitboards[p] & file_masks[square]).count("1") - 1)
                    score += -isolated_pawns_penalty if (board.bitboards[p] & isolated_pawns[square]) == 0 else 0
                    score -= passed_pawn_bonus[ranks[mirror_score[square]]] if (board.bitboards[P] & passed_pawns_b[square]) == 0 else 0
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: 
                    score -= bishop_score[mirror_score[square]]
                    score -= bin(get_bishop_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == r: 
                    score -= rook_score[mirror_score[square]]
                    score -= semi_open_file_score if (board.bitboards[p] & file_masks[square]) == 0 else 0
                    score -= open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                elif piece == q: 
                    score -= queen_score[mirror_score[square]]
                    score -= bin(get_queen_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == k: 
                    score -= king_score[mirror_score[square]]
                    score += semi_open_file_score if (board.bitboards[p] & file_masks[square]) == 0 else 0
                    score += open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                    score -= king_sheild_bonus * bin(king_attacks[square] & board.occupancies[black]).count('1')

                # pop bit
                bitboard ^= lsb

        return score if board.side == white else -score

    def score_move(self, move: int, board: Board) -> int:
        # if PV scoring
        if self.score_pv:
            # if move in PV
            if board.pv_table[0][self.ply] == move:
                # disable scorepv
                self.score_pv = 0 
                # give PV move the highest score
                return 50000

        # capture moves
        if (move >> 20) & 1:
            # mvv lva 
            target = P if board.side == black else p
            move_target = (move & 0xfc0) >> 6
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[(move & 0xf000) >> 12][target] + 10000
        else:
            # killer move
            if board.killer_moves[0][self.ply] == move:
                return 9000
            
            elif board.killer_moves[1][self.ply] == move:
                return 8000
            
            else:
                return board.history_moves[(move & 0xf000) >> 12][(move & 0xfc0) >> 6]
            
    
    def enable_pv_scoring(self, board):
        self.follow_pv = 0

        for mv in board.ML.moves[:board.ML.count]:
            # hit pv move
            if board.pv_table[0][self.ply] == mv:
                # enable scoring and following
                self.score_pv = self.follow_pv = 1

    def clear_vars(self, board):
        self.ply = self.nodes = self.best_move = self.follow_pv = self.score_pv = 0
        board.killer_moves = [[0] * MAX_PLY for _ in range(2)]
        board.history_moves = [[0] * 64 for _ in range(12)]
        board.pv_len = [0] * MAX_PLY
        board.pv_table = [[0] * MAX_PLY for _ in range(MAX_PLY)]
    

    def search_position(self, depth: int, board: Board) -> int:
        # reset all vars
        self.clear_vars(board)
        
        # iterative deepening 
        for d in range(1, depth + 1):
            self.nodes = 0

            # enable follow pv
            self.follow_pv = 1
            score = self.negamax_with_move_ordering(-100000, 100000, d, board)
            self.best_move = board.pv_table[0][0]

        return score
      
    # https://web.archive.org/web/20071031100051/http://www.brucemo.com/compchess/programming/hashing.htm
    def negamax_with_move_ordering(self, alpha: int, beta: int, depth: int, board: Board) -> int:
        # hash flag = hash flag alpha
        hashf = hfa

        # rule / repetiton based tie
        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # read hash table
        val = board.read_trans_table(alpha, beta, depth, self.ply)
        # if we have seen the move before
        if self.ply and val != 500000:
            return val

        # init PV len
        board.pv_len[self.ply] = self.ply 

        # base case
        if depth == 0:
            # run quiescent search
            return self.quiescence_with_move_ordering(alpha, beta, board)
        
        # too deep of a search
        if self.ply >= MAX_PLY:
            return self.evaluate(board)

        self.nodes += 1

        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        legal_moves = 0

        board.generate_moves()
        
        # following PV line
        if self.follow_pv:
            self.enable_pv_scoring(board)

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse
            score = -self.negamax_with_move_ordering(-beta, -alpha, depth - 1, board)
            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1
            
            # found better move
            if score > alpha:
                # update hash flag
                hashf = hfe

                # only quiet moves
                if not (mv >> 20) & 1:
                    # store history
                    board.history_moves[(mv & 0xf000) >> 12][(mv & 0xfc0) >> 6] += depth

                alpha = score

                # write PV move
                board.pv_table[self.ply][self.ply] = mv

                for next_ply in range(self.ply + 1, board.pv_len[self.ply + 1]):
                    # copy move from deeper ply
                    board.pv_table[self.ply][next_ply] = board.pv_table[self.ply + 1][next_ply]

                # adjust pv length
                board.pv_len[self.ply] = board.pv_len[self.ply + 1]

                # fail hard, store killer mvoes
                if score >= beta:
                    # store position
                    board.write_trans_table(depth, beta, hfb, self.ply)
                    # only quiet moves
                    if not (mv >> 20) & 1:
                        # store killer moves
                        board.killer_moves[1][self.ply] = board.killer_moves[0][self.ply]
                        board.killer_moves[0][self.ply] = mv
                    return beta

        # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -mate_value + self.ply
            else:
                # stalemate score
                return 0
            
        board.write_trans_table(depth, alpha, hashf, self.ply)

        return alpha
    

    def quiescence_with_move_ordering(self, alpha: int, beta: int, board: Board) -> int:
        self.nodes += 1

        score = self.evaluate(board)

        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence_with_move_ordering(-beta, -alpha, board)

            self.ply -= 1
            board.repetition_index -= 1

            board.restore_board(st)

            # fail hard
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha


# help from https://www.youtube.com/watch?v=UXW2yZndl7U
class MCNode:
    def __init__(self, state, move):
        self.state = state
        self.side = state[2]
        self.move = move
        self.children = set()
        self.parent = None
        self.N = 0
        self.n = 0
        self.t = 0

    
class MonteCarloAgentV1(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)
        

    def search_position(self, board: Board) -> None:
        board.generate_moves()
        st = board.copy_board()
        root = MCNode(st, 0)
        self.best_move = self.mcts_pred(root)

    def mcts_pred(self, cur_node: MCNode, iter=100) -> int:

        temp = Board()
        temp.restore_board(cur_node.state)
        temp.generate_moves()
        moves = temp.ML.moves[:temp.ML.count]

        # add all children to current state
        for move in moves:
            st = temp.copy_board()
            if not temp.make_move_montey(move, all_moves):
                continue
            if temp.isCM(self.color ^ 1):
                return move
            child = MCNode(temp.copy_board(), move)
            child.parent = cur_node
            cur_node.children.add(child)
            temp.restore_board(st)

        while iter > 0:
            if not cur_node.children:
                return 0
            # get child with max ucb value
            child = max(cur_node.children, key=lambda x: self.ucb(x))

            # child hasnt been visited
            if child.n == 0:
                leaf, val = self.rollout(child)
                self.rollback(leaf, val)
            else:
                new = self.expand(child)
                leaf, val = self.rollout(new)
                self.rollback(leaf, val)

            iter -= 1
        
        if not cur_node.children:
            return 0
        # get child with max ucb value
        child = max(cur_node.children, key=lambda x: self.ucb(x))

        return child.move
    
    def ucb(self, cur_node: MCNode) -> int:
        if cur_node.n == 0:
            return np.inf
        
        v = cur_node.t / cur_node.n
        return v + 1.41 * (np.sqrt(np.log(cur_node.parent.n + 0.01) / cur_node.n))
    
    def expand(self, cur_node: MCNode) -> MCNode:
        if cur_node.state[9] != -1:
            return cur_node
        
        if len(cur_node.children) == 0:
            board = Board()
            board.restore_board(cur_node.state)
            board.generate_moves()
            moves = board.ML.moves[:board.ML.count]

            # add children
            for m in moves:
                st = board.copy_board()
                if not board.make_move_montey(m, all_moves):
                    continue
                child = MCNode(board.copy_board(), m)
                child.parent = cur_node
                cur_node.children.add(child)
                board.restore_board(st)

        if not cur_node.children:
            return cur_node
        
        # get child with max ucb value
        child = max(cur_node.children, key=lambda x: self.ucb(x))
        
        return child
    
    def rollout(self, cur_node: MCNode) -> tuple[MCNode, int]:
        board = Board()
        board.restore_board(cur_node.state)
        depth = 0

        while True:
            # if there is a terminal state
            if board.winner != -1:
                if board.winner == self.color:
                    return cur_node, 1
                elif board.winner == self.color ^ 1:
                    return cur_node, -1
                else:
                    return cur_node, 0
                
            if depth >= 50:
                return cur_node, 0
            
            # generate moves from this node
            board.generate_moves()
            moves = board.ML.moves[:board.ML.count]

            if not moves:
                return cur_node, 0

            while moves:
                move = random.choice(moves)
                if board.make_move_montey(move, all_moves):
                    break
                moves.remove(move)
            
            self.nodes += 1
            depth += 1


    def rollback(self, cur_node: MCNode, reward: int) -> None:
        # backprogagate values up the tree
        while cur_node!= None:
            cur_node.n += 1
            cur_node.t += reward
            reward = -reward
            cur_node = cur_node.parent


class MonteCarloAgentV2(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)
        

    def search_position(self, board: Board) -> None:
        board.generate_moves()
        if board.winner != -1:
            return 0
        st = board.copy_board()
        root = MCNode(st, 0)
        self.best_move = self.mcts_pred(root)

    def mcts_pred(self, cur_node: MCNode, iter=100) -> int:

        temp = Board()
        temp.restore_board(cur_node.state)
        temp.generate_moves()
        moves = temp.ML.moves[:temp.ML.count]

        # add all children to current state
        for move in moves:
            st = temp.copy_board()
            if not temp.make_move_montey(move, all_moves):
                continue
            if temp.isCM(self.color ^ 1):
                return move
            child = MCNode(temp.copy_board(), move)
            child.parent = cur_node
            cur_node.children.add(child)
            temp.restore_board(st)

        while iter > 0:
            if not cur_node.children:
                return
            # get child with max ucb value
            child = max(cur_node.children, key=lambda x: self.ucb(x))

            # child hasnt been visited
            if child.n == 0:
                leaf, val = self.rollout(child)
                self.rollback(leaf, val)
            else:
                new = self.expand(child)
                leaf, val = self.rollout(new)
                self.rollback(leaf, val)

            iter -= 1
                
        # get child with max ucb value
        child = max(cur_node.children, key=lambda x: self.ucb(x))

        return child.move
    
    def ucb(self, cur_node: MCNode) -> int:
        if cur_node.n == 0:
            return np.inf
        
        v = cur_node.t / cur_node.n
        return v + 1.41 * (np.sqrt(np.log(cur_node.parent.n + np.e) / cur_node.n))
    
    def expand(self, cur_node: MCNode) -> MCNode:
        if cur_node.state[9] != -1:
            return cur_node
        
        if len(cur_node.children) == 0:
            board = Board()
            board.restore_board(cur_node.state)
            board.generate_moves()
            moves = board.ML.moves[:board.ML.count]

            # add children
            for m in moves:
                st = board.copy_board()
                if not board.make_move_montey(m, all_moves):
                    continue
                child = MCNode(board.copy_board(), m)
                child.parent = cur_node
                cur_node.children.add(child)
                board.restore_board(st)
        
        if not cur_node.children:
            return cur_node
        
        # get child with max ucb value
        child = max(cur_node.children, key=lambda x: self.ucb(x))
        
        return child
    
    def rollout(self, cur_node: MCNode) -> tuple[MCNode, int]:
        board = Board()
        board.restore_board(cur_node.state)
        depth = 0
        while True:
            # if there is a terminal state
            if board.winner != -1:
                if board.winner == self.color:
                    return cur_node, 100000
                elif board.winner == self.color ^ 1:
                    return cur_node, -100000
                else:
                    return cur_node, 0
                
            if depth >= 50:
                return cur_node, self.evaluate(board)
            
            # generate moves from this node
            board.generate_moves()
            moves = board.ML.moves[:board.ML.count]

            if not moves:
                return cur_node, 0

            while moves:
                move = random.choice(moves)
                if board.make_move_montey(move, all_moves):
                    break
                moves.remove(move)
            
            self.nodes += 1
            depth += 1


    def rollback(self, cur_node: MCNode, reward: int) -> MCNode:
        # backprogagate values up the tree
        while cur_node!= None:
            cur_node.n += 1
            cur_node.t += reward
            cur_node = cur_node.parent


    def evaluate(self, board: Board) -> int:
        score = 0

        # loop over piece bitboards
        for bb in range(0, 12):
            bitboard = board.bitboards[bb]

            # loop over all pieces
            while bitboard:
                piece = bb

                # get index
                lsb = bitboard & -bitboard
                square = LS1B_IDX[lsb]

                # score weights
                score += material_score[piece]

                # positional scores for white pieces
                if piece == P:   
                    score += pawn_score[square]
                    score += double_pawn_penalty * (bin(board.bitboards[P] & file_masks[square]).count("1") - 1)
                    score += isolated_pawns_penalty if (board.bitboards[P] & isolated_pawns[square]) == 0 else 0
                    score += passed_pawn_bonus[ranks[square]] if (board.bitboards[p] & passed_pawns_w[square]) == 0 else 0
                elif piece == N: score += knight_score[square]
                elif piece == B: 
                    score += bishop_score[square]
                    score += bin(get_bishop_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == R: 
                    score += rook_score[square]
                    score += semi_open_file_score if (board.bitboards[P] & file_masks[square]) == 0 else 0
                    score += open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                elif piece == Q: 
                    score += queen_score[square]
                    score += bin(get_queen_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == K: 
                    score += king_score[square]
                    score -= semi_open_file_score if (board.bitboards[P] & file_masks[square]) == 0 else 0
                    score -= open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                    score += king_sheild_bonus * bin(king_attacks[square] & board.occupancies[white]).count('1')

                # positional scores for black pieces
                if piece == p:
                    score -= pawn_score[mirror_score[square]]
                    score += -double_pawn_penalty * (bin(board.bitboards[p] & file_masks[square]).count("1") - 1)
                    score += -isolated_pawns_penalty if (board.bitboards[p] & isolated_pawns[square]) == 0 else 0
                    score -= passed_pawn_bonus[ranks[mirror_score[square]]] if (board.bitboards[P] & passed_pawns_b[square]) == 0 else 0
                elif piece == n: score -= knight_score[mirror_score[square]]
                elif piece == b: 
                    score -= bishop_score[mirror_score[square]]
                    score -= bin(get_bishop_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == r: 
                    score -= rook_score[mirror_score[square]]
                    score -= semi_open_file_score if (board.bitboards[p] & file_masks[square]) == 0 else 0
                    score -= open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                elif piece == q: 
                    score -= queen_score[mirror_score[square]]
                    score -= bin(get_queen_attacks(square, board.occupancies[both])).count('1') // 2
                elif piece == k: 
                    score -= king_score[mirror_score[square]]
                    score += semi_open_file_score if (board.bitboards[p] & file_masks[square]) == 0 else 0
                    score += open_file_score if ((board.bitboards[P] | board.bitboards[p]) & file_masks[square]) == 0 else 0
                    score -= king_sheild_bonus * bin(king_attacks[square] & board.occupancies[black]).count('1')

                # pop bit
                bitboard ^= lsb

        # Version of sigmoid of the score, pushes the score into the bounds of [-1, 1]
        score = 2 / (1 + np.exp(-score / 500)) - 1
        return score if board.side == white else -score



class PositionWeightVectors:
  def __init__(self, n_positions=19, n_feats=768):
    self.weights = np.random.randn(n_positions, n_feats) * 0.01
    self.alphas = np.ones(n_positions, dtype=float)
    self.visits = np.zeros(n_positions, dtype=int)
  
  def get(self, pos_type: int) -> np.ndarray:
    return self.weights[pos_type]
  
  def set(self, pos_type: int, vector: np.array) -> None:
    self.weights[pos_type] = vector
  
  @classmethod
  def load_weights(self, path: str) -> np.ndarray:
    w = np.load(path)
    return w 
  
  def save_all(self, w_path='RLweightsV1.npy',
                      a_path='RLalphasV1.npy',
                      v_path='RLvisitsV1.npy',
                      model=None) -> None:
    if model == "V2":
        w_path = 'RLweightsV2.npy'
        a_path = 'RLalphasV2.npy'
        v_path = 'RLvisitsV2.npy'
    
    np.save(w_path, self.weights)
    np.save(a_path, self.alphas)
    np.save(v_path, self.visits)

  @classmethod
  def load_all(cls, w_path='RLweightsV1.npy',
                    a_path='RLalphasV1.npy',
                    v_path='RLvisitsV1.npy',
                    model=None):
    if model == "V2":
        w_path = 'RLweightsV2.npy'
        a_path = 'RLalphasV2.npy'
        v_path = 'RLvisitsV2.npy'

    obj = cls()
    obj.weights = np.load(w_path)
    obj.alphas  = np.load(a_path)
    obj.visits  = np.load(v_path)
    return obj



# RL Agent that uses more modular state representation
class RLAgentV2(Agent):
    def __init__(self,  color: int): 
        super().__init__(color)
        
        # follow the pv moves and score of the pv moves
        self.score_pv = 0
        self.follow_pv = 0

        self.best_val = 0
        self.states = []  # list of states

        self.epsilon = 0.00
        self.eps_start = self.epsilon
        self.eps_end = 0.00
        self.eps_decay_rate = 0.0002

    # exponential epsilon decay
    def update_epsilon(self, episode: int) -> None:
        self.epsilon = self.eps_end + \
            (self.eps_start - self.eps_end) * \
            np.exp(-self.eps_decay_rate * episode)

    # position type for early, mid, and late game
    def get_position_type(self, board: Board) -> int:
       
        # both queens on board
        queens = int(board.bitboards[Q] > 0 and board.bitboards[q] > 0)

        # minor pieces on board for both sides
        minor_piece_count = 0
        minor_piece_count += np.bitwise_count(board.bitboards[R] | board.bitboards[B] | board.bitboards[N])
        minor_piece_count += np.bitwise_count(board.bitboards[r] | board.bitboards[b] | board.bitboards[n])
        
        # endgame
        if (not queens and (minor_piece_count < 6)) or (queens and (minor_piece_count < 3)):
            return 2
        # mid game
        elif (not queens and (minor_piece_count < 9)) or (queens and (minor_piece_count < 6)):
            return 1
        
        # Early Game 
        return 0
    
    def build_feat_vec(self, board: Board) -> np.ndarray:
        # 0: White mobility, 1: Black mobility
        # 2–7: White piece counts, 8–13: Black piece counts
        # 14–77:  White pawn squares
        # 78–141: White knight squares
        # 142-205: White Bishop squares
        # 206-269: White Rook squares
        # 270–333: White queen squares
        # 334–397: White king squares
        # 398–400: White pawn structure
        # 401–403: Black pawn structure
        # 404: White king shield, 405: Black king shield
        # 406: king Manhattan distance − 1
        # 407: total piece count
        vec = np.zeros(408)
        w_occ = board.occupancies[white]
        b_occ = board.occupancies[black]
        occ_both = board.occupancies[both]
        white_pieces = (P, N, B, R, Q, K)
        black_pieces = (p, n, b, r, q, k)

        w_king_pos = 0
        b_king_pos = 0

        # white side
        for i, piece in enumerate(white_pieces):
            bb = board.bitboards[piece]
            while bb:
                lsb = bb & -bb
                idx = LS1B_IDX[lsb]

                # Piece val
                vec[2 + i] += 1

                # Pos val
                vec[14 + (64 * i) + idx] += 1

                # total pieces
                vec[407] += 1

                # Pawns
                if i == 0:

                    vec[398] = 1 if (board.bitboards[piece] & isolated_pawns[idx]) == 0 else 0
                    vec[399] = 1 * bin(board.bitboards[piece] & file_masks[idx]).count('1')
                    vec[400] = 1 if (board.bitboards[p] & passed_pawns_w[idx]) == 0 else 0

                # Knights
                elif i == 1:
                    # squares to move to no allies
                    vec[0] += bin(knight_attacks[idx] & ~w_occ).count('1')

                # bishop
                elif i == 2:
                    # attacking squares
                    vec[0] += bin(get_bishop_attacks(idx, occ_both)).count('1')

                # rook
                elif i == 3:
                    # attacking squares 
                    vec[0] += bin(get_rook_attacks(idx, occ_both)).count('1')
                
                # Queen
                elif i == 4:
                    # attacking squares 
                    vec[0] += bin(get_queen_attacks(idx, occ_both)).count('1')

                # king
                else:
                    w_king_pos = idx
                    vec[404] = 1 * bin(king_attacks[idx] & w_occ).count('1')

                bb ^= lsb

            
        # black side
        for i, piece in enumerate(black_pieces):
            bb = board.bitboards[piece]
            while bb:
                lsb = bb & -bb
                idx = LS1B_IDX[lsb]

                # Piece val
                vec[8 + i] += 1

                # Pos val
                vec[mirror_score[idx] + 14 + (64 * i)] -= 1

                # total pieces
                vec[407] += 1

                # Pawns
                if i == 0:

                    vec[401] = 1 if (board.bitboards[piece] & isolated_pawns[idx]) == 0 else 0
                    vec[402] = 1 * bin(board.bitboards[piece] & file_masks[idx]).count('1')
                    vec[403] = 1 if (board.bitboards[P] & passed_pawns_b[idx]) == 0 else 0

                # Knights
                elif i == 1:
                    # squares to move to no allies
                    vec[1] += bin(knight_attacks[idx] & ~b_occ).count('1')

                # bishop
                elif i == 2:
                    # attacking squares
                    vec[1] += bin(get_bishop_attacks(idx, occ_both)).count('1')

                # rook
                elif i == 3:
                    # attacking squares 
                    vec[1] += bin(get_rook_attacks(idx, occ_both)).count('1')
                
                # Queen
                elif i == 4:
                    # attacking squares 
                    vec[1] += bin(get_queen_attacks(idx, occ_both)).count('1')

                # king
                else:
                    b_king_pos = idx
                    vec[405] = 1 * bin(king_attacks[idx] & b_occ).count('1')

                bb ^= lsb
            
        # rank and file of kings
        wr, wf = divmod(w_king_pos, 8)
        br, bf = divmod(b_king_pos, 8)

        # manhattan distance of kings
        vec[406] = (abs(wr - br) + abs(wf - bf)) - 1
        
        # normalize
        mag = np.linalg.norm(vec)
        return vec / mag
        

    def evaluate(self, board: Board, weights: np.ndarray) -> tuple[int, np.ndarray, int]:
        pos_type = self.get_position_type(board)
        vec = self.build_feat_vec(board)
        return (vec.dot(weights[pos_type]), vec, pos_type) if board.side == white else \
               (-vec.dot(weights[pos_type]), vec, pos_type)

    def score_move(self, move: int, board: Board) -> int:
        # if PV scoring
        if self.score_pv:
            # if move in PV
            if board.pv_table[0][self.ply] == move:
                # disable scorepv
                self.score_pv = 0 
                # give PV move the highest score
                return 50000

        # capture moves
        if (move >> 20) & 1:
            # mvv lva 
            target = P if board.side == black else p
            move_target = (move & 0xfc0) >> 6
            if board.side == white: start, end = p, k
            else: start, end = P, K

            # attacker bitboards
            for bb in range(start, end + 1):
                # piece on target square
                if board.bitboards[bb] & BIT[move_target]:
                    target = bb
                    break

            return mvv_lva[(move & 0xf000) >> 12][target] + 10000
        else:
            # killer move
            if board.killer_moves[0][self.ply] == move:
                return 9000
            
            elif board.killer_moves[1][self.ply] == move:
                return 8000
            
            else:
                return board.history_moves[(move & 0xf000) >> 12][(move & 0xfc0) >> 6]

    
    def enable_pv_scoring(self, board: Board) -> None:
        self.follow_pv = 0

        for mv in board.ML.moves[:board.ML.count]:
            # hit pv move
            if board.pv_table[0][self.ply] == mv:
                # enable scoring and following
                self.score_pv = self.follow_pv = 1

    def clear_vars(self, board: Board) -> None:
        self.ply = self.nodes = self.best_move = self.follow_pv = self.score_pv = 0
        board.killer_moves = [[0] * MAX_PLY for _ in range(2)]
        board.history_moves = [[0] * 64 for _ in range(12)]
        board.pv_len = [0] * MAX_PLY
        board.pv_table = [[0] * MAX_PLY for _ in range(MAX_PLY)]
    

    def search_position(self, depth: int, board: Board, weights: np.ndarray) -> int:
        # reset all vars
        self.clear_vars(board)
        
        if np.random.rand() < self.epsilon:
            return self.pick_random_move(board, weights)

        # iterative deepening 
        for d in range(1, depth + 1):
            self.nodes = 0

            # enable follow pv
            self.follow_pv = 1
            self.best_val = self.negamax_with_move_ordering(-100000, 100000, d, board, weights)
            self.best_move = board.pv_table[0][0]

        val, best_vec, best_pt = self.evaluate(board, weights)
        if self.best_move == 0:
          score_to_store = val
        else:
          score_to_store = (val if abs(self.best_val) >= 90000
                          else self.best_val)
          
        self.states.append((best_vec, best_pt, score_to_store))

        return self.best_val
    
    # epsilon greedy move selection
    def pick_random_move(self, board: Board, weights: np.ndarray) -> int:
        # game rule based tie
        if board.fifty >= 100:
            self.best_move = 0
            return 0
        
        board.generate_moves()
        moves = board.ML.moves[:board.ML.count]
        val, vec, pt = self.evaluate(board, weights)
        # pick random move until you pick a legal one
        while moves:
            st = board.copy_board()
            move = random.choice(moves)
            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            if not board.make_move(move, all_moves):
                moves.remove(move)
                self.ply -= 1
                board.repetition_index -= 1
                continue

            board.restore_board(st)
            self.best_move = move
            self.ply -= 1
            board.repetition_index -= 1
            break
        
        # if no legal moves in position
        if len(moves) == 0:
            self.best_move = 0
        
        self.states.append((vec, pt, val))
        return val
      
    # https://web.archive.org/web/20071031100051/http://www.brucemo.com/compchess/programming/hashing.htm
    def negamax_with_move_ordering(self, alpha: int, beta: int, depth: int, board: Board, weights: np.ndarray) -> int:
        # hash flag = hash flag alpha
        hashf = hfa

        if (self.ply and board.is_repetition()) or board.fifty >= 100:
            return 0

        # read hash table
        val = board.read_trans_table(alpha, beta, depth, self.ply)
        # if we have seen the move before
        if self.ply and val != 500000:
            _, vec, pt = self.evaluate(board, weights)
            return val

        # init PV len
        board.pv_len[self.ply] = self.ply 

        # base case
        if depth == 0:
            # run quiescent search
            return self.quiescence_with_move_ordering(alpha, beta, board, weights)
        
        # too deep of a search
        if self.ply >= MAX_PLY:
            val, _, _ = self.evaluate(board, weights)
            return val
        
        self.nodes += 1

        king = K if board.side == white else k
        lsb = board.bitboards[king] & -board.bitboards[king]
        king_in_check = board.is_square_attacked(LS1B_IDX[lsb], board.side ^ 1)
        if king_in_check:
            depth += 1

        legal_moves = 0

        board.generate_moves()
        
        # following PV line
        if self.follow_pv:
            self.enable_pv_scoring(board)

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, all_moves):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            # incr legal moves
            legal_moves += 1

            # recurse
            score = -self.negamax_with_move_ordering(-beta, -alpha, depth - 1, board, weights)
            board.restore_board(st)
            self.ply -= 1
            board.repetition_index -= 1
            
            # found better move
            if score > alpha:
                # update hash flag
                hashf = hfe

                # only quiet moves
                if not (mv >> 20) & 1:
                    # store history
                    board.history_moves[(mv & 0xf000) >> 12][(mv & 0xfc0) >> 6] += depth

                alpha = score

                # write PV move
                board.pv_table[self.ply][self.ply] = mv

                for next_ply in range(self.ply + 1, board.pv_len[self.ply + 1]):
                    # copy move from deeper ply
                    board.pv_table[self.ply][next_ply] = board.pv_table[self.ply + 1][next_ply]

                # adjust pv length
                board.pv_len[self.ply] = board.pv_len[self.ply + 1]

                # fail hard, store killer mvoes
                if score >= beta:
                    # store position
                    board.write_trans_table(depth, beta, hfb, self.ply)
                    # only quiet moves
                    if not (mv >> 20) & 1:
                        # store killer moves
                        board.killer_moves[1][self.ply] = board.killer_moves[0][self.ply]
                        board.killer_moves[0][self.ply] = mv
                    return beta

        # no legal moves
        if legal_moves == 0:
            if king_in_check:
                # checkmate score
                return -np.inf + self.ply
            else:
                # stalemate score
                return 0
            
        board.write_trans_table(depth, alpha, hashf, self.ply)

        return alpha
    

    def quiescence_with_move_ordering(self, alpha: int, beta: int, board: Board, weights: np.ndarray) -> int:
        self.nodes += 1

        score, _, _ = self.evaluate(board, weights)

        if score >= beta:
            return beta
        
        if score > alpha:
            alpha = score

        board.generate_moves()

        moves = self.sort_moves(board)

        # loop over all available moves
        for mv in moves:
            # preserve current state
            st = board.copy_board()
            # incr moves
            self.ply += 1

            board.repetition_index += 1
            board.repetition_table[board.repetition_index] = board.hash_key

            # illegal move
            if not board.make_move(mv, only_captures):
                self.ply -= 1
                board.repetition_index -= 1
                continue

            score = -self.quiescence_with_move_ordering(-beta, -alpha, board, weights)

            self.ply -= 1
            board.repetition_index -= 1

            board.restore_board(st)

            # fail hard
            if score >= beta:
                return beta

            if score > alpha:
                alpha = score
        
        return alpha
    

    # TD(lambda) algorithm based off https://www.researchgate.net/publication/228573088_Using_reinforcement_learning_in_chess_engines
    def td_lamba_update(self, r: int, lam: float, all_vecs: PositionWeightVectors) -> list[np.float64]:

        # load relative parameters
        ws, alphas, visits = all_vecs.weights, all_vecs.alphas, all_vecs.visits
        alpha_min = 0.01

        # if there is no states for any reason just return
        if not self.states:
            return 0

        # get the relative values from our list of states
        vecs, types, vals = zip(*self.states)
        N = len(vals)
        # append rerward to the end of values
        vals = list(vals) + [r]

        # compute deltas. Only take negative deltas here such that we dont
        # learn from our opponents mistakes. clip so that the values dont run away
        sign = 1 if self.color == white else -1
        deltas = [np.clip(sign *(vals[t+1] - vals[t]), -5.0, 5.0)
                for t in range(N)]
        
        # loop over all states in previous game
        for t in range(N):
            # get position type for that state
            pos_type = types[t]
            # get the vecotr for that state
            w_t = vecs[t]

            # compute the delta_t value using lamba discounting
            del_t = sum((lam**(j-t)) * deltas[j] for j in range(t, N))

            # get the relevent alpha value for that state
            a = alphas[pos_type]

            # incriment the weight vector with all of these values
            # not that w_t is the same as the gradient of the value function at time t
            ws[pos_type] += a * del_t * w_t

            # incriment visits and the alpha for that position
            visits[pos_type] += 1
            alphas[pos_type] = max(alpha_min, 1.0 / np.sqrt(1 + 0.00005 * visits[pos_type]))
        
        # failsafe to make sure no values hget too larcge or small
        ws = np.nan_to_num(ws, nan=0.0, posinf=1e6, neginf=-1e6)

        # multiply the all weights by 0.7 to make sure they dont get too large
        #ws *= 0.995

        # save values
        all_vecs.weights, all_vecs.alphas, all_vecs.visits = ws, alphas, visits

        return deltas
    
def reset_vals() -> None:
  s = PositionWeightVectors(n_feats=408, n_positions=3)
  s.save_all(model="V2")


# training loop for RL agents
def trainRL(lam=0.7, num_games=100000) -> None:
    # open file with starting positions obtained from chess.com
    f = open(r"openings\eco_fens.txt", "r")
    lines = [l.strip() for l in f if l.strip()]
    # sample lines from the file for the amount of games we have
    sampled = random.choices(lines, k=num_games + 1)

    # load the vectors
    all_vecs = PositionWeightVectors.load_all(model="V2")

    returns = []
    for game, fen in enumerate(sampled):
        # new board
        board = Board(fen)

        # setup agents
        white_agent = RLAgentV2(white)
        black_agent = RLAgentV2(black)

        # update agent epsilons
        white_agent.update_epsilon(game)
        black_agent.update_epsilon(game)

        # play the game
        while True:
            agent = white_agent if board.side == white else black_agent
            if agent.color == black:
                agent.search_position(2, board, all_vecs.weights)
            else:
                agent.search_position(2, board, all_vecs.weights)
            if not board.make_move(agent.best_move, all_moves):
                break

        # assign rewards to agents
        wr = br = 0
        if board.winner == white:
            wr, br = 1, -1
        elif board.winner == black:
            wr, br = -1, 1
        
        # update weight vectors
        deltaw = white_agent.td_lamba_update(wr, lam, all_vecs)
        deltab = black_agent.td_lamba_update(br, lam, all_vecs)

        # clear state array of agents
        white_agent.states.clear()
        black_agent.states.clear()

        print(game)
        returns.append(wr)

        # every 10 games print metrics and save values
        if game % 10 == 0:
            print(f"‖w‖₂={np.linalg.norm(all_vecs.weights):.1f}")
            print(f"epsilon: {white_agent.epsilon}")
            print(returns)
            print(deltaw)
            print(deltab)
            print(np.mean(deltaw))
            print(all_vecs.alphas)
            all_vecs.save_all(model="V2")


if __name__ == "__main__":
    trainRL()