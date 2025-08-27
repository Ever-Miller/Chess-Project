from dataclasses import dataclass
import numpy as np
#--------------------------------------------------ENUMS--------------------------------------------------------------

# board squares
a8, b8, c8, d8, e8, f8, g8, h8 = 0, 1, 2, 3, 4, 5, 6, 7
a7, b7, c7, d7, e7, f7, g7, h7 = 8, 9, 10, 11, 12, 13, 14, 15
a6, b6, c6, d6, e6, f6, g6, h6 = 16, 17, 18, 19, 20, 21, 22, 23
a5, b5, c5, d5, e5, f5, g5, h5 = 24, 25, 26, 27, 28, 29, 30, 31
a4, b4, c4, d4, e4, f4, g4, h4 = 32, 33, 34, 35, 36, 37, 38, 39
a3, b3, c3, d3, e3, f3, g3, h3 = 40, 41, 42, 43, 44, 45, 46, 47
a2, b2, c2, d2, e2, f2, g2, h2 = 48, 49, 50, 51, 52, 53, 54, 55
a1, b1, c1, d1, e1, f1, g1, h1 = 56, 57, 58, 59, 60, 61, 62, 63
no_square = 64

# sides
white = 0
black = 1
both = 2

# castling types
wk, wq, bk, bq = 1, 2, 4, 8
# piece types
P, N, B, R, Q, K, p, n, b, r, q, k = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11

# for making moves
all_moves, only_captures = 0, 1

# for hashing (hash flag exact, alpha, and beta)
hfe, hfa, hfb = 0, 1, 2

mate_score, mate_value = 98000, 99000
# ------------------------------ LOOKUP TABLES ---------------------------------
rook_magics = [
    0x8a80104000800020,
    0x140002000100040,
    0x2801880a0017001,
    0x100081001000420,
    0x200020010080420,
    0x3001c0002010008,
    0x8480008002000100,
    0x2080088004402900,
    0x800098204000,
    0x2024401000200040,
    0x100802000801000,
    0x120800800801000,
    0x208808088000400,
    0x2802200800400,
    0x2200800100020080,
    0x801000060821100,
    0x80044006422000,
    0x100808020004000,
    0x12108a0010204200,
    0x140848010000802,
    0x481828014002800,
    0x8094004002004100,
    0x4010040010010802,
    0x20008806104,
    0x100400080208000,
    0x2040002120081000,
    0x21200680100081,
    0x20100080080080,
    0x2000a00200410,
    0x20080800400,
    0x80088400100102,
    0x80004600042881,
    0x4040008040800020,
    0x440003000200801,
    0x4200011004500,
    0x188020010100100,
    0x14800401802800,
    0x2080040080800200,
    0x124080204001001,
    0x200046502000484,
    0x480400080088020,
    0x1000422010034000,
    0x30200100110040,
    0x100021010009,
    0x2002080100110004,
    0x202008004008002,
    0x20020004010100,
    0x2048440040820001,
    0x101002200408200,
    0x40802000401080,
    0x4008142004410100,
    0x2060820c0120200,
    0x1001004080100,
    0x20c020080040080,
    0x2935610830022400,
    0x44440041009200,
    0x280001040802101,
    0x2100190040002085,
    0x80c0084100102001,
    0x4024081001000421,
    0x20030a0244872,
    0x12001008414402,
    0x2006104900a0804,
    0x1004081002402
]

bishop_magics = [
    0x40040844404084,
    0x2004208a004208,
    0x10190041080202,
    0x108060845042010,
    0x581104180800210,
    0x2112080446200010,
    0x1080820820060210,
    0x3c0808410220200,
    0x4050404440404,
    0x21001420088,
    0x24d0080801082102,
    0x1020a0a020400,
    0x40308200402,
    0x4011002100800,
    0x401484104104005,
    0x801010402020200,
    0x400210c3880100,
    0x404022024108200,
    0x810018200204102,
    0x4002801a02003,
    0x85040820080400,
    0x810102c808880400,
    0xe900410884800,
    0x8002020480840102,
    0x220200865090201,
    0x2010100a02021202,
    0x152048408022401,
    0x20080002081110,
    0x4001001021004000,
    0x800040400a011002,
    0xe4004081011002,
    0x1c004001012080,
    0x8004200962a00220,
    0x8422100208500202,
    0x2000402200300c08,
    0x8646020080080080,
    0x80020a0200100808,
    0x2010004880111000,
    0x623000a080011400,
    0x42008c0340209202,
    0x209188240001000,
    0x400408a884001800,
    0x110400a6080400,
    0x1840060a44020800,
    0x90080104000041,
    0x201011000808101,
    0x1a2208080504f080,
    0x8012020600211212,
    0x500861011240000,
    0x180806108200800,
    0x4000020e01040044,
    0x300000261044000a,
    0x802241102020002,
    0x20906061210001,
    0x5a84841004010310,
    0x4010801011c04,
    0xa010109502200,
    0x4a02012000,
    0x500201010098b028,
    0x8040002811040900,
    0x28000010020204,
    0x6000020202d0240,
    0x8918844842082200,
    0x4010011029020020
]

# Occupancy bit tables for calcualting magics
bishop_relevant_bits = [6, 5, 5, 5, 5, 5, 5, 6, 
                        5, 5, 5, 5, 5, 5, 5, 5, 
                        5, 5, 7, 7, 7, 7, 5, 5,
                        5, 5, 7, 9, 9, 7, 5, 5,
                        5, 5, 7, 9, 9, 7, 5, 5,
                        5, 5, 7, 7, 7, 7, 5, 5,
                        5, 5, 5, 5, 5, 5, 5, 5,
                        6, 5, 5, 5, 5, 5, 5, 6]

rook_relevant_bits = [12, 11, 11, 11, 11, 11, 11, 12, 
                      11, 10, 10, 10, 10, 10, 10, 11, 
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      12, 11, 11, 11, 11, 11, 11, 12]

# masks for ranks and files
RANK = {
    1: 0x00000000000000ff,
    2: 0x000000000000ff00,
    3: 0x0000000000ff0000,
    4: 0x00000000ff000000,
    5: 0x000000ff00000000,
    6: 0x0000ff0000000000,
    7: 0x00ff000000000000,
    8: 0xff00000000000000
}

FILE = {
    'a': 0x0101010101010101,
    'b': 0x0202020202020202,
    'c': 0x0404040404040404,
    'd': 0x0808080808080808,
    'e': 0x1010101010101010,
    'f': 0x2020202020202020,
    'g': 0x4040404040404040,
    'h': 0x8080808080808080,
}

NOT_FILE = {
    'a': 0xfefefefefefefefe,
    'b': 0xfdfdfdfdfdfdfdfd,
    'c': 0xfbfbfbfbfbfbfbfb,
    'd': 0xf7f7f7f7f7f7f7f7,
    'e': 0xefefefefefefefef,
    'f': 0xdfdfdfdfdfdfdfdf,
    'g': 0xbfbfbfbfbfbfbfbf,
    'h': 0x7f7f7f7f7f7f7f7f,
}

CORD_TO_SQUARE = {
    0: "a8",  1: "b8",  2: "c8",  3: "d8",  4: "e8",  5: "f8",  6: "g8",  7: "h8",
    8: "a7",  9: "b7", 10: "c7", 11: "d7", 12: "e7", 13: "f7", 14: "g7", 15: "h7",
   16: "a6", 17: "b6", 18: "c6", 19: "d6", 20: "e6", 21: "f6", 22: "g6", 23: "h6",
   24: "a5", 25: "b5", 26: "c5", 27: "d5", 28: "e5", 29: "f5", 30: "g5", 31: "h5",
   32: "a4", 33: "b4", 34: "c4", 35: "d4", 36: "e4", 37: "f4", 38: "g4", 39: "h4",
   40: "a3", 41: "b3", 42: "c3", 43: "d3", 44: "e3", 45: "f3", 46: "g3", 47: "h3",
   48: "a2", 49: "b2", 50: "c2", 51: "d2", 52: "e2", 53: "f2", 54: "g2", 55: "h2",
   56: "a1", 57: "b1", 58: "c1", 59: "d1", 60: "e1", 61: "f1", 62: "g1", 63: "h1", 64: "no"
}

SQUARE_TO_CORD = {
    "a8": 0,  "b8": 1,  "c8": 2,  "d8": 3,  "e8": 4,  "f8": 5,  "g8": 6,  "h8": 7,
    "a7": 8,  "b7": 9,  "c7":10,  "d7":11,  "e7":12,  "f7":13,  "g7":14,  "h7":15,
    "a6":16,  "b6":17,  "c6":18,  "d6":19,  "e6":20,  "f6":21,  "g6":22,  "h6":23,
    "a5":24,  "b5":25,  "c5":26,  "d5":27,  "e5":28,  "f5":29,  "g5":30,  "h5":31,
    "a4":32,  "b4":33,  "c4":34,  "d4":35,  "e4":36,  "f4":37,  "g4":38,  "h4":39,
    "a3":40,  "b3":41,  "c3":42,  "d3":43,  "e3":44,  "f3":45,  "g3":46,  "h3":47,
    "a2":48,  "b2":49,  "c2":50,  "d2":51,  "e2":52,  "f2":53,  "g2":54,  "h2":55,
    "a1":56,  "b1":57,  "c1":58,  "d1":59,  "e1":60,  "f1":61,  "g1":62,  "h1":63
}

# pieces for printing
ascii_pieces = ["♟", "♞", "♝", "♜", "♛", "♚", "♙", "♘", "♗", "♖", "♕", "♔"]
#ascii_pieces = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
char_pieces = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11
}

promoted_pieces = {
    0: ' ',
    Q: 'q',
    R: 'r',
    B: 'b',
    N: 'n',
    q: 'q',
    r: 'r',
    b: 'b',
    n: 'n'
}

# https://github.com/Mk-Chan/libchess/blob/master/Position.h
"""
                           castling   move     in      in
                              right update     binary  decimal

 king & rooks didn't move:     1111 & 1111  =  1111    15

        white king  moved:     1111 & 1100  =  1100    12
  white king's rook moved:     1111 & 1110  =  1110    14
 white queen's rook moved:     1111 & 1101  =  1101    13
     
         black king moved:     1111 & 0011  =  1011    3
  black king's rook moved:     1111 & 1011  =  1011    11
 black queen's rook moved:     1111 & 0111  =  0111    7
"""

# castling right update array
castling_rights = [
     7, 15, 15, 15,  3, 15, 15, 11,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    13, 15, 15, 15, 12, 15, 15, 14
]


# gets index of LSB
LS1B_IDX = {1<<i:i for i in range(64)}
# lookup table for 1 << n patterns n = {0 - 63}
BIT = tuple(1 << sq for sq in range(64))

MASK32 = 0xffffffff
MASK64 = 0xffffffffffffffff

# FEN dedug positions from https://www.chessprogramming.org/Perft_Results
empty_board = "8/8/8/8/8/8/8/8 b -"
start_position = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq"
tricky_position = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq"
# [side][square]
pawn_attacks = [[0] * 64 for _ in range(2)]

# [square]
knight_attacks = [0] * 64

# [square]
king_attacks = [0] * 64

# [square]
bishop_masks = [0] * 64

# [square]
rook_masks = [0] * 64

# [square][occupancies]
bishop_attacks = [[0] * 512  for _ in range(64)]

# [square][occupancies]
rook_attacks = [[0] * 4096 for _ in range(64)]

INITIALIZED = False

# -----------------------------------------------------------RANDOM-------------------------------------------------------

random_state = 1804289383

# generate 32_bit pseudo leagl nums https://en.wikipedia.org/wiki/Xorshift
def get_random_U32_number():
    global random_state

    # xor shift 32 algorithm
    x = random_state & MASK32

    x ^= (x << 13) & MASK32
    x ^= (x >> 17) & MASK32
    x ^= (x <<  5) & MASK32

    random_state = x
    return x

# generate random 64 bit number using logic above
def get_random_U64_number():
    
    n1 = (get_random_U32_number() & 0xffff)
    n2 = (get_random_U32_number() & 0xffff)
    n3 = (get_random_U32_number() & 0xffff)
    n4 = (get_random_U32_number() & 0xffff)

    return n1 | (n2 << 16) | (n3 << 32) | (n4 << 48)

# and 3 numbers together to get sparse 64-bit int
def generate_magic_number() -> int:
    return get_random_U64_number() & get_random_U64_number() & get_random_U64_number()

# finds magic numbers for attack tables https://www.chessprogramming.org/Looking_for_Magics
def find_magic_number(square: int, relevent_bits: int, bishop: bool) -> int:
    occupancies = [0] * 4096
    attacks = [0] * 4096

    attack_mask = mask_bishop_attacks(square) if bishop else mask_rook_attacks(square)

    occupancy_indicies = 1 << relevent_bits

    for index in range(occupancy_indicies):
        # get occumancy for current index
        occupancies[index] = set_occupancy(index, relevent_bits, attack_mask)

        # get attack map for current index
        attacks[index] = all_bishop_attacks(square, occupancies[index]) if bishop else all_rook_attacks(square, occupancies[index])

    # test magic numbers
    for _ in range(1000000):
        used_attacks = [0] * occupancy_indicies

        # generate magic number candidate 64-bit
        magic_number = generate_magic_number()

        # skip bad numbers
        top = ((attack_mask * magic_number) & MASK64) >> 56
        if bin(top).count('1') < 6:
            continue
        
        fail = False
        for i in range(occupancy_indicies):

            # init magic index
            u64 = (occupancies[i] * magic_number) & MASK64
            magic_index = (u64) >> (64 - relevent_bits)

            # if index works
            if used_attacks[magic_index] == 0:
                # itit used atacks
                used_attacks[magic_index] = attacks[i]
            
            elif used_attacks[magic_index] != attacks[i]:
                # magic index does not work
                fail = True
                break
            
        # magic number works
        if not fail:
            return magic_number
    
    return 0

def init_magic_numbers() -> None:

    # loop over board
    for square in range(0, 64):
        # rook numbers
        print(hex(find_magic_number(square, rook_relevant_bits[square], False)))

    # loop over board
    for square in range(0, 64):
        # bishop numbers
        print(hex(find_magic_number(square, bishop_relevant_bits[square], True)))

# --------------------------------------HELPER FUNCTIONS----------------------------------------

# https://cs.stackexchange.com/questions/81896/how-to-access-the-bit-at-a-particular-index
# returns the bit at the squareth index 
def get_bit(bb: int, square: int) -> int:
    return bb & (1 << square)

# sets the bit at square index
def set_bit(bb: int, square: int) -> int:
    return bb | (1 << square)

# removes the bit at the index
def pop_bit(bb: int, square: int) -> int:
    return bb & ~(1 << square)

# returns # of bits in a 64 bit int 
def count_bits(bb: int) -> int:
    return bin(bb).count('1')

# returns index of lsb
def get_lsb_index(bb: int) -> int:
    if bb:
        # count the trailing 1's from lsb
        return count_bits((bb & -bb) - 1)
    else:
        return -1
    

""" MOVE FORMAT 
    0000 0000 0000 0000 0011 1111   source square
    0000 0000 0000 1111 1100 0000   target square 
    0000 0000 1111 0000 0000 0000   piece
    0000 1111 0000 0000 0000 0000   promo piece
    0001 0000 0000 0000 0000 0000   capture flag 
    0010 1000 0000 0000 0000 0000   double pawn flag 
    0100 0000 0000 0000 0000 0000   enpassant flag 
    1000 0000 0000 0000 0000 0000   castling
"""

# encode move to int
def encode_move(source: int, target: int, piece: int, promo: int, capture: int, double: int, enpassant: int, castling: int) -> int:
    return (source) | (target << 6) | (piece << 12) | (promo << 16) | (capture << 20) | (double << 21) | (enpassant << 22) | (castling << 23)

def get_move_source(move: int) -> int:
    return move & 0x3f

def get_move_target(move: int) -> int:
    return (move & 0xfc0) >> 6

def get_move_piece(move: int) -> int:
    return (move & 0xf000) >> 12

def get_move_promo(move: int) -> int:
    return (move & 0xf0000) >> 16

def get_move_capture(move: int) -> int:
    return (move >> 20) & 1

def get_move_double(move: int) -> int:
    return (move >> 21) & 1

def get_move_enpassant(move: int) -> int:
    return (move >> 22) & 1

def get_move_castling(move: int) -> int:
    return (move >> 23) & 1


def print_bitboard(bitboard: int) -> None:
    for rank in range(8):
        rank_str = ""
        for file in range(8):
            square = rank * 8 + file

            if file == 0:
                rank_str += str(8 - rank) + " "

            if get_bit(bitboard, square):
                num = 1
            else:
                num = 0
            rank_str += str(num) + " "
        print(rank_str)

    print("  a b c d e f g h")
    print(bitboard)



# -----------------------------------------ATTACKS----------------------------------------------

# https://www.chessprogramming.org/Pawn_Attacks_(Bitboards)
# generates pawn attacks for a certain square and side
def mask_pawn_attacks(side: int, square: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    # white pawns
    if side == white:
        # attack right with no wrapping
        if (bb >> 7) & NOT_FILE["a"]: attacks |= (bb >> 7)  
        # attack left
        if (bb >> 9) & NOT_FILE["h"]: attacks |= (bb >> 9)

    # black pawns
    else:
        # attack right with no wrapping
        if (bb << 7) & NOT_FILE["h"]: attacks |= (bb << 7) 
        # attack left
        if (bb << 9) & NOT_FILE["a"]: attacks |= (bb << 9)

    return attacks & MASK64

# https://www.chessprogramming.org/Knight_Pattern
# generate knight attacks
def mask_knight_attacks(square: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    if (bb >> 17 & NOT_FILE['h']): attacks |= (bb >> 17)
    if (bb >> 15 & NOT_FILE['a']): attacks |= (bb >> 15)
    if (bb >> 10 & (NOT_FILE['h'] & NOT_FILE['g'])): attacks |= (bb >> 10)
    if (bb >> 6  & (NOT_FILE['a'] & NOT_FILE['b'])): attacks |= (bb >> 6)

    if (bb << 17 & NOT_FILE['a']): attacks |= (bb << 17)
    if (bb << 15 & NOT_FILE['h']): attacks |= (bb << 15)
    if (bb << 10 & (NOT_FILE['a'] & NOT_FILE['b'])): attacks |= (bb << 10)
    if (bb << 6  & (NOT_FILE['h'] & NOT_FILE['g'])): attacks |= (bb << 6)

    return attacks

# generate king attacks
def mask_king_attacks(square: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    if (bb >> 9 & NOT_FILE['h']): attacks |= (bb >> 9)
    if (bb >> 8): attacks |= (bb >> 8)
    if (bb >> 7 & NOT_FILE['a']): attacks |= (bb >> 7)
    if (bb >> 1 & NOT_FILE['h']): attacks |= (bb >> 1)

    if (bb << 9 & NOT_FILE['a']): attacks |= (bb << 9)
    if (bb << 8): attacks |= (bb << 8)
    if (bb << 7 & NOT_FILE['h']): attacks |= (bb << 7)
    if (bb << 1 & NOT_FILE['a']): attacks |= (bb << 1)

    return attacks & MASK64

# generate bishop attacks
def mask_bishop_attacks(square: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    # target ranks and files
    tr = square // 8
    tf = square % 8

    # diagonal directions
    rank, file = tr + 1, tf + 1
    while rank <= 6 and file <= 6:
        # get the bit at current position
        attacks |= (BIT[(rank * 8 + file)])
        rank += 1
        file += 1

    rank, file = tr - 1, tf + 1
    while rank >= 1 and file <= 6:
        attacks |= (BIT[(rank * 8 + file)])
        rank -= 1
        file += 1

    rank, file = tr + 1, tf - 1
    while rank <= 6 and file >= 1:
        attacks |= (BIT[(rank * 8 + file)])
        rank += 1
        file -= 1

    rank, file = tr - 1, tf - 1
    while rank >= 1 and file >= 1:
        attacks |= (BIT[(rank * 8 + file)])
        rank -= 1
        file -= 1

    return attacks


# generate rook attacks
def mask_rook_attacks(square: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    # target ranks and files
    tr = square // 8
    tf = square % 8

    # all 4 directions
    for r in range(tr+1, 7):     attacks |= (1 << (r * 8 + tf))
    for r in range(tr-1, 0, -1): attacks |= (1 << (r * 8 + tf))

    for f in range(tf+1, 7):     attacks |= (1 << (tr * 8 + f))
    for f in range(tf-1, 0, -1): attacks |= (1 << (tr * 8 + f))

    return attacks


# generate bishop attacks accounting for blockers
def all_bishop_attacks(square: int, block: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    # target ranks and files
    tr = square // 8
    tf = square % 8

    # scan through diagonals again
    rank, file = tr + 1, tf + 1
    while rank <= 7 and file <= 7:
        attacks |= (BIT[(rank * 8 + file)])
        # if there is a piece blocking break current direction
        if (BIT[(rank * 8 + file)]) & block:
            break
        rank += 1
        file += 1

    rank, file = tr - 1, tf + 1
    while rank >= 0 and file <= 7:
        attacks |= (BIT[(rank * 8 + file)])
        if (BIT[(rank * 8 + file)]) & block:
            break
        rank -= 1
        file += 1

    rank, file = tr + 1, tf - 1
    while rank <= 7 and file >= 0:
        attacks |= (BIT[(rank * 8 + file)])
        if (BIT[(rank * 8 + file)]) & block:
            break
        rank += 1
        file -= 1

    rank, file = tr - 1, tf - 1
    while rank >= 0 and file >= 0:
        attacks |= (BIT[(rank * 8 + file)])
        if (BIT[(rank * 8 + file)]) & block:
            break
        rank -= 1
        file -= 1

    return attacks


# generate rook attacks with blockers
def all_rook_attacks(square: int, blocker: int) -> int:
    attacks = 0
    bb = 0
    bb = set_bit(bb, square)

    # target ranks and files
    tr = square // 8
    tf = square % 8

    for r in range(tr+1, 8):     
        attacks |= (1 << (r * 8 + tf))
        # if there is a blocker break direction
        if (1 << (r * 8 + tf)) & blocker:
            break

    for r in range(tr-1, -1, -1): 
        attacks |= (1 << (r * 8 + tf))
        if (1 << (r * 8 + tf)) & blocker:
            break

    for f in range(tf+1, 8):     
        attacks |= (1 << (tr * 8 + f))
        if (1 << (tr * 8 + f)) & blocker:
            break

    for f in range(tf-1, -1, -1):
        attacks |= (1 << (tr * 8 + f))
        if (1 << (tr * 8 + f)) & blocker:
            break

    return attacks


# goes through all variations of occupancies in attack_mask
def set_occupancy(index: int, bits_in_mask: int, attack_mask: int) -> int:
    # occupancy map
    occupancy = 0

    for count in range(bits_in_mask):
        # get lsb idx of attack mask
        square = get_lsb_index(attack_mask)
        # pop lsb
        attack_mask = pop_bit(attack_mask, square)

        # if occupancy is on board
        if index & (1 << count):
            occupancy |= (1 << square)

    return occupancy

        
# set up masks for pawns, kings, and knights
def init_leaper_attacks() -> None:
    # loop over board
    for square in range(64):

        # pawn attacks
        pawn_attacks[white][square] = mask_pawn_attacks(white, square)
        pawn_attacks[black][square] = mask_pawn_attacks(black, square)

        # knight attacks
        knight_attacks[square] = mask_knight_attacks(square)

        # king attacks
        king_attacks[square] = mask_king_attacks(square)


# inits attack tables for sliding pieces
# https://www.chessprogramming.org/Magic_Bitboards
def init_sliders_attacks(bishop: bool) -> None:
    # bishop and rook masks
    for square in range(0, 64):

        # get bishop and rook attack masks for current square
        bishop_masks[square] = mask_bishop_attacks(square) & MASK64
        rook_masks[square] = mask_rook_attacks(square) & MASK64

        # current mask
        attack_mask = bishop_masks[square] if bishop else rook_masks[square]

        # relevant bit count in mask
        relevent_bits = count_bits(attack_mask)

        # init occupancy indiciees
        occupancy_indicies = 1 << relevent_bits

        # loop over all possible combinations of occupancies
        for index in range(occupancy_indicies):
            # cur occupancy (pieces in attack mask)
            occupancy = set_occupancy(index, relevent_bits, attack_mask)
            if bishop:
                # magic index
                magic_index = ((occupancy * bishop_magics[square]) & MASK64) >> (64 - bishop_relevant_bits[square])
                # set bishop attacks
                bishop_attacks[square][magic_index] = all_bishop_attacks(square, occupancy)
            else:
                # magic index
                magic_index = ((occupancy * rook_magics[square]) & MASK64) >> (64 - rook_relevant_bits[square])
                # set rook attacks
                rook_attacks[square][magic_index] = all_rook_attacks(square, occupancy)

def get_bishop_attacks(square: int, occupancy: int) -> int:
    # get attacks from current board occupancy
    occupancy &= bishop_masks[square]
    # get magic index
    idx = ((occupancy * bishop_magics[square]) & MASK64) >> (64 - bishop_relevant_bits[square])

    return bishop_attacks[square][idx]

def get_rook_attacks(square: int, occupancy: int) -> int:
    # get attacks from current board occupancy
    occupancy &= rook_masks[square]
    # get magic index
    idx = ((occupancy * rook_magics[square]) & MASK64) >> (64 - rook_relevant_bits[square])

    return rook_attacks[square][idx]

def get_queen_attacks(square: int, occupancy: int) -> int:
    # init rook and bishop occupancy
    r_occupancy = b_occupancy = occupancy

    # bishop attacks
    b_occupancy &= bishop_masks[square]
    b_idx = ((b_occupancy * bishop_magics[square]) & MASK64) >> (64 - bishop_relevant_bits[square])

    # rook attacks
    r_occupancy &= rook_masks[square]
    r_idx = ((r_occupancy * rook_magics[square]) & MASK64) >> (64 - rook_relevant_bits[square])

    # combine for queen
    return bishop_attacks[square][b_idx] | rook_attacks[square][r_idx]

# initialize all piece attacks maps
def init_all() -> None:
   global INITIALIZED
   INITIALIZED = True
   #init_magic_numbers()
   init_leaper_attacks()
   init_sliders_attacks(True)
   init_sliders_attacks(False)

# ------------------------------------------ENGINE-----------------------------------------------

# This class keeps track of the available moves in a position
@dataclass
class MoveList:
    moves: list[int]
    count: int

    # initialize an array of moves and count 
    # array is fixed size for beter performance
    def __init__(self):
        self.moves = [0] * 256
        self.count = 0

    def add_move(self, move):
        self.moves[self.count] = move
        self.count += 1

    # reset moves and count
    def clear(self):
        self.moves = [0] * 256
        self.count = 0


# this class holds the main engine for making and generating moves
class Board:
   def __init__(self, fen=None):
      if not INITIALIZED:
         # initialize piece attack maps
         init_all()
      
      # max # of moves
      MAX_PLY = 64
      # [id][ply]
      self.killer_moves = [[0] * MAX_PLY for _ in range(2)]
      # [piece][square]
      self.history_moves = [[0] * 64 for _ in range(12)]

      # principlt variation tables
      self.pv_len = [0] * MAX_PLY
      self.pv_table = [[0] * MAX_PLY for _ in range(MAX_PLY)]

      # for gui
      self.capped_pieces = []

      # piece bitboards
      self.bitboards = [0] *  12
      # occupancy bitboards
      self.occupancies = [0] * 3
      # side to move
      self.side = 0
      # en passant
      self.enpassant = no_square
      # castling rights
      self.castling = 0

      # available moves
      self.ML = MoveList()

      self.last_move = -1

      # random piece keys
      # [piece][square]
      self.piece_keys = np.random.randint(0, high=(2**64)- 1, size=(12, 64), dtype=np.uint64)
      self.castle_keys = np.random.randint(0, high=(2**64)- 1, size=16, dtype=np.uint64)
      self.enpassant_keys = np.random.randint(0, high=(2**64)- 1, size=64, dtype=np.uint64)
      self.side_key = np.random.randint(0, high=(2**64)- 1, dtype=np.uint64)
      self.hash_key = np.uint64(0)

      # entries in the form (key, depth, flag, score)
      self.TT = [(0,0,0,0)] * 0x400000 # 4mb

      # repetitons
      self.repetition_table = [0] * 1000
      self.repetition_index = 0

      self.fifty = 0

      self.winner = -1

      if not fen:
         self.parse_fen(start_position)
      else:
         self.parse_fen(fen)
   
   
   def is_repetition(self):
      for key in self.repetition_table[:self.repetition_index]:
         if key == self.hash_key:
            return 1
      return 0
          

   # use hash table https://web.archive.org/web/20071031100051/http://www.brucemo.com/compchess/programming/hashing.htm
   def read_trans_table(self, alpha: int, beta: int, depth: int, ply: int) -> int:
      entry = self.TT[self.hash_key % 0x400000]

      # make hash_keys and depth allign
      if entry[0] == self.hash_key and entry[1] >= depth:
         score = entry[3]
         if score < -mate_score: score += ply
         if score > mate_score: score -= ply
         # match flag
         if entry[2] == hfe:
            # return stored val
            return score
         if entry[2] == hfa and score <= alpha:
            return alpha
         if entry[2] == hfb and score >= beta:
            return beta

      return 500000
   
   # https://github.com/maksimKorzh/chess_programming/blob/master/src/bbc/tt_search_mating_scores/TT_mate_scoring.txt
   def write_trans_table(self, depth: int, val: int, hashf: int, ply: int) -> None:
      # encode earlier mats into hash table
      if val < -mate_score: val -= ply
      if val > mate_score: val += ply

      self.TT[self.hash_key % 0x400000] = (self.hash_key, depth, hashf, val)

   # almost unique position key
   def generate_hash_key(self) -> int:
      final_key = np.uint64(0)

      # loop over piece bitboards
      for piece in range(12):
          bb = self.bitboards[piece]

          while bb:
              lsb = bb & -bb
              idx = LS1B_IDX[lsb]

              # hash piece
              final_key ^= np.uint64(self.piece_keys[piece][idx])

              bb ^= lsb

      # encode enpassant
      if self.enpassant != no_square:
         # hash enpassant
         final_key ^= np.uint64(self.enpassant_keys[self.enpassant])

      # encode castling rights
      final_key ^= np.uint64(self.castle_keys[self.castling])

      # hash only if black to move
      if self.side == black: final_key ^= self.side_key

      return int(final_key)

   # copy board state
   def copy_board(self) -> tuple[int]:
      return (
      self.bitboards.copy(),
      self.occupancies.copy(),
      self.side,
      self.enpassant,
      self.castling,
      self.capped_pieces.copy(),
      self.last_move,
      self.hash_key,
      self.fifty,
      self.winner
      )
   
   # restore board state
   def restore_board(self, state: tuple[int]) -> None:
      self.bitboards, self.occupancies, self.side, self.enpassant, \
         self.castling, self.capped_pieces, self.last_move, self.hash_key, \
            self.fifty, self.winner = state
   
   # reset all global vars
   def reset_all(self) -> None:
      for i in range(len(self.bitboards)):
         self.bitboards[i] = 0
      # occupancy bitboards
      self.occupancies[white] = self.occupancies[black] = self.occupancies[both] = 0
      # side to move
      self.side = 0
      # en passant
      self.enpassant = no_square
      # castling rights
      self.castling = 0

      self.hash_key = np.uint64(0)
      self.repetition_index = 0
      self.repetition_table = [0] * 1000

   # generate all moves
   def generate_moves(self) -> None:
      # clear move list
      self.ML.clear()
      ml = []

      # inline functions for faster generation
      add = ml.append
      enc = encode_move

      # precalculate occupancies
      oc_both = self.occupancies[both]
      oc_white = self.occupancies[white]
      oc_black = self.occupancies[black]

      # promotion pieces
      promos_w = {N, B, R, Q}
      promos_b = {n, b, r, q}

      # init source and target squares
      source_square = target_square = 0

      # init bitboard copy and attacks
      bitboard = attacks = 0

      # piece indices
      if self.side == white:
         # white pieces
         start, stop = 0, 6
      else:
         # black pieces
         start, stop = 6, 12

      # loop over all piece bitboards
      for piece in range(start, stop):
         # init piece bb
         bitboard = self.bitboards[piece]

         # white pawn and white king castle
         if self.side == white:
               
               # pawn logic
               if piece == P:
                  # loop over pawns
                  while bitboard:
                     # get lsb (pawn)
                     lsb = bitboard & -bitboard
                     # index of pawn
                     source_square = LS1B_IDX[lsb]
                     # target move for white pawns
                     target_square = source_square - 8

                     # quiet pawn moves (no captures)
                     if not target_square < a8 and not oc_both & (BIT[target_square]):
                           # promotion
                           if h7 >= source_square >= a7:
                              # initialize moves for all promo pieces
                              for promo in promos_w:
                                 add(enc(source_square, target_square, piece, promo, 0, 0, 0, 0))

                           else:
                              # one square move
                              add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))
                              # two square moves
                              if h2 >= source_square >= a2 and not oc_both & BIT[target_square - 8]:
                                 add(enc(source_square, target_square - 8, piece, 0, 0, 1, 0, 0))

                     # init pawn attacks for only black pieces
                     attacks = pawn_attacks[self.side][source_square] & oc_black

                     # generate pawn caps
                     while attacks:
                           # get first attack and index
                           lsba  = attacks & -attacks
                           target_square = LS1B_IDX[lsba]

                           # promotion capture 
                           if h7 >= source_square >= a7:
                              for promo in promos_w:
                                 add(enc(source_square, target_square, piece, promo, 1, 0, 0, 0))

                           else:
                              # regular capture
                              add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                           # pop lsb
                           attacks ^= lsba

                     # if there is an enpassant square
                     if self.enpassant != no_square:
                           # check if en passant square is in attacks
                           enpassant_attacks = pawn_attacks[self.side][source_square] & BIT[self.enpassant]

                           if enpassant_attacks:
                              # get lsb and index of enpassant
                              lsbe = enpassant_attacks & -enpassant_attacks
                              enpassant_target = LS1B_IDX[lsbe]
                              add(enc(source_square, enpassant_target, piece, 0, 1, 0, 1, 0))

                     # pop lsb from bitboard copy
                     bitboard ^= lsb 

               # castling moves
               if piece == K:
                  # king side castle
                  if self.castling & wk:
                     # squares between king and rook are empty
                     if not oc_both & BIT[f1] and not oc_both & BIT[g1]:
                           # next square and king are not attacked
                           if not self.is_square_attacked(e1, black) and not self.is_square_attacked(f1, black):
                              add(enc(e1, g1, piece, 0, 0, 0, 0, 1))

                  # queen side castle
                  if self.castling & wq:
                     # squares between king and rook are empty
                     if not oc_both & BIT[c1] and not oc_both & BIT[d1] and not oc_both & BIT[b1]:
                           # next square and king are not attacked
                           if not self.is_square_attacked(e1, black) and not self.is_square_attacked(d1, black):
                              add(enc(e1, c1, piece, 0, 0, 0, 0, 1))

         # black pawn and black king castle
         else:
               # black pawn
               if piece == p:
                  # loop over black pawns
                  while bitboard:
                     # get lsb and index
                     lsb = bitboard & -bitboard
                     source_square = LS1B_IDX[lsb]
                     target_square = source_square + 8

                     # quiet pawn moves (no captures)
                     if not target_square > h1 and not oc_both & (BIT[target_square]):
                           # promotion
                           if h2 >= source_square >= a2:
                              # add move for all promo possibilites
                              for promo in promos_b:
                                 add(enc(source_square, target_square, piece, promo, 0, 0, 0, 0))

                           else:
                              # one square move
                              add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))
                              # two squares move 
                              if h7 >= source_square >= a7 and not oc_both & BIT[target_square + 8]:
                                 add(enc(source_square, target_square + 8, piece, 0, 0, 1, 0, 0))

                     # init pawn attacks for only white pieces
                     attacks = pawn_attacks[self.side][source_square] & oc_white

                     # generate pawn caps
                     while attacks:
                           # get first attack and index
                           lsba  = attacks & -attacks
                           target_square = LS1B_IDX[lsba]

                           # promotion capture
                           if h2 >= source_square >= a2:
                              for promo in promos_b:
                                 add(enc(source_square, target_square, piece, promo, 1, 0, 0, 0))
                           else:
                              # regular capture
                              add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                           # pop lsb
                           attacks ^= lsba

                     # if there is an enpassant square
                     if self.enpassant != no_square:
                           enpassant_attacks = pawn_attacks[self.side][source_square] & BIT[self.enpassant]

                           # if ep square is attacked
                           if enpassant_attacks:
                              lsbe = enpassant_attacks & -enpassant_attacks
                              enpassant_target = LS1B_IDX[lsbe]
                              add(enc(source_square, enpassant_target, piece, 0, 1, 0, 1, 0))


                     # pop lsb from bitboard copy
                     bitboard ^= lsb 
               
               # castling moves
               if piece == k:
                  # king side castle
                  if self.castling & bk:
                     # squares between king and rook are empty
                     if not oc_both & BIT[f8] and not oc_both & BIT[g8]:
                           # next square and king are not attacked
                           if not self.is_square_attacked(e8, white) and not self.is_square_attacked(f8, white):
                              add(enc(e8, g8, piece, 0, 0, 0, 0, 1))

                  # queen side castle
                  if self.castling & bq:
                     # squares between king and rook are empty
                     if not oc_both & BIT[c8] and not oc_both & BIT[d8] and not oc_both & BIT[b8]:
                           # next square and king are not attacked
                           if not self.is_square_attacked(e8, white) and not self.is_square_attacked(d8, white):
                              add(enc(e8, c8, piece, 0, 0, 0, 0, 1))

         # squares with ally pieces 1 -> 0
         occ = ~oc_white if self.side == white else ~oc_black
         # occupancy of enemy pieces
         occs = oc_black if self.side == white else oc_white

         # generate knight moves
         if (piece == N) if (self.side == white) else (piece == n):
               # loop over source squares
               while bitboard:
                  # get lsb index 
                  lsb = bitboard & -bitboard
                  source_square = LS1B_IDX[lsb]

                  # piece attacks (not attacking allies)
                  attacks = knight_attacks[source_square] & occ

                  # loop over target squares
                  while attacks:
                     lsba  = attacks & -attacks
                     target_square = LS1B_IDX[lsba]

                     # quiet moves
                     if not occs & (BIT[target_square]):
                           add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))

                     # captures
                     else:
                           add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                     attacks ^= lsba

                  bitboard ^= lsb


         # generate bishop moves
         if (piece == B) if (self.side == white) else (piece == b):
               # loop over source squares
               while bitboard:
                  lsb = bitboard & -bitboard
                  source_square = LS1B_IDX[lsb]

                  # piece attacks (not capturing allies)
                  attacks = get_bishop_attacks(source_square, oc_both) & occ

                  # loop over target squares
                  while attacks:
                     lsba  = attacks & -attacks
                     target_square = LS1B_IDX[lsba]

                     # quiet moves
                     if not occs & (BIT[target_square]):
                           add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))

                     # captures
                     else:
                           add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                     attacks ^= lsba

                  bitboard ^= lsb

         # generate rook moves
         if (piece == R) if (self.side == white) else (piece == r):
               # loop over source squares
               while bitboard:
                  lsb = bitboard & -bitboard
                  source_square = LS1B_IDX[lsb]

                  # piece attacks (not allies)
                  attacks = get_rook_attacks(source_square, oc_both) & occ

                  # loop over target squares
                  while attacks:
                     lsba  = attacks & -attacks
                     target_square = LS1B_IDX[lsba]

                     # quiet moves
                     if not occs & (BIT[target_square]):
                           add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))

                     # captures
                     else:
                           add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                     attacks ^= lsba

                  bitboard ^= lsb
         
         # generate queen moves
         if (piece == Q) if (self.side == white) else (piece == q):
               # loop over source squares
               while bitboard:
                  lsb = bitboard & -bitboard
                  source_square = LS1B_IDX[lsb]

                  # piece attacks (not allies)
                  attacks = get_queen_attacks(source_square, oc_both) & occ

                  # loop over target squares
                  while attacks:
                     lsba  = attacks & -attacks
                     target_square = LS1B_IDX[lsba]

                     # quiet moves
                     if not occs & (BIT[target_square]):
                           add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))

                     # captures
                     else:
                           add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                     attacks ^= lsba

                  bitboard ^= lsb

         # generate king moves
         if (piece == K) if (self.side == white) else (piece == k):
               # loop over source squares
               while bitboard:
                  lsb = bitboard & -bitboard
                  source_square = LS1B_IDX[lsb]

                  # piece attacks (not allies)
                  attacks = king_attacks[source_square] & occ

                  # loop over target squares
                  while attacks:
                     lsba  = attacks & -attacks
                     target_square = LS1B_IDX[lsba]

                     # quiet moves
                     if not occs & (BIT[target_square]):
                           add(enc(source_square, target_square, piece, 0, 0, 0, 0, 0))

                     # captures
                     else:
                           add(enc(source_square, target_square, piece, 0, 1, 0, 0, 0))

                     attacks ^= lsba

                  bitboard ^= lsb
      
      # set movelist moves to ml
      self.ML.moves = ml
      self.ML.count = len(ml)
   

   # make the move on the board
   def make_move(self, move: int, move_flag: int) -> int:
      if move_flag == 10:
         self.generate_moves()
         for mv in self.ML.moves[:self.ML.count]:
            st = self.copy_board()
            if self.make_move(mv, all_moves):
               self.restore_board(st)
               return 0
         king = K if self.side == white else k
         lsb = self.bitboards[king] & -self.bitboards[king]
         # check if king is in check
         if self.is_square_attacked(LS1B_IDX[lsb], self.side ^ 1):
               # Checkmate in no moves and king is attacked
               self.winner = self.side ^ 1
         else:
               # Stalemate if no moves and king is not attacked
               self.winner = 2
         return 0
          
      if self.fifty >= 100:
         # tie
         self.winner = 2
         return 0
      # no legal moves
      if move == 0:
         # get king
         king = K if self.side == white else k
         lsb = self.bitboards[king] & -self.bitboards[king]
         # check if king is in check
         if self.is_square_attacked(LS1B_IDX[lsb], self.side ^ 1):
               # Checkmate in no moves and king is attacked
               self.winner = self.side ^ 1
         else:
               # Stalemate if no moves and king is not attacked
               self.winner = 2
         return 0
      
      # all movoes
      if move_flag == all_moves:
         # copy state
         st = self.copy_board()

         # caputured piece if any
         captured = -1

         # parse move
         source_square = move & 0x3f
         target_square = (move >> 6) & 0x3f
         piece = (move >> 12) & 0xf
         promo = (move >> 16) & 0xf
         capture = (move >> 20) & 1
         double = (move >> 21) & 1
         enpass = (move >> 22) & 1
         castle = move >> 23

         # pre compute masks for source and target square
         src = BIT[source_square]
         tgt = BIT[target_square]

         # pre compute masks for occupancies
         side_occ   = self.occupancies[self.side]
         enemy_occ  = self.occupancies[self.side ^ 1]

         # move piece
         self.bitboards[piece] ^= src | tgt
         # update occupancy
         side_occ ^= src | tgt
         
         # hash new piece position
         self.hash_key ^= np.uint64(self.piece_keys[piece][source_square])
         self.hash_key ^= np.uint64(self.piece_keys[piece][target_square])

         # incr move count
         self.fifty += 1

         # reset move counter if pawn move 
         if piece == P or piece == p:
            self.fifty = 0

         # handle caps
         if capture:
               # reset move counter
               self.fifty = 0
               start_piece = end_piece = 0
               if self.side == white:
                  start_piece = p
                  end_piece = k
               else:
                  start_piece = P
                  end_piece = K

               # loop over piece bitboards
               for bb in range(start_piece, end_piece + 1):
                  # get bit of target piece
                  if self.bitboards[bb] & tgt:
                     # remove target piece from its bitboard
                     self.bitboards[bb] = self.bitboards[bb] ^ tgt
                     # update occupancy
                     enemy_occ ^= tgt
                     captured = bb

                     # update hash key
                     self.hash_key ^= np.uint64(self.piece_keys[bb][target_square])

                     break
         
         # pawn promotion
         if promo:
               pawn = p if self.side == black else P
               # erase pawn
               self.bitboards[pawn] ^= tgt
               # add pawn to bitboard of selected piece
               self.bitboards[promo] ^= tgt


               # remove pawn from hash key
               self.hash_key ^= np.uint64(self.piece_keys[pawn][target_square])
               # add promo piece to hash key 
               self.hash_key ^= np.uint64(self.piece_keys[promo][target_square])

         # enpassant
         if enpass:
               # erase pawn depending on side
               cap_sq = target_square + 8 if self.side == white else target_square - 8
               cap_bit = BIT[cap_sq]
               pawn = p if self.side == white else P
               # remove capped pawn from bitboard
               self.bitboards[pawn] ^= cap_bit
               # update occupancy
               enemy_occ ^= cap_bit
               captured = pawn

               # Hash removed pawn
               self.hash_key ^= np.uint64(self.piece_keys[pawn][cap_sq])
                   

         # hash empassant
         if self.enpassant != no_square:
             self.hash_key ^= np.uint64(self.enpassant_keys[self.enpassant])

         # reset enpassant
         self.enpassant = no_square


         # handle double pawn push 
         if double:
               # set enp square
               self.enpassant = target_square + 8 if self.side == white else target_square - 8
               # hash enpassant square
               self.hash_key ^= self.enpassant_keys[self.enpassant]
                   

         # handle castle (move rooks to correct squares)
         if castle:
               # white king side
               if target_square == g1:
                  self.bitboards[R] ^= BIT[h1] | BIT[f1]
                  side_occ ^= BIT[h1] | BIT[f1]

                  # hash rook
                  self.hash_key ^= np.uint64(self.piece_keys[R][h1])
                  self.hash_key ^= np.uint64(self.piece_keys[R][f1])

               # white queen side
               elif target_square == c1:
                  self.bitboards[R] ^= BIT[a1] | BIT[d1]
                  side_occ ^= BIT[a1] | BIT[d1]

                  # hash rook
                  self.hash_key ^= np.uint64(self.piece_keys[R][a1])
                  self.hash_key ^= np.uint64(self.piece_keys[R][d1])

               # black king side
               elif target_square == g8:
                  self.bitboards[r] ^= BIT[h8] | BIT[f8]
                  side_occ ^= BIT[h8] | BIT[f8]

                  # hash rook
                  self.hash_key ^= np.uint64(self.piece_keys[r][h8])
                  self.hash_key ^= np.uint64(self.piece_keys[r][f8])
                  
               # black queen side
               elif target_square == c8:
                  self.bitboards[r] ^= BIT[a8] | BIT[d8]
                  side_occ ^= BIT[a8] | BIT[d8]

                  # hash rook
                  self.hash_key ^= np.uint64(self.piece_keys[r][a8])
                  self.hash_key ^= np.uint64(self.piece_keys[r][d8])
                  
         
         # hash castling
         self.hash_key ^= np.uint64(self.castle_keys[self.castling])

         # update castle rights
         self.castling &= castling_rights[source_square] & castling_rights[target_square]

         # hash castling
         self.hash_key ^= np.uint64(self.castle_keys[self.castling])

         # reset occupancies
         self.occupancies[self.side] = side_occ
         self.occupancies[self.side ^ 1] = enemy_occ
         self.occupancies[both] = side_occ | enemy_occ

         # change side
         self.side ^= 1

         # hash side
         self.hash_key ^= np.uint64(self.side_key)

         # get current players king
         king = k if self.side == white else K
         lsb = self.bitboards[king] & -self.bitboards[king]
         # make sure king isnt in check
         if self.is_square_attacked(LS1B_IDX[lsb], self.side):
               # illegal move take back if the king is in check after the move
               self.restore_board(st)
               # return illegal move
               return 0

         # add capped piece if any
         if captured != -1:
               self.capped_pieces.append(captured)
         
         # set last move
         self.last_move = move

         # move is legal
         return 1

      # capture moves
      else:
         # move is a capture
         if (move >> 20) & 1:
               self.make_move(move, all_moves)
         else:
               return 0
         
   def isCM(self, side: int) -> bool:
      tmp = self.side
      if self.side != side:
         self.side = side
      self.generate_moves()

      for move in self.ML.moves[:self.ML.count]:
         st = self.copy_board()
         if self.make_move_montey(move, all_moves):
            self.restore_board(st)
            self.side = tmp 
            return False

      king = K if self.side == white else k
      lsb = self.bitboards[king] & -self.bitboards[king]
      if self.is_square_attacked(LS1B_IDX[lsb], side ^ 1):
         self.side = tmp
         return True
      
      self.side = tmp
      return False
      
         
   # make the move on the board
   def make_move_montey(self, move: int, move_flag: int) -> int:
      if self.fifty >= 100:
         # tie
         self.winner = 2
         return 0
      # no legal moves
      if move == 0:
         # get king
         king = K if self.side == white else k
         lsb = self.bitboards[king] & -self.bitboards[king]
         # check if king is in check
         if self.is_square_attacked(LS1B_IDX[lsb], self.side ^ 1):
               # Checkmate in no moves and king is attacked
               self.winner = self.side ^ 1
         else:
               # Stalemate if no moves and king is not attacked
               self.winner = 2
         return 0
      
      # all movoes
      if move_flag == all_moves:
         # copy state
         st = self.copy_board()

         # caputured piece if any
         captured = -1

         # parse move
         source_square = move & 0x3f
         target_square = (move >> 6) & 0x3f
         piece = (move >> 12) & 0xf
         promo = (move >> 16) & 0xf
         capture = (move >> 20) & 1
         double = (move >> 21) & 1
         enpass = (move >> 22) & 1
         castle = move >> 23

         # pre compute masks for source and target square
         src = BIT[source_square]
         tgt = BIT[target_square]

         # pre compute masks for occupancies
         side_occ   = self.occupancies[self.side]
         enemy_occ  = self.occupancies[self.side ^ 1]

         # move piece
         self.bitboards[piece] ^= src | tgt
         # update occupancy
         side_occ ^= src | tgt
         
         # incr move count
         self.fifty += 1

         # reset move counter if pawn move 
         if piece == P or piece == p:
            self.fifty = 0

         # handle caps
         if capture:
               # reset move counter
               self.fifty = 0
               start_piece = end_piece = 0
               if self.side == white:
                  start_piece = p
                  end_piece = k
               else:
                  start_piece = P
                  end_piece = K

               # loop over piece bitboards
               for bb in range(start_piece, end_piece + 1):
                  # get bit of target piece
                  if self.bitboards[bb] & tgt:
                     # remove target piece from its bitboard
                     self.bitboards[bb] = self.bitboards[bb] ^ tgt
                     # update occupancy
                     enemy_occ ^= tgt
                     captured = bb

                     if captured in (k, K):
                        self.winner = self.side
                        return 1
                     
                     break
         
         # pawn promotion
         if promo:
               pawn = p if self.side == black else P
               # erase pawn
               self.bitboards[pawn] ^= tgt
               # add pawn to bitboard of selected piece
               self.bitboards[promo] ^= tgt

         # enpassant
         if enpass:
               # erase pawn depending on side
               cap_sq = target_square + 8 if self.side == white else target_square - 8
               cap_bit = BIT[cap_sq]
               pawn = p if self.side == white else P
               # remove capped pawn from bitboard
               self.bitboards[pawn] ^= cap_bit
               # update occupancy
               enemy_occ ^= cap_bit
               captured = pawn

         # reset enpassant
         self.enpassant = no_square

         # handle double pawn push 
         if double:
               # set enp square
               self.enpassant = target_square + 8 if self.side == white else target_square - 8
                   

         # handle castle (move rooks to correct squares)
         if castle:
               # white king side
               if target_square == g1:
                  self.bitboards[R] ^= BIT[h1] | BIT[f1]
                  side_occ ^= BIT[h1] | BIT[f1]

               # white queen side
               elif target_square == c1:
                  self.bitboards[R] ^= BIT[a1] | BIT[d1]
                  side_occ ^= BIT[a1] | BIT[d1]

               # black king side
               elif target_square == g8:
                  self.bitboards[r] ^= BIT[h8] | BIT[f8]
                  side_occ ^= BIT[h8] | BIT[f8]
                  
               # black queen side
               elif target_square == c8:
                  self.bitboards[r] ^= BIT[a8] | BIT[d8]
                  side_occ ^= BIT[a8] | BIT[d8]

         # update castle rights
         self.castling &= castling_rights[source_square] & castling_rights[target_square]

         # reset occupancies
         self.occupancies[self.side] = side_occ
         self.occupancies[self.side ^ 1] = enemy_occ
         self.occupancies[both] = side_occ | enemy_occ

         # change side
         self.side ^= 1

         # get current players king
         king = k if self.side == white else K
         lsb = self.bitboards[king] & -self.bitboards[king]
         if lsb == 0:
            self.winner = self.side ^ 1
            return 1
         # make sure king isnt in check
         if self.is_square_attacked(LS1B_IDX[lsb], self.side):
               # illegal move take back if the king is in check after the move
               self.restore_board(st)
               # return illegal move
               return 0

         # add capped piece if any
         if captured != -1:
               self.capped_pieces.append(captured)
         
         # set last move
         self.last_move = move

         # move is legal
         return 1

      # capture moves
      else:
         # move is a capture
         if (move >> 20) & 1:
               self.make_move(move, all_moves)
         else:
               return 0
         
      # calculates if square attacked given a side
   def is_square_attacked(self, square: int, side: int) -> int:

      # ––– pawns ––––––––––––––––––––––––––––––––––––––––––
      if side == white:
         # white wants to know if *its* square is hit by black pawns
         if pawn_attacks[black][square] & self.bitboards[P]:
               return 1
      else:                              # side == black
         if pawn_attacks[white][square] & self.bitboards[p]:
               return 1

      # ––– knights ––––––––––––––––––––––––––––––––––––––––
      if knight_attacks[square] & (self.bitboards[N] if side == white else self.bitboards[n]):
         return 1

      # pre‑compute joint occupancy
      occ = self.occupancies[both]

      # ––– bishops ––––––––––––––––––––––––––––––––––––––––
      if get_bishop_attacks(square, occ) & (self.bitboards[B] if side == white else self.bitboards[b]):
         return 1

      # ––– rooks ––––––––––––––––––––––––––––––––––––––––––
      if get_rook_attacks(square, occ) & (self.bitboards[R] if side == white else self.bitboards[r]):
         return 1

      # ––– queens –––––––––––––––––––––––––––––––––––––––––
      if get_queen_attacks(square, occ) & (self.bitboards[Q] if side == white else self.bitboards[q]):
         return 1

      # ––– kings ––––––––––––––––––––––––––––––––––––––––––
      if king_attacks[square] & (self.bitboards[K] if side == white else self.bitboards[k]):
         return 1

      # not attacked
      return 0
   
   
   # parses a fen string to board
   def parse_fen(self, fen: str) -> None:
      # reset all flags and bitboards
      self.reset_all()
      parts = fen.split()
      # board, turn, castle, enp, half, full = parts
      board, turn, castle = parts
      enp = "-"
      square = 0 
      # iterates through string
      for char in board:
         if char == "/":
               continue
         elif char.isdigit():
               # incr square by number
               square += int(char)
         else:
               # get piece typeand set it in the relative bit board
               piece = char_pieces[char]
               self.bitboards[piece] = set_bit(self.bitboards[piece], square)
               square += 1
      

      # set en passant flags
      self.side = white if turn == "w" else black
      for c in castle:
         # king side castle white
         if c == "K":
               self.castling |= wk
         # queen side castle white
         elif c == "Q":
               self.castling |= wq
         # king side castle black 
         elif c == "k":
               self.castling |= bk
         # queen side castle black
         elif c == "q":
               self.castling |= bq

      # parse enpassant
      if enp != "-":
         self.enpassant = SQUARE_TO_CORD[enp]
      # no enpassant
      else:
         self.enpassant = no_square
      

      # init white occupancies
      for piece in range(6):
         self.occupancies[white] |= self.bitboards[piece]

      # init black occupancies
      for piece in range(6, 12):
         self.occupancies[black] |= self.bitboards[piece]
      
      # init all pieces occupancies
      self.occupancies[both] = (self.occupancies[white] | self.occupancies[black])

      # init hash key 
      self.hash_key = np.uint64(self.generate_hash_key())
   

   # parse string input "e7e8q"
   def parse_move(self, move: str) -> int:
      self.generate_moves()

      # source square
      source_square = move[:2]
      source_index = SQUARE_TO_CORD[source_square]

      # target square
      target_square = move[2:4]
      target_index = SQUARE_TO_CORD[target_square]

      for mv in self.ML.moves[:self.ML.count]:
         # check source and target squares in the generated move
         if source_index == get_move_source(mv) and target_index == get_move_target(mv):
               promo_piece = get_move_promo(mv)
               if promo_piece:
                  if (promo_piece == Q or promo_piece == q) and move[4] == "q":
                     return mv
                  elif (promo_piece == R or promo_piece == r) and move[4] == "r":
                     return mv
                  elif (promo_piece == B or promo_piece == b) and move[4] == "b":
                     return mv
                  elif (promo_piece == N or promo_piece == n) and move[4] == "n":
                     return mv
                  
                  continue
                  
               # return legal move
               return mv
      
      return 0

   # test if all of the move logic was implemented correctly
   # https://www.chessprogramming.org/Perft
   def perft_driver(self, depth: int) -> int:
      # escape condition
      if depth == 0:
         return 1
      
      nodes = 0
      self.generate_moves()

      # iterate over all moves
      for mv in self.ML.moves[:self.ML.count]:
         st = self.copy_board()
         
         # if illegal move
         if not self.make_move(mv, all_moves):
               continue
         
         # recurse
         nodes += self.perft_driver(depth - 1)

         # restore board state
         self.restore_board(st)


      return nodes

   # prints given move
   def print_move(self, move: int) -> None:
      move_str = f"{CORD_TO_SQUARE[get_move_source(move)]}{CORD_TO_SQUARE[get_move_target(move)]}{promoted_pieces[get_move_promo(move)]}"
      move_str += f"   {ascii_pieces[get_move_piece(move)]}      {get_move_capture(move)}       {get_move_double(move)}      {get_move_enpassant(move)}       {get_move_castling(move)}"
      print(move_str)

   # prints move list
   def print_move_list(self) -> None:
      if not self.ML.count:
         print("No Moves")
         return

      print("move   piece capture double enpass castling")
      # loop over moves
      for i in range(self.ML.count):
         move = self.ML.moves[i]
         move_str = f"{CORD_TO_SQUARE[get_move_source(move)]}{CORD_TO_SQUARE[get_move_target(move)]}{promoted_pieces[get_move_promo(move)]}"
         move_str += f"   {get_move_piece(move)}      {get_move_capture(move)}       {get_move_double(move)}      {get_move_enpassant(move)}       {get_move_castling(move)}"
         
         print(move_str)
      
      print()
      print(f"Total number of moves: {self.ML.count}")

   # prints ascii board representation
   def print_board(self) -> None:
      print()
      for rank in range(8):
         rank_str = ""
         for file in range(8):

               square = rank * 8 + file

               if not file:
                  rank_str += " " + str(8-rank) + " "

               piece = -1

               # loop over all bbs
               for bb in range(12):
                  if get_bit(self.bitboards[bb], square):
                     piece = bb

               p = "." if piece == -1 else ascii_pieces[piece]
               rank_str += p + " "
         print(rank_str)
      print("   a b c d e f g h")
      print()

      s = "White" if not self.side else "Black"
      print(f"   Side:     {s}")
      print(f"   Enpass:      {CORD_TO_SQUARE[self.enpassant]}")
      c1 = "K" if self.castling & wk else "-"
      c2 = "Q" if self.castling & wq else "-"
      c3 = "k" if self.castling & bk else "-"
      c4 = "q" if self.castling & bq else "-"
      print(f"   Castling:  {c1+c2+c3+c4}")