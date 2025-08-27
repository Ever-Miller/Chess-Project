import cv2 as cv
import numpy as np
import algorithms
from algorithms import PositionWeightVectors

# ranges in the array
p_maps = {
    "pawn": (14,  78),
    "knight": (78, 142),
    "bishop": (142, 206),
    "rook": (206, 270),
    "queen": (270, 334),
    "king": (334, 398)
}


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
def show_piece_map(piece: int, stage: int) -> None:
    while True:
        key = cv.waitKey(200)
        vecs = PositionWeightVectors.load_all(model="V2")
        start, end = p_maps[piece]
        raw = vecs.weights[stage][start:end]
        kmap = raw.reshape((8, 8))

        gray = cv.normalize(kmap, None, alpha=0, beta=255,
                        norm_type=cv.NORM_MINMAX).astype(np.uint8)

        heat8 = cv.applyColorMap(gray, cv.COLORMAP_HOT)

        heat800 = cv.resize(heat8, (800, 800), interpolation=cv.INTER_NEAREST)

        bar = np.zeros((256, 20, 3), dtype=np.uint8)
        for i in range(256):
            bar[255 - i, :] = cv.applyColorMap(
                np.full((1,1), i, dtype=np.uint8),
                cv.COLORMAP_HOT
            )[0,0]

        bar = cv.resize(bar, (20, 800), interpolation=cv.INTER_NEAREST)

        img = np.zeros((1000, 1000, 3), np.uint8)
        img[50:850, 50:850] = heat800
        img[50:850, 860:880] = bar

        cv.rectangle(img, (40, 40), (860, 860), (154, 203, 227), 5)
        cv.putText(img, "High", (885, 80),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
        cv.putText(img, "Low", (885, 840),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)


        cv.imshow("piece map", img)
        if key == 27:
            cv.destroyAllWindows()
            break


def main() -> None:
    piece = input("Which Piece Would You Like to See? (pawn, knight, bishop, rook, queen, king) ")
    stage = int(input("Early game (0) mid game (1) or late game (2) "))
    show_piece_map(piece, stage)


main()