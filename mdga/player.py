from abc import ABC, abstractmethod
from random import Random

from mdga.board import (
    MAX_PLAYERS,
    MAX_ROLL,
    PIECES_PER_PLAYER,
    TRANSIT_FIELDS,
    Board,
    InvalidMoveError,
    Piece,
    PieceState,
)

# Roll, one-hot encoded player ID, (transit, target, home) for each piece
ENCODED_MOVE_SIZE = 1 + MAX_PLAYERS + 3 * MAX_PLAYERS * PIECES_PER_PLAYER

def encode_move(board: Board, id: int, roll: int) -> list[float]:
    def encode_piece(piece: Piece) -> tuple[float, ...]:
        if piece.position is None:
            return 1, 0, 0

        if piece.position >= 0:
            # Normalize to the transit fields the range [0, 1]
            return 0, piece.position / (TRANSIT_FIELDS - 1), 0

        # Normalize the target areas to the range [0, 1]
        return 0, 0, -(piece.position + 1) / (PIECES_PER_PLAYER - 1)

    state = [roll / MAX_ROLL]
    state += [float(id == p) for p in range(MAX_PLAYERS)]

    for piece in board.pieces:
        state += encode_piece(piece)

    assert len(state) == ENCODED_MOVE_SIZE
    return state


class Player(ABC):
    decisions: list[tuple[list[float], int]]

    def __init__(self) -> None:
        super().__init__()
        self.decisions = list()

    def move(self, board: Board, id: int, roll: int) -> Piece:
        pieces = board.filter(id=id)
        piece = self.select_move(board, id, roll, pieces)

        self.decisions.append(
            (
                encode_move(board, id, roll),
                pieces.index(piece),
            )
        )

        assert piece in pieces
        return piece

    @abstractmethod
    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        raise NotImplementedError()

    def valid_moves(self, board: Board, roll: int, pieces: tuple[Piece, ...]) -> tuple[Piece, ...]:
        return tuple(filter(lambda piece: self.is_valid_move(board, piece, roll), pieces))

    def knockout_moves(self, board: Board, roll: int, pieces: tuple[Piece, ...]) -> tuple[Piece, ...]:
        return tuple(filter(lambda piece: self.is_knockout_move(board, piece, roll), pieces))

    def is_valid_move(self, board: Board, piece: Piece, roll: int) -> bool:
        try:
            board.simulate_move(piece, roll)
            return True
        except InvalidMoveError:
            return False

    def is_knockout_move(self, board: Board, piece: Piece, roll: int) -> bool:
        # Only valid moves from home to transit or transit to transit can knock out pieces
        if piece.state == PieceState.target:
            return False

        # We can safely assume that none of the pieces on the target position are from the current ID
        # since such a move would be an invalid move and shouldn't be passed to this function
        return any(board.filter(position=board.simulate_move(piece, roll)))

    def __str__(self) -> str:
        return self.__class__.__name__


class FurthestPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        return sorted(self.valid_moves(board, roll, pieces), key=board.distance)[-1]


class NearestPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        return sorted(self.valid_moves(board, roll, pieces), key=board.distance)[0]


class RandomPlayer(Player):
    random: Random

    def __init__(self, random: Random = Random()) -> None:
        super().__init__()
        self.random = random

    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        return self.random.choice(self.valid_moves(board, roll, pieces))


class KnockoutPlayer(Player):
    parent: Player

    def __init__(self, parent: Player) -> None:
        super().__init__()
        self.parent = parent

    def __str__(self) -> str:
        return f"{super().__str__()}({self.parent})"

    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        valid_moves = self.valid_moves(board, roll, pieces)
        knockout_moves = self.knockout_moves(board, roll, valid_moves)

        try:
            return self.parent.select_move(board, id, roll, knockout_moves)
        except IndexError:
            return self.parent.select_move(board, id, roll, valid_moves)


class NeuralNetworkPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        raise NotImplementedError()
