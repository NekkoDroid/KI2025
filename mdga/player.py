from abc import ABC, abstractmethod
from random import Random

import numpy as np

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

# Current player ID (one-hot) + roll (normalized) + Piece ID (one-hot)
BOARD_STATE_FEATURES = MAX_PLAYERS + 1 + MAX_PLAYERS
# Interleaved groups of (Target for player X, Home for player X, Transit fields from X-Y) * 4
BOARD_STATE_POSITIONS = 2 * (PIECES_PER_PLAYER * MAX_PLAYERS) + TRANSIT_FIELDS


class Player(ABC):
    decisions: list[tuple[np.ndarray, int]]

    def __init__(self) -> None:
        super().__init__()
        self.decisions = list()

    def move(self, board: Board, id: int, roll: int) -> Piece:
        pieces = board.filter(id=id)
        piece = self.select_move(board, id, roll, pieces)

        self.decisions.append(
            (
                self.encode_move(board, id, roll),
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

    def encode_move(self, board: Board, id: int, roll: int) -> np.ndarray:
        # We encode each representation of "home", "transit" and "target" seperately and
        # interleave each of them, so that there is a somewhat continuous representation of the board

        target = np.zeros((MAX_PLAYERS, PIECES_PER_PLAYER * MAX_PLAYERS), dtype=np.float32)
        for player_id in range(MAX_PLAYERS):
            for piece in board.filter(id=player_id, state=PieceState.target):
                assert piece.position is not None and piece.position < 0
                target[piece.id][piece.id * PIECES_PER_PLAYER - (piece.position + 1)] = 1.0
        target = np.split(target, MAX_PLAYERS, axis=1)

        home = np.zeros((MAX_PLAYERS, PIECES_PER_PLAYER * MAX_PLAYERS))
        for player_id in range(MAX_PLAYERS):
            for offset, piece in enumerate(board.filter(id=player_id, state=PieceState.home)):
                home[piece.id][piece.id * PIECES_PER_PLAYER + offset] = 1.0
        home = np.split(home, MAX_PLAYERS, axis=1)

        transit = np.zeros((MAX_PLAYERS, TRANSIT_FIELDS))
        for position in range(TRANSIT_FIELDS):
            for piece in board.filter(position=position):
                assert piece.position is not None and piece.position >= 0
                transit[piece.id][piece.position] = 1.0
        transit = np.split(transit, MAX_PLAYERS, axis=1)

        arrays = list()
        for player_id in range(MAX_PLAYERS):
            arrays.extend([
                home[player_id],
                transit[player_id],
                target[player_id],
            ])

        fields = np.concat(arrays, axis=1)
        assert fields.shape[0] == MAX_PLAYERS and fields.shape[1] == BOARD_STATE_POSITIONS

        # Current player ID (one-hot) + roll (normalized)
        remainder = np.zeros((MAX_PLAYERS + 1, BOARD_STATE_POSITIONS), dtype=np.float32)
        remainder[id, :] = 1.0
        remainder[MAX_PLAYERS, :] = roll / MAX_ROLL

        rv = np.concat([remainder, fields])
        assert rv.shape[0] == BOARD_STATE_FEATURES and rv.shape[1] == BOARD_STATE_POSITIONS
        return rv

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

