from abc import ABC, abstractmethod
from random import Random

from mdga.board import TRANSIT_FIELDS, Board, InvalidMoveError, Piece, PieceState


class Player(ABC):
    def move(self, board: Board, id: int, roll: int) -> Piece:
        pieces = board.filter(id=id)
        piece = self.select_move(board, pieces, roll)

        assert piece in pieces
        return piece

    @abstractmethod
    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        raise NotImplementedError()

    def valid_moves(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> tuple[Piece, ...]:
        return tuple(filter(lambda piece: self.is_valid_move(board, piece, roll), pieces))

    def knockout_moves(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> tuple[Piece, ...]:
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


class MoveFirstPlayer(Player):
    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        return sorted(self.valid_moves(board, pieces, roll), key=board.distance)[-1]


class MoveLastPlayer(Player):
    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        return sorted(self.valid_moves(board, pieces, roll), key=board.distance)[0]


class MoveRandomPlayer(Player):
    random: Random

    def __init__(self, random: Random = Random()) -> None:
        super().__init__()
        self.random = random

    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        return self.random.choice(self.valid_moves(board, pieces, roll))


class MoveKnockoutPlayer(MoveRandomPlayer):
    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        valid_moves = self.valid_moves(board, pieces, roll)
        knockout_moves = self.knockout_moves(board, valid_moves, roll)

        try:
            return self.random.choice(knockout_moves)
        except IndexError:
            return self.random.choice(valid_moves)


class MoveRulesPlayer(Player):
    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        moves = sorted(self.valid_moves(board, pieces, roll), key=board.distance)

        # Prefer to move pieces out of transit fields if possible
        for piece in moves:
            if board.distance(piece) > TRANSIT_FIELDS:
                return piece

        # If we can knock out a piece, do it
        for piece in moves:
            if self.is_knockout_move(board, piece, roll):
                return piece

        return moves[-1]


class NeuralNetworkPlayer(Player):
    def select_move(self, board: Board, pieces: tuple[Piece, ...], roll: int) -> Piece:
        raise NotImplementedError()
