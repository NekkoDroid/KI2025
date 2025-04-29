from abc import ABC, abstractmethod
from random import Random

from mdga.board import Board, InvalidMoveError, Piece


class Player(ABC):
    @abstractmethod
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        raise NotImplementedError()

    def valid_moves(self, board: Board, id: int, roll: int) -> tuple[Piece, ...]:
        def is_valid_move(piece: Piece) -> bool:
            try:
                board.simulate_move(piece, roll)
                return True
            except InvalidMoveError:
                return False

        pieces = board.filter(id=id)
        pieces = filter(is_valid_move, pieces)
        return tuple(pieces)


class MoveFirstPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        raise NotImplementedError()


class MoveLastPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        raise NotImplementedError()


class MoveRandomPlayer(Player):
    random: Random

    def __init__(self, random: Random = Random()) -> None:
        self.random = random

    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        return self.random.choice(self.valid_moves(board, id, roll))


class NeuralNetworkPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        raise NotImplementedError()
