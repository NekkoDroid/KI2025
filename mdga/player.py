from abc import ABC, abstractmethod
from random import Random

from mdga.board import Board, InvalidMoveError, Piece, PieceState


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

    def __str__(self) -> str:
        return self.__class__.__name__


class MoveFirstPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        return sorted(self.valid_moves(board, id, roll), key=board.distance)[-1]


class MoveLastPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        return sorted(self.valid_moves(board, id, roll), key=board.distance)[0]


class MoveRandomPlayer(Player):
    random: Random

    def __init__(self, random: Random = Random()) -> None:
        self.random = random

    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        return self.random.choice(self.valid_moves(board, id, roll))


class MoveKnockoutPlayer(MoveRandomPlayer):
    parent: Player
    random: Random

    def __init__(self, parent: Player, random: Random = Random()) -> None:
        self.parent = parent
        self.random = random

    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        def is_knockout(piece: Piece) -> bool:
            if piece.state != PieceState.transit:
                return False

            # We can safely assume that none of the pieces on the target position are from the current ID
            return any(board.filter(position=board.simulate_move(piece, roll)))

        valid_moves = self.valid_moves(board, id, roll)
        knockout_moves = tuple(filter(is_knockout, valid_moves))

        try:
            return self.random.choice(knockout_moves)
        except IndexError:
            return self.parent.select_move(board, id, roll)

    def __str__(self) -> str:
        return f"{super().__str__()}({self.parent})"


class NeuralNetworkPlayer(Player):
    def select_move(self, board: Board, id: int, roll: int) -> Piece:
        raise NotImplementedError()
