import logging
import os
from random import Random
import sys
from typing import Optional
from mdga.board import MAX_PLAYERS, MAX_ROLL, MIN_PLAYERS, MIN_ROLL, Board, PieceState
from mdga.player import Player


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(os.getenv("LOG_LEVEL", "critical").upper())
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


class Game:
    board: Board
    winner: Optional[Player]
    players: tuple[Player, ...]
    random: Random

    # NOTE: this default random parameter is shared between all instances of Game
    # Usually this isn't desirable, but in this case it doesn't matter
    def __init__(self, *players: Player, random: Random = Random()) -> None:
        # We only support 2-4 players, the board should be able to handle any number of players
        assert MIN_PLAYERS <= len(players) <= MAX_PLAYERS

        self.board = Board()
        self.winner = None

        self.players = players
        self.random = random

    def play(self) -> Player:
        LOGGER.info("Players:")
        for id, player in enumerate(self.players):
            LOGGER.info(f"  {id}: {player}")

        while self.winner is None:
            self.play_round()

        LOGGER.info(f"We have a winner: Player {self.players.index(self.winner)}")
        return self.winner

    def play_round(self) -> None:
        for id, player in enumerate(self.players):
            # Sanity check that we don't play a round when someone already won the game
            if self.winner is not None:
                break

            # NOTE: For now we don't enforce that all 6 rolls need to clear out the home area this is to allow a bit
            # more freedom to the AI and see how it evolves. At a later point it might make sense to implement that we
            # force move from the home area, but that will be decided later on.
            roll = self.random.randint(MIN_ROLL, MAX_ROLL)

            try:
                target = player.move(self.board, id, roll)
                assert target.id == id

                LOGGER.info(f"Player {id} moving piece from {target.position} by {roll}")
                self.board.move(target, roll)

            except LookupError:
                LOGGER.warning(f"Player {id} has no available moves, passing.")

            # When all pieces of a specific ID are in the target state we have a winner
            if all(piece.state == PieceState.target for piece in self.board.filter(id=id)):
                self.winner = player

            self.board.print(0, LOGGER.debug)
