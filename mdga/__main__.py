from random import Random
import sys
from typing import Callable

from mdga.game import Game
from mdga.player import MoveFirstPlayer, MoveKnockoutPlayer, MoveLastPlayer, MoveRandomPlayer, Player


def main() -> None:
    random = Random()

    PLAYER_TYPES: list[Callable[[], Player]] = [
        MoveFirstPlayer,
        MoveLastPlayer,
        lambda: MoveRandomPlayer(random),
        lambda: MoveKnockoutPlayer(MoveFirstPlayer(), random),
        lambda: MoveKnockoutPlayer(MoveLastPlayer(), random),
        lambda: MoveKnockoutPlayer(MoveRandomPlayer(random), random),
    ]

    Game(
        random.choice(PLAYER_TYPES)(),
        random.choice(PLAYER_TYPES)(),
        random.choice(PLAYER_TYPES)(),
        random.choice(PLAYER_TYPES)(),
        random=random
    ).play()


if __name__ == "__main__":
    sys.exit(main())
