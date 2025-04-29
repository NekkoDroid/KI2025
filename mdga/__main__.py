import sys

from mdga.game import Game
from mdga.player import MoveRandomPlayer


def main() -> None:
    Game(
        MoveRandomPlayer(),
        MoveRandomPlayer(),
        MoveRandomPlayer(),
        MoveRandomPlayer(),
    ).play()


if __name__ == "__main__":
    sys.exit(main())
