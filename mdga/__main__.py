from random import Random
import sys
from typing import Callable
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import Counter

from mdga.game import Game
from mdga.player import MoveFirstPlayer, MoveKnockoutPlayer, MoveLastPlayer, MoveRandomPlayer, Player


def update_plot(winners: Counter, fig: Figure, plot) -> None:
    winner_names = list(winners.keys())
    winner_count = list(winners.values())

    plot.clear()
    plot.bar(winner_names, winner_count)
    plot.set_title("Player Performance: Number of Wins by Player Type")
    plot.set_xlabel("Player Type")
    plot.set_ylabel("Number of Wins")
    plot.set_xticks(range(len(winner_names)))
    plot.set_xticklabels(winner_names, rotation=20)
    plot.set_ylim(0, max(winner_count) * 1.1 if winner_count else 1)
    fig.tight_layout()



def main() -> None:
    random = Random()

    PLAYER_TYPES: list[Callable[[], Player]] = [
        MoveFirstPlayer,
        MoveLastPlayer,
        lambda: MoveRandomPlayer(random),
        lambda: MoveKnockoutPlayer(random),
    ]

    # List of rounds with the player types and the winner index
    winners = Counter()

    fig, plot = plt.subplots()
    while plt.fignum_exists(fig.number):
        game = Game(
            random.choice(PLAYER_TYPES)(),
            random.choice(PLAYER_TYPES)(),
            random.choice(PLAYER_TYPES)(),
            random.choice(PLAYER_TYPES)(),
            random=random,
        )

        winner = game.play()
        winners[str(winner)] += 1

        update_plot(winners, fig, plot)
        plt.pause(0.1)


if __name__ == "__main__":
    sys.exit(main())
