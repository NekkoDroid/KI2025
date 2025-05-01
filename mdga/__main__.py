from random import Random
import sys
from typing import Callable
import matplotlib.pyplot as plt
from collections import Counter

from mdga.game import Game
from mdga.player import MoveFirstPlayer, MoveKnockoutPlayer, MoveLastPlayer, MoveRandomPlayer, Player, MoveRulesPlayer


def plot_counter(counter: Counter, title: str, plot) -> None:
    names = list(counter.keys())
    count = list(counter.values())

    plot.clear()
    plot.bar(names, count)
    plot.set_title(title)
    plot.set_xticks(range(len(names)))
    plot.set_xticklabels(names, rotation=20)
    plot.set_ylim(0, max(count) * 1.1 if count else 1)


def main() -> None:
    random = Random()

    PLAYER_TYPES: list[Callable[[], Player]] = [
        MoveFirstPlayer,
        MoveLastPlayer,
        MoveRulesPlayer,
        lambda: MoveRandomPlayer(random),
        lambda: MoveKnockoutPlayer(random),
    ]

    # List of rounds with the player types and the winner index
    games_won = Counter()
    games_played = Counter()

    fig, (plot_games_played, plot_games_won) = plt.subplots(ncols=2)
    while plt.fignum_exists(fig.number):
        game = Game(
            random.choice(PLAYER_TYPES)(),
            random.choice(PLAYER_TYPES)(),
            random.choice(PLAYER_TYPES)(),
            random.choice(PLAYER_TYPES)(),
            random=random,
        )

        winner = game.play()

        games_played.update(map(str, game.players))
        games_won.update([str(winner)])

        plot_counter(games_played, "Player Performance: Number of Games Played by Player Type", plot_games_played)
        plot_counter(games_won, "Player Performance: Number of Wins by Player Type", plot_games_won)
        fig.tight_layout()
        plt.pause(0.1)


if __name__ == "__main__":
    sys.exit(main())
