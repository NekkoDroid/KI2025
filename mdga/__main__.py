from pathlib import Path
from random import Random
import sys
import matplotlib.pyplot as plt
from collections import deque

from mdga.game import Game
from mdga.neural_network import NeuralNetworkPlayer
from mdga.player import FurthestPlayer, KnockoutPlayer, NearestPlayer, RandomPlayer, Player


NN_MODEL_PATH = Path("./mdga.pt")


def main() -> None:
    random = Random()

    nn_player = NeuralNetworkPlayer(random)
    if NN_MODEL_PATH.exists():
        nn_player.load(NN_MODEL_PATH)

    PLAYER_TYPES: list[Player] = [
        nn_player,
        FurthestPlayer(),
        NearestPlayer(),
        RandomPlayer(random),
        KnockoutPlayer(FurthestPlayer()),
        KnockoutPlayer(NearestPlayer()),
        KnockoutPlayer(RandomPlayer(random)),
    ]

    histories: dict[Player, deque[bool]] = {
        player: deque(maxlen=1000) for player in PLAYER_TYPES
    }

    averages: dict[Player, list[float]] = {
        player: list() for player in PLAYER_TYPES
    }

    fig, plot = plt.subplots()
    while plt.fignum_exists(fig.number):
        game = Game(
            *random.choices(PLAYER_TYPES, k=4),
            random=random,
        )

        winner = game.play()

        for player in game.players:
            histories[player].append(winner == player)
            averages[player].append(sum(histories[player]) / len(histories[player]))

        plot.clear()
        plot.set_title("Average winrate of players")
        plot.set_xlabel("Games played")
        plot.set_ylabel("Winrate")

        for player in PLAYER_TYPES:
            plot.plot(averages[player], label=str(player))

        plot.set_ylim(0, 1)
        plot.set_xlim(
            min(map(len, averages.values())) * 0.9,
            max(map(len, averages.values())) * 1.1,
        )
        plot.legend()
        plot.grid()

        fig.tight_layout()
        plt.pause(0.1)

        nn_player.learn(winner.decisions)
        for player in game.players:
            player.decisions.clear()

    nn_player.save(NN_MODEL_PATH)


if __name__ == "__main__":
    sys.exit(main())
