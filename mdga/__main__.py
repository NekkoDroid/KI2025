from pathlib import Path
from random import Random
import sys
import matplotlib.pyplot as plt
from collections import deque

import torch

from mdga.game import Game
from mdga.neural_network import NeuralNetworkPlayer, NeuralNetworkPopulation
from mdga.player import FurthestPlayer, KnockoutPlayer, NearestPlayer, RandomPlayer, Player


NN_MODEL_SUPERVISED_PATH = Path("./mdga-supervised.pt")
NN_MODEL_GENETIC_PATH = Path("./mdga-genetic.pt")


def has_stagnated(history: list[float], threshold: float, patience: int = 1000) -> bool:
    if len(history) < patience:
        return False

    recent_performance = history[-patience:]
    return (max(recent_performance) - min(recent_performance)) < threshold


def main() -> None:
    random = Random()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_player = NeuralNetworkPlayer(device, random)
    if NN_MODEL_SUPERVISED_PATH.exists():
        nn_player.load(NN_MODEL_SUPERVISED_PATH)

    PLAYER_TYPES: list[Player] = [
        nn_player,
        FurthestPlayer(),
        NearestPlayer(),
        RandomPlayer(random),
        KnockoutPlayer(FurthestPlayer()),
        KnockoutPlayer(NearestPlayer()),
        KnockoutPlayer(RandomPlayer(random)),
    ]

    histories: dict[Player, deque[bool]] = {player: deque(maxlen=1000) for player in PLAYER_TYPES}
    averages: dict[Player, list[float]] = {player: list() for player in PLAYER_TYPES}

    best_fitness: list[float] = list()
    worst_fitness: list[float] = list()
    average_fitness: list[float] = list()
    median_fitness: list[float] = list()

    fig, (plot1, plot2) = plt.subplots(ncols=2)

    def update_plot() -> None:
        plot1.clear()
        plot1.set_title("Average winrate of players")
        plot1.set_xlabel("Games played")
        plot1.set_ylabel("Winrate")

        for player in PLAYER_TYPES:
            plot1.plot(averages[player], label=str(player))

        plot1.set_ylim(0, 1)
        plot1.set_xlim(
            min(map(len, averages.values())) * 0.9,
            max(map(len, averages.values())) * 1.1,
        )
        plot1.legend()
        plot1.grid()

        plot2.clear()
        plot2.set_title("Fitness of each generation")
        plot2.set_xlabel("Generation")
        plot2.set_ylabel("Winrate")

        plot2.plot(best_fitness, label="Best fitness")
        plot2.plot(worst_fitness, label="Worst fitness")
        plot2.plot(average_fitness, label="Average fitness")
        plot2.plot(median_fitness, label="Median fitness")

        plot2.legend()
        plot2.grid()

        fig.tight_layout()
        plt.pause(0.1)


    while plt.fignum_exists(fig.number):
        game = Game(
            *random.choices(PLAYER_TYPES, k=4),
            random=random,
        )

        winner = game.play()

        for player in game.players:
            histories[player].append(winner == player)
            averages[player].append(sum(histories[player]) / len(histories[player]))

        update_plot()

        nn_player.learn(winner.decisions)
        for player in game.players:
            player.decisions.clear()

        STAGNATION_THRESHOLD = 0.03
        STAGNATION_PATIENCE = 300
        if has_stagnated(averages[nn_player], STAGNATION_THRESHOLD, STAGNATION_PATIENCE):
            break

    nn_player.save(NN_MODEL_SUPERVISED_PATH)

    POPULATION_SIZE = 100
    population = NeuralNetworkPopulation(POPULATION_SIZE, device, random)
    for player in population:
        player.network.load_state_dict(nn_player.network.state_dict())

    while plt.fignum_exists(fig.number):
        def play_game(player: NeuralNetworkPlayer) -> Game:
            game = Game(
                player,
                *random.choices(PLAYER_TYPES, k=3),
                random=random,
            )

            game.play()
            for p in game.players:
                p.decisions.clear()

            plt.pause(0.1)
            return game

        FITNESS_AVERAGE = 100
        fitness = [player.fitness(play_game, FITNESS_AVERAGE) for player in population]
        population.next_generation(fitness, mutation_rate=0.1)

        best_fitness.append(max(fitness))
        worst_fitness.append(min(fitness))
        average_fitness.append(sum(fitness) / len(fitness))
        median_fitness.append(sorted(fitness)[len(fitness) // 2])

        update_plot()

    # Save the best player of the generation
    NN_MODEL_GENETIC_PATH.mkdir(exist_ok=True)
    for i, player in enumerate(population.population):
        player.save(NN_MODEL_GENETIC_PATH / f"{i}.pt")


if __name__ == "__main__":
    sys.exit(main())
