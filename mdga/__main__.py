from concurrent import futures
from contextlib import contextmanager
from pathlib import Path
from random import Random
import sys
import time
from typing import Callable, Iterator
import matplotlib.pyplot as plt
from collections import deque

import torch

from mdga.game import Game
from mdga.neural_network import NeuralNetworkPlayer, NeuralNetworkPopulation
from mdga.player import FurthestPlayer, KnockoutPlayer, NearestPlayer, RandomPlayer, Player, SmartPlayer


NN_MODEL_SUPERVISED_PATH = Path("./mdga-supervised.pt")
NN_MODEL_GENETIC_PATH = Path("./mdga-genetic.pt")


@contextmanager
def print_duration(message: str = "Time elapsed: {}", func: Callable[[str], None] = print) -> Iterator[None]:
    start = time.time()
    try:
        yield
    finally:
        func(message.format(time.time() - start))


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
        SmartPlayer(),
    ]

    histories: dict[Player, deque[bool]] = {player: deque(maxlen=1000) for player in PLAYER_TYPES}
    averages: dict[Player, list[float]] = {player: list() for player in PLAYER_TYPES}

    best_fitness: list[float] = list()
    worst_fitness: list[float] = list()
    average_fitness: list[float] = list()
    median_fitness: list[float] = list()

    best_winrate: list[float] = list()
    worst_winrate: list[float] = list()
    average_winrate: list[float] = list()
    median_winrate: list[float] = list()

    STAGNATION_PATIENCE = 300

    fig, (plot1, plot2) = plt.subplots(ncols=2)

    def update_plot() -> None:
        plot1.clear()
        plot1.set_title("Average winrate of players")
        plot1.set_xlabel("Games played")
        plot1.set_ylabel("Winrate")

        for player in PLAYER_TYPES:
            plot1.plot(averages[player], label=str(player))

        if averages:
            plot_min = min(map(len, averages.values()))
            plot_min = max(0, plot_min - STAGNATION_PATIENCE) # Show X of the previous values

            plot_max = max(map(len, averages.values()))
            plot_max = max(1, plot_max + (plot_max - plot_min) * 0.1) # Show 10% forwards

            plot1.set_ylim(0, 1)
            plot1.set_xlim(plot_min, plot_max)

        plot1.legend()
        plot1.grid()

        plot2.clear()
        plot2.set_title("Fitness of each generation")
        plot2.set_xlabel("Generation")
        plot2.set_ylabel("Fitness")

        plot2.plot(best_fitness, label="Best fitness")
        plot2.plot(worst_fitness, label="Worst fitness")
        plot2.plot(average_fitness, label="Average fitness")
        plot2.plot(median_fitness, label="Median fitness")

        plot2.legend()
        plot2.grid()

        fig.tight_layout()


    while plt.fignum_exists(fig.number):
        game = Game(
            *random.sample(PLAYER_TYPES, k=4),
            random=random,
        )

        winner = game.play()

        for player in game.players:
            histories[player].append(winner == player)
            averages[player].append(sum(histories[player]) / len(histories[player]))

        update_plot()
        plt.pause(0.01)

        nn_player.learn(winner.decisions)
        for player in game.players:
            player.decisions.clear()

        #STAGNATION_THRESHOLD = 0.01
        #if has_stagnated(averages[nn_player], STAGNATION_THRESHOLD, STAGNATION_PATIENCE):
        #    break

    nn_player.save(NN_MODEL_SUPERVISED_PATH)

    POPULATION_SIZE = 32
    MUTATION_RATE = 0.1

    population = NeuralNetworkPopulation(POPULATION_SIZE, device, random)
    for player in population:
        player.network.load_state_dict(nn_player.network.state_dict())
        player.mutate(MUTATION_RATE)

    with futures.ThreadPoolExecutor() as executor:
        while plt.fignum_exists(fig.number):
            def play_randoms(player: NeuralNetworkPlayer) -> Game:
                game = Game(
                    player,
                    *random.sample(PLAYER_TYPES, k=3),
                    random=random,
                )
                game.play()

                for p in game.players:
                    p.decisions.clear()

                return game

            FITNESS_AVERAGE = 100
            with print_duration("Determining performance took {} seconds"):
                winrate = population.evaluate(play_randoms, FITNESS_AVERAGE, executor)

            best_winrate.append(max(winrate))
            worst_winrate.append(min(winrate))
            average_winrate.append(sum(winrate) / len(winrate))
            median_winrate.append(sorted(winrate)[len(winrate) // 2])

            update_plot()
            plt.pause(1)

            with print_duration("Determing fitness took {} seconds"):
                fitness = population.fitnesses(FITNESS_AVERAGE, executor)

            best_fitness.append(max(fitness))
            worst_fitness.append(min(fitness))
            average_fitness.append(sum(fitness) / len(fitness))
            median_fitness.append(sorted(fitness)[len(fitness) // 2])

            update_plot()
            plt.pause(1)

            with print_duration("Creating new generation took {} seconds"):
                population.next_generation(fitness, MUTATION_RATE)

    # Save the best player of the generation
    NN_MODEL_GENETIC_PATH.mkdir(exist_ok=True)
    for i, player in enumerate(population.population):
        player.save(NN_MODEL_GENETIC_PATH / f"{i}.pt")


if __name__ == "__main__":
    sys.exit(main())
