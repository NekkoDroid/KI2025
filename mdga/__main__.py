from concurrent import futures
from contextlib import contextmanager
import itertools
from pathlib import Path
from random import Random
import sys
import time
from typing import Callable, Iterator
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from collections import deque

import numpy as np
import torch

from mdga.game import Game
from mdga.neural_network import NeuralNetworkPlayer, NeuralNetworkPopulation
from mdga.player import FurthestPlayer, KnockoutPlayer, NearestPlayer, RandomPlayer, Player, SmartPlayer


random = Random()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NN_SUPERVISED_DIR = Path("./models-supervised/")
NN_GENETIC_DIR = Path("./models-genetic/")


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


def make_plots() -> tuple[Figure, tuple[Axes, Axes]]:
    fig = plt.figure(figsize=(20, 12))
    return fig, tuple(fig.subplots(ncols=2))


def train_supervised() -> None:
    STAGNATION_PATIENCE = 300
    GAMES_PER_GENERATION = 1000

    nn_player = NeuralNetworkPlayer(device, random)

    if models := sorted(NN_SUPERVISED_DIR.glob("*.pt"), key=lambda file: int(file.stem)):
        nn_player.load(models[-1])

    def save(epoch: int) -> None:
        NN_SUPERVISED_DIR.mkdir(exist_ok=True, parents=True)
        fig.savefig(NN_SUPERVISED_DIR / f"{epoch}.svg")
        nn_player.save(NN_SUPERVISED_DIR / f"{epoch}.pt")

    PLAYERS: list[Player] = [
        nn_player,
        FurthestPlayer(),
        NearestPlayer(),
        RandomPlayer(random),
        KnockoutPlayer(FurthestPlayer()),
        KnockoutPlayer(NearestPlayer()),
        KnockoutPlayer(RandomPlayer(random)),
        SmartPlayer(),
    ]

    histories: dict[Player, deque[bool]] = {player: deque(maxlen=1000) for player in PLAYERS}
    averages: dict[Player, list[float]] = {player: list() for player in PLAYERS}

    training_losses: list[float] = list()
    testing_losses: list[float] = list()

    fig, (plot1, plot2) = make_plots()
    def update_plots() -> None:
        plot1.clear()
        plot1.set_title("Average winrate of players")
        plot1.set_xlabel("Games played")
        plot1.set_ylabel("Winrate")

        for player in PLAYERS:
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
        plot2.set_title("Losses of each epoch")
        plot2.set_xlabel("Epoch")
        plot2.set_ylabel("Loss")

        plot2.plot(training_losses, label="Training losses")
        plot2.plot(testing_losses, label="Testing losses")

        plot2.legend()
        plot2.grid()

        fig.tight_layout()

    for epoch in itertools.count():
        if not plt.fignum_exists(fig.number):
            break

        game_decisions: list[tuple[np.ndarray, int]] = list()
        for game_index in range(GAMES_PER_GENERATION):
            if not plt.fignum_exists(fig.number):
                break

            game = Game(
                *random.sample(PLAYERS, k=4),
                random=random,
            )

            winner = game.play()
            game_decisions += winner.decisions

            for player in game.players:
                histories[player].append(winner == player)
                averages[player].append(sum(histories[player]) / len(histories[player]))
                player.decisions.clear()

            update_plots()
            plt.pause(0.01)

        game_decisions_count = len(game_decisions)
        game_decisions_split = int(game_decisions_count * 0.8)

        BATCH_SIZE = 512
        training_loss = nn_player.train(game_decisions[:game_decisions_split], BATCH_SIZE)
        testing_loss = nn_player.test(game_decisions[game_decisions_split:], BATCH_SIZE)
        training_losses.append(training_loss)
        testing_losses.append(testing_loss)

        print(" | ".join([
            f"Epoch: {epoch:>3}",
            f"Game data size: {game_decisions_count:>6}",
            f"Training loss: {training_loss}",
            f"Testing loss: {testing_loss}",
        ]))

        update_plots()
        plt.pause(0.01)
        save(epoch)


def train_genetic() -> None:
    POPULATION_SIZE = 32
    MUTATION_RATE = 0.1
    FITNESS_AVERAGE = 100

    nn_population = NeuralNetworkPopulation(POPULATION_SIZE, device, random)

    for i, file in enumerate(NN_GENETIC_DIR.glob("*.pt")):
        nn_population[i].load(file)

    def save(epoch: int) -> None:
        (dir := NN_GENETIC_DIR / str(epoch)).mkdir(exist_ok=True, parents=True)

        fig.savefig(dir / f"fig.svg")
        for i, player in enumerate(nn_population.population):
            player.save(dir / f"{i}.pt")

    PLAYERS: list[Player] = [
        FurthestPlayer(),
        NearestPlayer(),
        RandomPlayer(random),
        KnockoutPlayer(FurthestPlayer()),
        KnockoutPlayer(NearestPlayer()),
        KnockoutPlayer(RandomPlayer(random)),
        SmartPlayer(),
    ]

    best_fitness: list[float] = list()
    worst_fitness: list[float] = list()
    average_fitness: list[float] = list()
    median_fitness: list[float] = list()

    best_winrate: list[float] = list()
    worst_winrate: list[float] = list()
    average_winrate: list[float] = list()
    median_winrate: list[float] = list()

    fig, (plot1, plot2) = make_plots()
    def update_plots() -> None:
        plot1.clear()
        plot1.set_title("Fitness of each generation")
        plot1.set_xlabel("Generation")
        plot1.set_ylabel("Fitness")

        plot1.plot(best_fitness, label="Best fitness")
        plot1.plot(worst_fitness, label="Worst fitness")
        plot1.plot(average_fitness, label="Average fitness")
        plot1.plot(median_fitness, label="Median fitness")

        plot1.legend()
        plot1.grid()

        plot2.clear()
        plot2.set_title("Winrate of each generation")
        plot2.set_xlabel("Generation")
        plot2.set_ylabel("Winrate")

        plot2.plot(best_winrate, label="Best winrate")
        plot2.plot(worst_winrate, label="Worst winrate")
        plot2.plot(average_winrate, label="Average winrate")
        plot2.plot(median_winrate, label="Median winrate")

        plot2.legend()
        plot2.grid()

        fig.tight_layout()


    with futures.ThreadPoolExecutor() as executor:
        for epoch in itertools.count():
            if not plt.fignum_exists(fig.number):
                break

            def play_randoms(player: NeuralNetworkPlayer) -> Game:
                game = Game(
                    player,
                    *random.sample(PLAYERS, k=3),
                    random=random,
                )
                game.play()

                for p in game.players:
                    p.decisions.clear()

                return game

            FITNESS_AVERAGE = 100
            with print_duration("Determining performance took {} seconds"):
                winrate = nn_population.evaluate(play_randoms, FITNESS_AVERAGE, executor)

            best_winrate.append(max(winrate))
            worst_winrate.append(min(winrate))
            average_winrate.append(sum(winrate) / len(winrate))
            median_winrate.append(sorted(winrate)[len(winrate) // 2])

            update_plots()
            plt.pause(1)

            with print_duration("Determing fitness took {} seconds"):
                fitness = nn_population.fitnesses(FITNESS_AVERAGE, executor)

            best_fitness.append(max(fitness))
            worst_fitness.append(min(fitness))
            average_fitness.append(sum(fitness) / len(fitness))
            median_fitness.append(sorted(fitness)[len(fitness) // 2])

            update_plots()
            plt.pause(1)
            save(epoch)

            with print_duration("Creating new generation took {} seconds"):
                nn_population.next_generation(fitness, MUTATION_RATE)


def main() -> None:
    if len(sys.argv) <= 1:
        return train_supervised()

    match sys.argv[1]:
        case "supervised": train_supervised()
        case "genetic": train_genetic()


if __name__ == "__main__":
    sys.exit(main())
