from concurrent import futures
from contextlib import contextmanager
import itertools
from pathlib import Path
from random import Random
import sys
import time
from typing import Callable, Iterator
import matplotlib.pyplot as plt
from collections import deque

import numpy as np
import torch

from mdga.game import Game
from mdga.neural_network import NeuralNetworkPlayer, NeuralNetworkPopulation
from mdga.player import FurthestPlayer, KnockoutPlayer, NearestPlayer, RandomPlayer, Player, SmartPlayer


NN_MODELS_DIR = Path("./models/")
NN_GENETIC_DIR = Path("./mdga-genetic.pt")


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

    if models := sorted(NN_MODELS_DIR.glob("*.pt"), key=lambda file: int(file.stem)):
        nn_player.load(models[-1])

    for file in NN_MODELS_DIR.glob("*"):
        file.unlink()

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

    training_losses: list[float] = list()
    testing_losses: list[float] = list()

    STAGNATION_PATIENCE = 300

    fig = plt.figure(figsize=(20, 12))
    (plot1, plot2) = fig.subplots(ncols=2)

    def save(iter: int) -> None:
        NN_MODELS_DIR.mkdir(exist_ok=True)
        fig.savefig(NN_MODELS_DIR / f"{iter}.svg")
        nn_player.save(NN_MODELS_DIR / f"{iter}.pt")

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
        plot2.set_title("Losses of each epoch")
        plot2.set_xlabel("Epoch")
        plot2.set_ylabel("Loss")

        plot2.plot(training_losses, label="Training losses")
        plot2.plot(testing_losses, label="Testing losses")

        plot2.legend()
        plot2.grid()

        fig.tight_layout()

    epoch = 1
    for epoch in itertools.count(start=epoch):
        if not plt.fignum_exists(fig.number):
            break

        game_decisions: list[tuple[np.ndarray, int]] = list()
        GAMES_PER_GENERATION = 1000

        for game_index in range(GAMES_PER_GENERATION):
            if not plt.fignum_exists(fig.number):
                break

            game = Game(
                *random.sample(PLAYER_TYPES, k=4),
                random=random,
            )

            winner = game.play()
            game_decisions += winner.decisions

            for player in game.players:
                histories[player].append(winner == player)
                averages[player].append(sum(histories[player]) / len(histories[player]))
                player.decisions.clear()

            update_plot()
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

        update_plot()
        plt.pause(0.01)
        save(epoch)

        #STAGNATION_THRESHOLD = 0.01
        #if has_stagnated(averages[nn_player], STAGNATION_THRESHOLD, STAGNATION_PATIENCE):
        #    break

    save(epoch)
    if not plt.fignum_exists(fig.number):
        return

    POPULATION_SIZE = 32
    MUTATION_RATE = 0.1

    population = NeuralNetworkPopulation(POPULATION_SIZE, device, random)
    for player in population:
        player.network.load_state_dict(nn_player.network.state_dict())
        player.mutate(MUTATION_RATE)

    with futures.ThreadPoolExecutor() as executor:
        epoch = 1
        for epoch in itertools.count(start=epoch):
            if not plt.fignum_exists(fig.number):
                break

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
    NN_GENETIC_DIR.mkdir(exist_ok=True)
    for i, player in enumerate(population.population):
        player.save(NN_GENETIC_DIR / f"{i}.pt")


if __name__ == "__main__":
    sys.exit(main())
