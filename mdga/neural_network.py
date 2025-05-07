from collections import Counter
from concurrent import futures
import itertools
from pathlib import Path
from random import Random
from typing import Callable, cast
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from mdga.board import MAX_PLAYERS, MAX_ROLL, PIECES_PER_PLAYER, Board, Piece, PieceState
from mdga.game import Game
from mdga.player import BOARD_STATE_FEATURES, BOARD_STATE_POSITIONS, Player


class Unsqueeze(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class NeuralNetwork(nn.Sequential):
    def __init__(self) -> None:
        # We need more than MAX_ROLL since home and target fields are interleaved into the fields
        MAX_MOVE_DISTANCE = MAX_ROLL + PIECES_PER_PLAYER * 2 + 1

        super().__init__(
            Unsqueeze(dim=1), # Add channel dimension
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                # Combine all features down to 1 and take into account all posssible move distances
                kernel_size=(BOARD_STATE_FEATURES, MAX_MOVE_DISTANCE),
                padding=(0, MAX_MOVE_DISTANCE // 2),
                padding_mode="circular",
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, MAX_MOVE_DISTANCE),
                padding=(0, MAX_MOVE_DISTANCE // 2),
                padding_mode="circular",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Downsample to half the width
            nn.MaxPool2d(
                kernel_size=(1, 2),
                stride=(1, 2),
            ),
            nn.Flatten(),
            nn.Linear((BOARD_STATE_POSITIONS // 2) * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, PIECES_PER_PLAYER),
            nn.Softmax(dim=-1),
        )


class NeuralNetworkPlayer(Player):
    device: torch.device
    random: Random
    network: nn.Module
    optimizer: optim.Optimizer
    lossfunc: nn.Module

    def __init__(self, device: torch.device, random: Random = Random()) -> None:
        super().__init__()
        self.device = device
        self.random = random
        self.network = NeuralNetwork().to(self.device)
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.95)
        self.lossfunc = nn.BCELoss().to(self.device)

    @torch.inference_mode()
    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        state = self.encode_move(board, id, roll)
        state = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)

        self.network.eval()
        probabilities = self.network(state)

        assert len(probabilities) == 1
        probabilities = probabilities[0]

        valid_moves = self.valid_moves(board, roll, pieces)
        valid_indices = [pieces.index(piece) for piece in valid_moves]

        if not valid_indices:
            raise LookupError("No valid moves available")

        valid_probabilities = [max(probabilities[i], 1e-8) for i in valid_indices]
        selected_index = self.random.choices(valid_indices, valid_probabilities)[0]
        return pieces[selected_index]

    def save(self, path: str | Path) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.network.load_state_dict(torch.load(path))

    def split_data(self, data: list[tuple[np.ndarray, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        states, targets = zip(*data)

        states = torch.stack([
            torch.tensor(
                state,
                dtype=torch.float32,
                device=self.device,
            ) for state in states
        ])

        targets = torch.stack([
            torch.tensor(
                    [float(target == i) for i in range(PIECES_PER_PLAYER)],
                    dtype=torch.float32,
                    device=self.device,
            ) for target in targets
        ])

        return states, targets

    def train(self, training_data: list[tuple[np.ndarray, int]], batch_size: int = 32) -> float:
        states, targets = self.split_data(training_data)
        dataset = TensorDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.train()
        total_loss = 0.0

        for states, targets in dataloader:
            outputs: torch.Tensor = self.network(states)
            loss: torch.Tensor = self.lossfunc(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()

        return total_loss / len(dataloader)

    @torch.inference_mode()
    def test(self, test_data: list[tuple[np.ndarray, int]], batch_size: int = 32) -> float:
        states, targets = self.split_data(test_data)
        dataset = TensorDataset(states, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.network.eval()
        total_loss = 0.0

        for states, targets in dataloader:
            outputs: torch.Tensor = self.network(states)
            loss: torch.Tensor = self.lossfunc(outputs, targets)
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def mutate(self, mutation_rate: float) -> None:
        state = self.network.state_dict()

        for key in state:
            if torch.is_floating_point(state[key]) and self.random.random() < mutation_rate:
                state[key] += torch.randn_like(state[key]) * 0.01

        self.network.load_state_dict(state)

    @classmethod
    def crossover(cls, device: torch.device, parents: list["NeuralNetworkPlayer"], random: Random = Random()) -> "NeuralNetworkPlayer":
        child = cls(device, random)
        state = child.network.state_dict()

        for key in state:
            state[key] = child.random.choice(parents).network.state_dict()[key]

        child.network.load_state_dict(state)
        return child


class NeuralNetworkPopulation:
    device: torch.device
    random: Random
    population: list[NeuralNetworkPlayer]

    def __init__(self, population: int, device: torch.device, random: Random = Random()) -> None:
        self.device = device
        self.random = random
        assert population % MAX_PLAYERS == 0  # Ensure that we can fill a game with players
        self.population = [NeuralNetworkPlayer(device, random) for _ in range(population)]

    def __len__(self) -> int:
        return len(self.population)

    def __getitem__(self, index: int) -> NeuralNetworkPlayer:
        return self.population[index]

    def random_player(self) -> NeuralNetworkPlayer:
        return self.random.choice(self.population)

    def next_generation(self, scores: list[float], mutation_rate: float) -> None:
        population_size = len(self.population)
        assert population_size == len(scores)

        # We only want to keep the best half of the population
        # and create new children only inheriting from that remaining population
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_indices = sorted_indices[: population_size // 2]

        sorted_parents = [self.population[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        self.population = list(sorted_parents)

        # Add 10% new random individuals
        for _ in range(len(self.population) // 10):
            self.population.append(NeuralNetworkPlayer(self.device, self.random))

        # Fill the remaining with individuals inheriting from the parents
        for _ in range(population_size - len(self.population)):
            parents = self.random.choices(sorted_parents, sorted_scores, k=2)
            child = NeuralNetworkPlayer.crossover(self.device, parents, self.random)
            child.mutate(mutation_rate)
            self.population.append(child)

        assert len(self.population) == population_size

    def evaluate(self, play_game: Callable[[NeuralNetworkPlayer], Game], games: int, executor: futures.Executor) -> list[float]:
        def play_games(player: NeuralNetworkPlayer) -> float:
            wins = 0

            for _ in range(games):
                if play_game(player).play() == player:
                    wins += 1

            return wins / games

        return list(executor.map(play_games, self.population))

    def fitnesses(self, games: int, executor: futures.Executor) -> list[float]:
        wins = Counter()
        population = list(self.population)

        for _ in range(games):
            def play_game(players: tuple[NeuralNetworkPlayer, ...]) -> NeuralNetworkPlayer:
                game = Game(*players, random=self.random)
                winner = game.play()

                for player in game.players:
                    player.decisions.clear()

                return cast(NeuralNetworkPlayer, winner)

            self.random.shuffle(population)
            matchups = itertools.batched(population, MAX_PLAYERS)
            winners = executor.map(play_game, matchups)
            wins.update(winners)

        fitness = [0.0] * len(self.population)
        for i, player in enumerate(self.population):
            fitness[i] = wins[player] / games

        return fitness
