from pathlib import Path
from random import Random
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim

from mdga.board import PIECES_PER_PLAYER, Board, Piece
from mdga.player import ENCODED_MOVE_SIZE, Player, encode_move


class NeuralNetwork(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(ENCODED_MOVE_SIZE, 4096),
            nn.ReLU(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, PIECES_PER_PLAYER),
            nn.Softmax(dim=-1),
        )

class NeuralNetworkPlayer(Player):
    random: Random
    network: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module

    def __init__(self, random: Random = Random()) -> None:
        super().__init__()
        self.random = random
        self.network = NeuralNetwork()
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        state = encode_move(board, id, roll)
        state = torch.tensor(state, dtype=torch.float32)

        self.network.eval()
        probabilities = self.network(state)

        valid_moves = self.valid_moves(board, roll, pieces)
        valid_indices = [pieces.index(piece) for piece in valid_moves]

        try:
            best_move = max(valid_indices, key=lambda i: probabilities[i])
            return pieces[best_move]
        except ValueError:
            raise LookupError("No valid moves available")

    def save(self, path: str | Path) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.network.load_state_dict(torch.load(path))

    def learn(self, training_data: list[tuple[list[float], int]]) -> None:
        self.network.train()
        for state, target in training_data:
            state = torch.tensor(state, dtype=torch.float32)
            target = torch.tensor([float(target == i) for i in range(PIECES_PER_PLAYER)], dtype=torch.float32)

            self.optimizer.zero_grad()
            output = self.network(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

    def mutate(self, mutation_rate: float) -> None:
        state = self.network.state_dict()

        for key in state:
            if self.random.random() < mutation_rate:
                state[key] += torch.randn_like(state[key])

        self.network.load_state_dict(state)

    @classmethod
    def crossover(cls, parents: list["NeuralNetworkPlayer"], random: Random = Random()) -> "NeuralNetworkPlayer":
        child = cls(random)
        state = child.network.state_dict()

        for key in state:
            state[key] = child.random.choice(parents).network.state_dict()[key]

        child.network.load_state_dict(state)
        return child

    def fitness(self, playround: Callable[["NeuralNetworkPlayer"], Player], games: int) -> float:
        wins = 0

        for _ in range(games):
            winner = playround(self)
            if winner == self:
                wins += 1

        # NOTE: Should the fitness also include how long the game lasted?
        # We don't really care about the number of turns, but it could be a good indicator
        # Same for how many pieces are in the target area?
        # We also don't really care about the number of pieces in the target area
        # since we are only trying to win the game and that is just a binary
        # but it could help the network learn faster since it allows incremental improvements
        # instead of a binary win/loss

        return wins / games


class NeuralNetworkPopulation:
    random: Random
    population: list[NeuralNetworkPlayer]

    def __init__(self, population: int, random: Random = Random()) -> None:
        self.random = random
        self.population = [NeuralNetworkPlayer(random) for _ in range(population)]

    def __len__(self) -> int:
        return len(self.population)

    def __getitem__(self, index: int) -> NeuralNetworkPlayer:
        return self.population[index]

    def random_player(self) -> NeuralNetworkPlayer:
        return self.random.choice(self.population)

    def next_generation(self, scores: list[float], mutation_rate: float) -> None:
        assert len(scores) == len(self.population)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_networks = [self.population[i] for i in sorted_indices]

        # We only want to keep the best half of the population
        # and create new children for the rest of the population
        top_population = sorted_networks[: len(self.population) // 2]

        new_population: list[NeuralNetworkPlayer] = list()
        for _ in range(len(self.population) - len(top_population)):
            parents = self.random.sample(top_population, 2)
            child = NeuralNetworkPlayer.crossover(parents, self.random)
            child.mutate(mutation_rate)
            new_population.append(child)

        new_population = top_population + new_population
        assert len(new_population) == len(self.population)
        self.population = new_population
