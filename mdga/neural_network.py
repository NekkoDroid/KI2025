from pathlib import Path
from random import Random
from typing import Callable
import torch
import torch.nn as nn
import torch.optim as optim

from mdga.board import PIECES_PER_PLAYER, Board, Piece, PieceState
from mdga.game import Game
from mdga.player import ENCODED_MOVE_SIZE, Player, encode_move


class NeuralNetwork(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(ENCODED_MOVE_SIZE, 16384),
            nn.ReLU(),
            nn.Linear(16384, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, PIECES_PER_PLAYER),
            nn.Softmax(dim=-1),
        )

class NeuralNetworkPlayer(Player):
    device: torch.device
    random: Random
    network: nn.Module
    optimizer: optim.Optimizer
    criterion: nn.Module

    def __init__(self, device: torch.device, random: Random = Random()) -> None:
        super().__init__()
        self.device = device
        self.random = random
        self.network = NeuralNetwork().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def select_move(self, board: Board, id: int, roll: int, pieces: tuple[Piece, ...]) -> Piece:
        state = encode_move(board, id, roll)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)

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
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            target = torch.tensor(
                [float(target == i) for i in range(PIECES_PER_PLAYER)],
                dtype=torch.float32,
                device=self.device,
            )

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
    def crossover(cls, device: torch.device, parents: list["NeuralNetworkPlayer"], random: Random = Random()) -> "NeuralNetworkPlayer":
        child = cls(device, random)
        state = child.network.state_dict()

        for key in state:
            state[key] = child.random.choice(parents).network.state_dict()[key]

        child.network.load_state_dict(state)
        return child

    def fitness(self, play_game: Callable[["NeuralNetworkPlayer"], Game], games: int) -> float:
        score: float = 0

        for _ in range(games):
            game = play_game(self)

            if game.play() == self:
                score += 1

            # Add 0.25 extra score for each piece that we managed to get into the target area
            # This means that we get 2 points if we win and < 1 for any game we didn't win
            # we want to encourage that we get pieces into the target area, but we want to majorly
            # encourage actually winning the game
            id = game.players.index(self)
            score += len(game.board.filter(id=id, state=PieceState.target)) / PIECES_PER_PLAYER

        # NOTE: Should the fitness also include how long the game lasted?
        # We don't really care about the number of turns, but it could be a good indicator
        # Same for how many pieces are in the target area?

        return score / games


class NeuralNetworkPopulation:
    device: torch.device
    random: Random
    population: list[NeuralNetworkPlayer]

    def __init__(self, population: int, device: torch.device, random: Random = Random()) -> None:
        self.device = device
        self.random = random
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
