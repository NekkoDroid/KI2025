from pathlib import Path
from random import Random
import torch
import torch.nn as nn
import torch.optim as optim

from mdga.board import PIECES_PER_PLAYER, Board, Piece
from mdga.player import ENCODED_MOVE_SIZE, Player, encode_move


class NeuralNetwork(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Linear(ENCODED_MOVE_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, PIECES_PER_PLAYER),
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
