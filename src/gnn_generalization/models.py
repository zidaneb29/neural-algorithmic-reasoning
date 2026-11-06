"""Matched local message-passing architectures."""

import math

import torch
from torch import Tensor, nn


class SeededLinear(nn.Linear):
    """Initialize on CPU with an explicit generator."""

    def __init__(self, inputs: int, outputs: int, generator: torch.Generator):
        self.generator = generator
        super().__init__(inputs, outputs, device="cpu")

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound, generator=self.generator)
        nn.init.uniform_(self.bias, -bound, bound, generator=self.generator)


class MessagePassingLayer(nn.Module):
    def __init__(self, width: int, generator: torch.Generator):
        super().__init__()
        self.message = nn.Sequential(
            SeededLinear(width + 1, width, generator), nn.ReLU(),
            SeededLinear(width, width, generator),
        )
        self.update = nn.Sequential(
            SeededLinear(2 * width, width, generator), nn.ReLU(),
            SeededLinear(width, width, generator),
        )

    def forward(self, states: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        sender, receiver = edge_index
        messages = self.message(torch.cat((states[sender], edge_weight), dim=-1))
        aggregate = torch.zeros_like(states).index_add(0, receiver, messages)
        return self.update(torch.cat((states, aggregate), dim=-1))


class ShortestPathGNN(nn.Module):
    """Predict scaled weighted distances with one local round per layer."""

    def __init__(self, *, width: int, depth: int, residual: bool, seed: int):
        super().__init__()
        if min(width, depth) < 1:
            raise ValueError("width and depth must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.residual = residual
        self.encoder = SeededLinear(1, width, generator)
        self.layers = nn.ModuleList(
            MessagePassingLayer(width, generator) for _ in range(depth)
        )
        self.readout = SeededLinear(width, 1, generator)

    def forward(self, source_mask: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        states = self.encoder(source_mask[:, None].to(edge_weight.dtype))
        for layer in self.layers:
            update = layer(states, edge_index, edge_weight)
            states = states + update if self.residual else update
        return self.readout(states).squeeze(-1).masked_fill(source_mask, 0.0)
