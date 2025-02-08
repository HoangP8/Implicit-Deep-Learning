"""

This file contains the code for a simple feed forward network used in the experiments.

"""

import torch
from torch import nn
import numpy as np


class FashionMNIST_FFNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, output_size, bias=False),
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1).flatten(start_dim=-2)
        return self.linear_relu_stack(x)
        # return nn.functional.softmax(self.linear_relu_stack(x), dim=-1)

    def scale_network(self, factor):
        layers_indices = [0, 2, 4]

        max_norm = max(
            torch.linalg.norm(self.linear_relu_stack[i].weight, np.inf)
            for i in layers_indices
        )

        for i in layers_indices:
            weight = self.linear_relu_stack[i].weight
            scaled_weight = torch.nn.Parameter(weight / (max_norm * factor))
            self.linear_relu_stack[i].weight = scaled_weight

        scaled_norm = max(
            torch.linalg.norm(self.linear_relu_stack[i].weight, np.inf)
            for i in layers_indices
        )
        print(f"Original norm : {max_norm}, Scaled norm: {scaled_norm}")

        return self
