"""PyTorch Neural Network architecture and parameter exchange utilities for IVF Live-Birth Prediction.

Architecture:
Input (auto-determined, 35) -> Linear(128) -> ReLU -> Dropout(0.3)
                            -> Linear(64)  -> ReLU -> Dropout(0.3)
                            -> Linear(32)  -> ReLU
                            -> Linear(1)   -> Sigmoid

Loss: Binary Cross-Entropy (BCELoss)
Optimizer: Adam
"""

from collections import OrderedDict
from typing import List
import numpy as np
import torch
import torch.nn as nn


class IVFNet(nn.Module):
    """Deep binary classifier for clinical IVF live-birth prediction."""

    def __init__(self, input_dim: int = 35):
        super(IVFNet, self).__init__()
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass generating live-birth probability."""
        return self.network(x)


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays for Flower serialization."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays received from Flower server."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def create_model(input_dim: int = 35) -> IVFNet:
    """Instantiate a new IVFNet with reproducible weight initialization."""
    torch.manual_seed(42)
    model = IVFNet(input_dim=input_dim)
    return model


if __name__ == "__main__":
    test_net = create_model(input_dim=35)
    params = get_parameters(test_net)
    print(f"IVFNet instantiated with input_dim=35.")
    print(f"Number of parameter tensors: {len(params)}")
    dummy_input = torch.randn(4, 35)
    out = test_net(dummy_input)
    print(f"Output shape for batch of 4: {out.shape}, sample output: {out.squeeze().tolist()}")
    set_parameters(test_net, params)
    print("Parameter get/set roundtrip verified successfully.")
