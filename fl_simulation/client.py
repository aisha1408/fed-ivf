"""Flower NumPyClient implementation for privacy-preserving federated IVF learning.

Each client represents a distinct IVF clinic. Patient records remain strictly on the local
file system. Only model weights (NumPy arrays) and aggregate summary metrics are transmitted.
"""

import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.neural_net import IVFNet, create_model, get_parameters, set_parameters
from evaluation.metrics import calculate_auc, evaluate_model


class FlowerClient(fl.client.NumPyClient):
    """Flower client representing an individual IVF clinical center."""

    def __init__(
        self,
        client_id: int,
        csv_path: str,
        input_dim: int = 35,
        local_epochs: int = 2,
        batch_size: int = 16,
        learning_rate: float = 0.001
    ):
        self.client_id = client_id
        self.csv_path = csv_path
        self.input_dim = input_dim
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Load clinic's private local dataset
        df = pd.read_csv(csv_path)
        self.x = df.drop(columns=["live_birth"]).values.astype(np.float32)
        self.y = df["live_birth"].values.astype(np.float32)
        self.num_samples = len(self.y)

        self.dataset = TensorDataset(torch.tensor(self.x), torch.tensor(self.y).unsqueeze(1))
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        # Local neural network instance
        self.model = create_model(input_dim=self.input_dim)
        self.criterion = nn.BCELoss()

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return local model weights as NumPy arrays."""
        return get_parameters(self.model)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Receive global model parameters, train on local patients, return updated parameters."""
        # Step 1: Set model weights to global server parameters
        set_parameters(self.model, parameters)

        # Step 2: Local training
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        epochs = int(config.get("local_epochs", self.local_epochs)) if config else self.local_epochs
        self.model.train()
        total_loss = 0.0
        batches = 0

        for _ in range(epochs):
            for batch_x, batch_y in self.loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                batches += 1

        avg_loss = total_loss / max(1, batches)

        # Local evaluation
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(self.x)).cpu().numpy().ravel()
        auc = calculate_auc(self.y, preds)

        metrics = {
            "loss": float(avg_loss),
            "auc": float(auc)
        }

        # Step 3: Return updated parameters, sample count, and scalar metrics
        # NOTE: Raw patient data NEVER leaves the clinic
        return get_parameters(self.model), self.num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, str]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate received global parameters on local private clinic data."""
        set_parameters(self.model, parameters)

        self.model.eval()
        with torch.no_grad():
            preds_tensor = self.model(torch.tensor(self.x))
            loss = float(self.criterion(preds_tensor, torch.tensor(self.y).unsqueeze(1)).item())
            preds = preds_tensor.cpu().numpy().ravel()

        metrics = evaluate_model(self.y, preds)

        return loss, self.num_samples, {
            "auc": metrics["auc"],
            "f1": metrics["f1"],
            "brier": metrics["brier"],
            "loss": loss
        }


def make_client_fn(clients_dir: str = "data/clients", input_dim: int = 35):
    """Factory creating a client function for Flower simulation."""
    def client_fn(cid_or_context) -> fl.client.Client:
        # Flower can pass a string cid or a Context object depending on version
        if hasattr(cid_or_context, "node_id"):
            cid = str(getattr(cid_or_context, "node_id", 0))
        else:
            cid = str(cid_or_context)

        # Normalize cid to integer 0..num_clients-1
        try:
            client_id = int(cid) % 10
        except ValueError:
            client_id = 0

        csv_path = os.path.join(clients_dir, f"client_{client_id}.csv")
        client = FlowerClient(
            client_id=client_id,
            csv_path=csv_path,
            input_dim=input_dim,
            local_epochs=2,
            batch_size=16
        )
        return client.to_client()

    return client_fn
