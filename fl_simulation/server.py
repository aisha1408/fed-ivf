"""Flower Federated Learning Server configuration with FedAvg and custom metric aggregation.

Configures FedAvg strategy with 10 virtual IVF clinics, fraction_fit=1.0, tracking
global convergence (AUC, Loss, Brier score) across 30 communication rounds.
"""

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import flwr as fl
from flwr.common import Metrics, ndarrays_to_parameters, parameters_to_ndarrays

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.neural_net import create_model, set_parameters, get_parameters
from evaluation.metrics import evaluate_model


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training fit metrics from clients weighted by patient sample volume."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}

    agg_loss = sum([num_examples * m.get("loss", 0.0) for num_examples, m in metrics]) / total_examples
    agg_auc = sum([num_examples * m.get("auc", 0.5) for num_examples, m in metrics]) / total_examples

    return {
        "loss": float(agg_loss),
        "auc": float(agg_auc)
    }


def weighted_average_evaluate(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics from clients weighted by sample size."""
    total_examples = sum([num_examples for num_examples, _ in metrics])
    if total_examples == 0:
        return {}

    agg_auc = sum([num_examples * m.get("auc", 0.5) for num_examples, m in metrics]) / total_examples
    agg_loss = sum([num_examples * m.get("loss", 0.0) for num_examples, m in metrics]) / total_examples
    agg_f1 = sum([num_examples * m.get("f1", 0.0) for num_examples, m in metrics]) / total_examples
    agg_brier = sum([num_examples * m.get("brier", 0.0) for num_examples, m in metrics]) / total_examples

    return {
        "auc": float(agg_auc),
        "loss": float(agg_loss),
        "f1": float(agg_f1),
        "brier": float(agg_brier)
    }


def get_server_eval_fn(
    val_path: str = "data/processed_val.csv",
    input_dim: int = 35
) -> Callable:
    """Create a server-side evaluation function monitoring the global model on the validation split."""
    df_val = pd.read_csv(val_path)
    x_val = df_val.drop(columns=["live_birth"]).values.astype(np.float32)
    y_val = df_val["live_birth"].values.astype(np.float32)
    val_tensor_x = torch.tensor(x_val)

    eval_model = create_model(input_dim=input_dim)
    criterion = nn.BCELoss()

    def evaluate_fn(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar]
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        set_parameters(eval_model, parameters)
        eval_model.eval()
        with torch.no_grad():
            preds_tensor = eval_model(val_tensor_x)
            loss = float(criterion(preds_tensor, torch.tensor(y_val).unsqueeze(1)).item())
            preds = preds_tensor.cpu().numpy().ravel()

        metrics = evaluate_model(y_val, preds)
        return loss, {
            "auc": float(metrics["auc"]),
            "f1": float(metrics["f1"]),
            "brier": float(metrics["brier"]),
            "accuracy": float(metrics["accuracy"])
        }

    return evaluate_fn


def create_fedavg_strategy(
    initial_parameters: Optional[List[np.ndarray]] = None,
    val_path: str = "data/processed_val.csv",
    input_dim: int = 35,
    num_clients: int = 10
) -> fl.server.strategy.FedAvg:
    """Instantiate Flower FedAvg strategy with required client counts and metric aggregation."""
    eval_fn = get_server_eval_fn(val_path=val_path, input_dim=input_dim) if os.path.exists(val_path) else None

    init_params = None
    if initial_parameters is not None:
        init_params = ndarrays_to_parameters(initial_parameters)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=init_params,
        fit_metrics_aggregation_fn=weighted_average_fit,
        evaluate_metrics_aggregation_fn=weighted_average_evaluate,
        evaluate_fn=eval_fn
    )

    return strategy
