"""Federated Learning simulation orchestrator using Flower FedAvg.

Simulates 30 communication rounds across 10 virtual IVF clinics.
Extracts updated model weights, aggregates them via FedAvg, logs convergence metrics,
evaluates on the global test set, and saves the final federated model.
"""

import json
import os
import sys
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.neural_net import create_model, get_parameters, set_parameters
from fl_simulation.client import FlowerClient, make_client_fn
from fl_simulation.server import create_fedavg_strategy
from evaluation.metrics import evaluate_model


class SimpleSimClientProxy(ClientProxy):
    """Minimal ClientProxy adapter for in-process Flower FedAvg aggregation."""

    def __init__(self, cid: str):
        super().__init__(cid)

    def get_run_id(self) -> int:
        return 0

    def get_properties(self, ins, timeout, group_id):
        raise NotImplementedError()

    def get_parameters(self, ins, timeout, group_id):
        raise NotImplementedError()

    def fit(self, ins, timeout, group_id):
        raise NotImplementedError()

    def evaluate(self, ins, timeout, group_id):
        raise NotImplementedError()

    def reconnect(self, ins, timeout, group_id):
        raise NotImplementedError()


def run_flower_fedavg_simulation(
    clients_dir: str = "data/clients",
    test_path: str = "data/processed_test.csv",
    val_path: str = "data/processed_val.csv",
    num_rounds: int = 30,
    num_clients: int = 10,
    local_epochs: int = 2,
    batch_size: int = 16,
    input_dim: int = 35,
    metrics_output_path: str = "results/fl_metrics.json",
    model_output_path: str = "results/federated_model.pt",
    seed: int = 42
) -> Dict:
    """Execute Flower FedAvg simulation across 10 virtual IVF clinics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("\n" + "=" * 70)
    print(f"  STARTING FEDERATED LEARNING SIMULATION (FLOWER FEDAVG)")
    print(f"  Rounds: {num_rounds} | Clinics: {num_clients} | Local Epochs/Round: {local_epochs}")
    print(f"  Clinic Partitions: {clients_dir}")
    print("=" * 70)

    # Initialize shared model architecture
    initial_model = create_model(input_dim=input_dim)
    current_parameters = get_parameters(initial_model)

    # Create Flower strategy
    strategy = create_fedavg_strategy(
        initial_parameters=current_parameters,
        val_path=val_path,
        input_dim=input_dim,
        num_clients=num_clients
    )

    # Instantiate the 10 Flower clients
    clients: List[FlowerClient] = []
    client_proxies: List[ClientProxy] = []
    for cid in range(num_clients):
        csv_path = os.path.join(clients_dir, f"client_{cid}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing client partition CSV: {csv_path}")

        c = FlowerClient(
            client_id=cid,
            csv_path=csv_path,
            input_dim=input_dim,
            local_epochs=local_epochs,
            batch_size=batch_size
        )
        clients.append(c)
        client_proxies.append(SimpleSimClientProxy(str(cid)))

    # Metrics history tracker
    history = {
        "rounds": [],
        "auc": [],
        "loss": [],
        "f1": [],
        "brier": []
    }

    # Load validation set for server monitoring
    df_val = pd.read_csv(val_path)
    x_val = torch.tensor(df_val.drop(columns=["live_birth"]).values.astype(np.float32))
    y_val = df_val["live_birth"].values.astype(np.float32)

    eval_model = create_model(input_dim=input_dim)
    criterion = torch.nn.BCELoss()

    # Communication rounds loop
    for rnd in range(1, num_rounds + 1):
        fit_results = []

        # Local training on each clinic's private dataset
        for cid, (client, proxy) in enumerate(zip(clients, client_proxies)):
            updated_params, num_samples, fit_metrics = client.fit(
                current_parameters,
                {"local_epochs": str(local_epochs)}
            )

            fit_res = FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(updated_params),
                num_examples=num_samples,
                metrics=fit_metrics
            )
            fit_results.append((proxy, fit_res))

        # Flower FedAvg aggregation
        aggregated_parameters, _ = strategy.aggregate_fit(rnd, fit_results, [])
        if aggregated_parameters is not None:
            current_parameters = parameters_to_ndarrays(aggregated_parameters)

        # Global model evaluation on server validation set
        set_parameters(eval_model, current_parameters)
        eval_model.eval()
        with torch.no_grad():
            preds_tensor = eval_model(x_val)
            val_loss = float(criterion(preds_tensor, torch.tensor(y_val).unsqueeze(1)).item())
            preds = preds_tensor.cpu().numpy().ravel()

        round_metrics = evaluate_model(y_val, preds)

        history["rounds"].append(rnd)
        history["auc"].append(round_metrics["auc"])
        history["loss"].append(val_loss)
        history["f1"].append(round_metrics["f1"])
        history["brier"].append(round_metrics["brier"])

        if rnd == 1 or rnd % 5 == 0 or rnd == num_rounds:
            print(
                f"Round {rnd:>2}/{num_rounds:>2} | "
                f"Global Val AUC: {round_metrics['auc']:.4f} | "
                f"Loss: {val_loss:.4f} | "
                f"F1: {round_metrics['f1']:.4f} | "
                f"Brier: {round_metrics['brier']:.4f}"
            )

    # Save final model weights
    set_parameters(eval_model, current_parameters)
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    torch.save(eval_model.state_dict(), model_output_path)

    # Final Evaluation on unseen global test set
    df_test = pd.read_csv(test_path)
    x_test = torch.tensor(df_test.drop(columns=["live_birth"]).values.astype(np.float32))
    y_test = df_test["live_birth"].values.astype(np.float32)

    eval_model.eval()
    with torch.no_grad():
        test_preds = eval_model(x_test).cpu().numpy().ravel()

    test_metrics = evaluate_model(y_test, test_preds)

    # Save metrics JSON
    output_record = {
        "history": history,
        "final_test_metrics": test_metrics,
        "best_val_auc": float(max(history["auc"])),
        "num_rounds": num_rounds,
        "num_clients": num_clients
    }

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(output_record, f, indent=2)

    print("\n" + "=" * 35)
    print("FEDERATED LEARNING RESULTS")
    print("-" * 35)
    print(f"Final AUC:         {test_metrics['auc']:.4f}")
    print(f"Final F1:          {test_metrics['f1']:.4f}")
    print(f"Final Brier Score: {test_metrics['brier']:.4f}")
    print(f"Accuracy:          {test_metrics['accuracy']:.4f}")
    print(f"Best Val AUC:      {max(history['auc']):.4f}")
    print("=" * 35)

    return output_record


def run_all_fl_strategies(num_rounds: int = 30):
    """Run Federated Learning across IID, Age-skew, and Diagnosis-skew partitions."""
    print("\n>>> 1/3: Running FL with IID Clinic Partitions...")
    iid_res = run_flower_fedavg_simulation(
        clients_dir="data/clients",
        num_rounds=num_rounds,
        metrics_output_path="results/fl_metrics.json",
        model_output_path="results/federated_model.pt"
    )

    # Also save as fl_iid_metrics.json for comparison
    with open("results/fl_iid_metrics.json", "w", encoding="utf-8") as f:
        json.dump(iid_res, f, indent=2)

    print("\n>>> 2/3: Running FL with Age-Skew Non-IID Clinic Partitions...")
    age_res = run_flower_fedavg_simulation(
        clients_dir="data/clients_age_skew",
        num_rounds=num_rounds,
        metrics_output_path="results/fl_age_skew_metrics.json",
        model_output_path="results/federated_model_age_skew.pt"
    )

    print("\n>>> 3/3: Running FL with Diagnosis-Skew Non-IID Clinic Partitions...")
    diag_res = run_flower_fedavg_simulation(
        clients_dir="data/clients_diagnosis_skew",
        num_rounds=num_rounds,
        metrics_output_path="results/fl_diagnosis_skew_metrics.json",
        model_output_path="results/federated_model_diag_skew.pt"
    )

    return iid_res, age_res, diag_res


if __name__ == "__main__":
    run_flower_fedavg_simulation(num_rounds=30)
