"""Local-Only baseline training for IVF Live-Birth Prediction.

Each virtual clinic trains an isolated IVFNet solely on its private local patient records.
Each local model is evaluated against the shared global test set to quantify the performance
penalty of isolated, small-sample data silos compared to collaborative federated learning.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.neural_net import create_model
from evaluation.metrics import evaluate_model


def train_local_model(
    client_df: pd.DataFrame,
    input_dim: int,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001,
    seed: int = 42
) -> nn.Module:
    """Train an independent IVFNet exclusively on a single clinic's records."""
    torch.manual_seed(seed)
    model = create_model(input_dim=input_dim)

    x = client_df.drop(columns=["live_birth"]).values.astype(np.float32)
    y = client_df["live_birth"].values.astype(np.float32)

    dataset = TensorDataset(torch.tensor(x), torch.tensor(y).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()

    return model


def evaluate_local_baselines(
    clients_dir="data/clients",
    test_path="data/processed_test.csv",
    num_clients=10,
    epochs=30,
    results_path="results/local_results.json"
):
    """Train 10 isolated clinic models and evaluate each on the global test set."""
    df_test = pd.read_csv(test_path)
    x_test = df_test.drop(columns=["live_birth"]).values.astype(np.float32)
    y_test = df_test["live_birth"].values.astype(np.float32)
    test_tensor_x = torch.tensor(x_test)
    input_dim = x_test.shape[1]

    per_client_results = []

    print("\n" + "=" * 65)
    print("        LOCAL-ONLY BASELINE (ISOLATED CLINIC MODELS)")
    print("=" * 65)
    print(f"{'Clinic':<12} | {'Samples':<10} | {'AUC-ROC':<10} | {'F1':<10} | {'Brier Score'}")
    print("-" * 65)

    for i in range(num_clients):
        client_file = os.path.join(clients_dir, f"client_{i}.csv")
        if not os.path.exists(client_file):
            raise FileNotFoundError(f"Missing client partition: {client_file}")

        client_df = pd.read_csv(client_file)
        model = train_local_model(
            client_df=client_df,
            input_dim=input_dim,
            epochs=epochs,
            seed=42 + i
        )

        model.eval()
        with torch.no_grad():
            preds = model(test_tensor_x).cpu().numpy().ravel()

        metrics = evaluate_model(y_test, preds)
        metrics["client_id"] = i
        metrics["samples"] = len(client_df)
        per_client_results.append(metrics)

        print(f"Clinic {i:<5} | {len(client_df):<10} | {metrics['auc']:<10.4f} | {metrics['f1']:<10.4f} | {metrics['brier']:<10.4f}")

    # Calculate average metrics across all clinics
    avg_auc = float(np.mean([r["auc"] for r in per_client_results]))
    avg_f1 = float(np.mean([r["f1"] for r in per_client_results]))
    avg_brier = float(np.mean([r["brier"] for r in per_client_results]))
    avg_acc = float(np.mean([r["accuracy"] for r in per_client_results]))

    summary = {
        "average_auc": avg_auc,
        "average_f1": avg_f1,
        "average_brier": avg_brier,
        "average_accuracy": avg_acc,
        "per_client": per_client_results
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 65)
    print("LOCAL-ONLY BASELINE SUMMARY")
    print("-" * 65)
    print(f"Average AUC:         {avg_auc:.4f}")
    print(f"Average F1:          {avg_f1:.4f}")
    print(f"Average Brier Score: {avg_brier:.4f}")
    print(f"Average Accuracy:    {avg_acc:.4f}")
    print("=" * 65)

    return summary


if __name__ == "__main__":
    evaluate_local_baselines()
