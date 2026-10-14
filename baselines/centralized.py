"""Centralized baseline training for IVF Live-Birth Prediction.

Trains IVFNet on the fully pooled training dataset, monitors validation AUC,
and performs final evaluation on the unseen test set.
"""

import json
import os
import sys
import copy
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


def load_datasets(
    train_path="data/processed_train.csv",
    val_path="data/processed_val.csv",
    test_path="data/processed_test.csv"
):
    """Load train, val, and test datasets as PyTorch DataLoaders and arrays."""
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    x_train = df_train.drop(columns=["live_birth"]).values.astype(np.float32)
    y_train = df_train["live_birth"].values.astype(np.float32)

    x_val = df_val.drop(columns=["live_birth"]).values.astype(np.float32)
    y_val = df_val["live_birth"].values.astype(np.float32)

    x_test = df_test.drop(columns=["live_birth"]).values.astype(np.float32)
    y_test = df_test["live_birth"].values.astype(np.float32)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def train_centralized(
    train_path="data/processed_train.csv",
    val_path="data/processed_val.csv",
    test_path="data/processed_test.csv",
    epochs=40,
    batch_size=32,
    learning_rate=0.001,
    results_path="results/centralized_results.json",
    save_model_path="results/centralized_model.pt",
    seed=42
):
    """Train centralized baseline IVFNet model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_datasets(
        train_path, val_path, test_path
    )

    input_dim = x_train.shape[1]
    model = create_model(input_dim=input_dim)

    train_tensor = TensorDataset(torch.tensor(x_train), torch.tensor(y_train).unsqueeze(1))
    val_tensor_x = torch.tensor(x_val)
    val_tensor_y = torch.tensor(y_val).unsqueeze(1)
    test_tensor_x = torch.tensor(x_test)

    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_auc = -1.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

        # Validation monitoring
        model.eval()
        with torch.no_grad():
            val_preds = model(val_tensor_x).cpu().numpy().ravel()
            val_metrics = evaluate_model(y_val, val_preds)

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_weights = copy.deepcopy(model.state_dict())

    # Load best weights for test evaluation
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        test_preds = model(test_tensor_x).cpu().numpy().ravel()

    test_metrics = evaluate_model(y_test, test_preds)

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    if save_model_path:
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        torch.save(model.state_dict(), save_model_path)

    print("\n" + "=" * 35)
    print("CENTRALIZED BASELINE")
    print("-" * 35)
    print(f"AUC-ROC:     {test_metrics['auc']:.4f}")
    print(f"F1:          {test_metrics['f1']:.4f}")
    print(f"Brier Score: {test_metrics['brier']:.4f}")
    print(f"Accuracy:    {test_metrics['accuracy']:.4f}")
    print("=" * 35)

    return test_metrics, model


if __name__ == "__main__":
    train_centralized()
