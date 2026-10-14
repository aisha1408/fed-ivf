"""Client partitioning strategies for Federated Learning simulation.

Provides:
1. IID Partitioning (Homogeneous random distribution across clinics)
2. Age-Skew Non-IID Partitioning (Clinics specialize in specific age demographics)
3. Diagnosis-Skew Non-IID Partitioning (Clinics specialize in specific infertility etiologies)
"""

import os
import numpy as np
import pandas as pd


def create_iid_partitions(
    train_path="data/processed_train.csv",
    num_clients=10,
    random_state=42,
    output_dir="data/clients"
):
    """Partition processed training set into 10 IID virtual clinic datasets."""
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(train_path, str):
        df = pd.read_csv(train_path)
    else:
        df = train_path.copy()

    # Shuffle deterministically
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    partitions = np.array_split(shuffled_indices, num_clients)

    client_summaries = []

    print("\n" + "=" * 65)
    print("      IID CLINIC PARTITIONING (10 VIRTUAL IVF CLINICS)")
    print("=" * 65)
    print(f"{'Clinic':<12} | {'Samples':<10} | {'Positives':<12} | {'Negatives':<12} | {'Positive Rate'}")
    print("-" * 65)

    for i, idx in enumerate(partitions):
        client_df = df.iloc[idx].reset_index(drop=True)
        file_path = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(file_path, index=False)

        positives = int(client_df["live_birth"].sum())
        negatives = len(client_df) - positives
        pos_rate = positives / len(client_df) if len(client_df) > 0 else 0.0

        summary = {
            "client_id": i,
            "samples": len(client_df),
            "positives": positives,
            "negatives": negatives,
            "positive_rate": pos_rate,
            "path": file_path
        }
        client_summaries.append(summary)
        print(f"Clinic {i:<5} | {len(client_df):<10} | {positives:<12} | {negatives:<12} | {pos_rate * 100:>6.2f}%")

    print("=" * 65)
    return client_summaries


def create_age_skew_partitions(
    train_path="data/processed_train.csv",
    meta_path="data/processed_train_meta.csv",
    num_clients=10,
    random_state=42,
    output_dir="data/clients_age_skew"
):
    """Partition data by patient age into 10 Non-IID clinics simulating demographic specialization."""
    os.makedirs(output_dir, exist_ok=True)

    df_proc = pd.read_csv(train_path) if isinstance(train_path, str) else train_path.copy()
    df_meta = pd.read_csv(meta_path) if isinstance(meta_path, str) else meta_path.copy()

    # Sort indices by real female age to create natural demographic divergence
    sorted_indices = df_meta.sort_values(by="age").index.to_numpy()
    partitions = np.array_split(sorted_indices, num_clients)

    # Slight intra-partition shuffle so records are not strictly monotonic within a clinic
    np.random.seed(random_state)

    print("\n" + "=" * 75)
    print("      NON-IID AGE-SKEW PARTITIONING (DEMOGRAPHIC HETEROGENEITY)")
    print("=" * 75)
    print(f"{'Clinic':<12} | {'Samples':<8} | {'Age Range':<16} | {'Median Age':<12} | {'Positive Rate'}")
    print("-" * 75)

    client_summaries = []
    for i, idx in enumerate(partitions):
        shuffled_idx = np.random.permutation(idx)
        client_df = df_proc.iloc[shuffled_idx].reset_index(drop=True)
        file_path = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(file_path, index=False)

        client_meta = df_meta.iloc[idx]
        min_age = client_meta["age"].min()
        max_age = client_meta["age"].max()
        med_age = client_meta["age"].median()
        pos_rate = client_df["live_birth"].mean()

        summary = {
            "client_id": i,
            "samples": len(client_df),
            "age_min": float(min_age),
            "age_max": float(max_age),
            "age_median": float(med_age),
            "positive_rate": float(pos_rate),
            "path": file_path
        }
        client_summaries.append(summary)
        print(f"Clinic {i:<5} | {len(client_df):<8} | {min_age:4.1f} - {max_age:4.1f} yrs | {med_age:6.1f} yrs  | {pos_rate * 100:>6.2f}%")

    print("=" * 75)
    return client_summaries


def create_diagnosis_skew_partitions(
    train_path="data/processed_train.csv",
    meta_path="data/processed_train_meta.csv",
    num_clients=10,
    random_state=42,
    output_dir="data/clients_diagnosis_skew"
):
    """Partition data by infertility diagnosis etiology simulating specialized fertility centers."""
    os.makedirs(output_dir, exist_ok=True)

    df_proc = pd.read_csv(train_path) if isinstance(train_path, str) else train_path.copy()
    df_meta = pd.read_csv(meta_path) if isinstance(meta_path, str) else meta_path.copy()

    # Sort primarily by diagnosis code to cluster distinct etiologies together
    sorted_indices = df_meta.sort_values(by=["diag", "age"]).index.to_numpy()
    partitions = np.array_split(sorted_indices, num_clients)

    np.random.seed(random_state)

    print("\n" + "=" * 75)
    print("      NON-IID DIAGNOSIS-SKEW PARTITIONING (ETIOLOGY HETEROGENEITY)")
    print("=" * 75)
    print(f"{'Clinic':<12} | {'Samples':<8} | {'Dominant Diagnoses':<30} | {'Positive Rate'}")
    print("-" * 75)

    client_summaries = []
    for i, idx in enumerate(partitions):
        shuffled_idx = np.random.permutation(idx)
        client_df = df_proc.iloc[shuffled_idx].reset_index(drop=True)
        file_path = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(file_path, index=False)

        client_meta = df_meta.iloc[idx]
        diag_counts = client_meta["diag"].value_counts().head(2).to_dict()
        diag_str = ", ".join([f"{k}:{v}" for k, v in diag_counts.items()])
        pos_rate = client_df["live_birth"].mean()

        summary = {
            "client_id": i,
            "samples": len(client_df),
            "dominant_diagnoses": diag_str,
            "positive_rate": float(pos_rate),
            "path": file_path
        }
        client_summaries.append(summary)
        print(f"Clinic {i:<5} | {len(client_df):<8} | {diag_str:<30} | {pos_rate * 100:>6.2f}%")

    print("=" * 75)
    return client_summaries


if __name__ == "__main__":
    create_iid_partitions()
    create_age_skew_partitions()
    create_diagnosis_skew_partitions()
