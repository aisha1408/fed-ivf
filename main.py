"""Main orchestration pipeline for Federated Learning for IVF Live-Birth Prediction.

Executes the complete experimental pipeline:
1. Clinical dataset inspection and summary generation
2. Leakage-free preprocessing, imputation, scaling, and train/val/test splitting
3. Virtual clinic partitioning (IID, Age-skew Non-IID, Diagnosis-skew Non-IID)
4. Centralized pooled baseline training
5. Local-only isolated clinic baselines
6. Flower Federated Learning (FedAvg) simulation (30 communication rounds)
7. Non-IID federated benchmarking (Age-skew & Diagnosis-skew)
8. Fairness evaluation with Fairlearn
9. Visualization generation (convergence curves, model comparison, fairness disparities)
10. Final model comparison summary and artifact persistence
"""

import json
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from preprocessing.preprocess import (
    inspect_and_summarize_dataset,
    prepare_dataset
)
from preprocessing.partition import (
    create_iid_partitions,
    create_age_skew_partitions,
    create_diagnosis_skew_partitions
)
from baselines.centralized import train_centralized
from baselines.local_only import evaluate_local_baselines
from fl_simulation.run_simulation import run_flower_fedavg_simulation
from evaluation.fairness import run_fairness_evaluation
from results.plots import generate_all_figures


def run_pipeline(
    excel_path: str = "data/ivf_data.xlsx",
    num_rounds: int = 30,
    num_clients: int = 10,
    run_non_iid_benchmarks: bool = True
):
    """Run end-to-end pipeline."""
    start_time = time.time()

    print("\n" + "=" * 76)
    print("      FEDERATED LEARNING FOR IVF LIVE-BIRTH PREDICTION PIPELINE")
    print("=" * 76)
    print("Privacy Guarantee: Private EHR records remain in local clinic databases.")
    print("Only neural network weights and metrics are exchanged using Flower FedAvg.")
    print("=" * 76 + "\n")

    # Step 1: Inspect Dataset
    print("[1/10] Inspecting real clinical dataset...")
    inspect_and_summarize_dataset(excel_path=excel_path, output_path="results/dataset_summary.txt")

    # Step 2: Preprocess Dataset
    print("\n[2/10] Preprocessing clinical data (leakage-free train/val/test splits)...")
    prep_res = prepare_dataset(excel_path=excel_path, output_dir="data", random_state=42)
    schema = prep_res["schema"]

    # Step 3: Virtual Clinic Partitioning
    print("\n[3/10] Partitioning training data into 10 virtual IVF clinics...")
    create_iid_partitions(train_path="data/processed_train.csv", num_clients=num_clients)
    if run_non_iid_benchmarks:
        create_age_skew_partitions(
            train_path="data/processed_train.csv",
            meta_path="data/processed_train_meta.csv",
            num_clients=num_clients
        )
        create_diagnosis_skew_partitions(
            train_path="data/processed_train.csv",
            meta_path="data/processed_train_meta.csv",
            num_clients=num_clients
        )

    # Step 4: Centralized Pooled Baseline
    print("\n[4/10] Training Centralized Baseline (Full Pooled Data)...")
    centralized_metrics, _ = train_centralized(
        train_path="data/processed_train.csv",
        val_path="data/processed_val.csv",
        test_path="data/processed_test.csv",
        epochs=35,
        results_path="results/centralized_results.json"
    )

    # Step 5: Local-Only Isolated Baselines
    print("\n[5/10] Training Local-Only Baselines (10 Isolated Clinics)...")
    local_summary = evaluate_local_baselines(
        clients_dir="data/clients",
        test_path="data/processed_test.csv",
        num_clients=num_clients,
        epochs=25,
        results_path="results/local_results.json"
    )

    # Step 6: Federated Learning Simulation (IID)
    print(f"\n[6/10] Running Federated Learning (Flower FedAvg, {num_rounds} Rounds)...")
    fl_iid_results = run_flower_fedavg_simulation(
        clients_dir="data/clients",
        test_path="data/processed_test.csv",
        val_path="data/processed_val.csv",
        num_rounds=num_rounds,
        num_clients=num_clients,
        local_epochs=2,
        metrics_output_path="results/fl_metrics.json",
        model_output_path="results/federated_model.pt"
    )
    # Save a copy as fl_iid_metrics.json
    with open("results/fl_iid_metrics.json", "w", encoding="utf-8") as f:
        json.dump(fl_iid_results, f, indent=2)

    fl_metrics = fl_iid_results["final_test_metrics"]
    best_fl_val_auc = fl_iid_results["best_val_auc"]

    # Step 7: Non-IID Federated Learning Benchmarks (Optional / Extension)
    age_skew_auc = None
    diag_skew_auc = None
    if run_non_iid_benchmarks:
        print("\n[7/10] Benchmarking Non-IID Federated Learning (Age-Skew & Diagnosis-Skew)...")
        age_skew_res = run_flower_fedavg_simulation(
            clients_dir="data/clients_age_skew",
            test_path="data/processed_test.csv",
            val_path="data/processed_val.csv",
            num_rounds=min(num_rounds, 15),
            num_clients=num_clients,
            local_epochs=2,
            metrics_output_path="results/fl_age_skew_metrics.json",
            model_output_path="results/federated_model_age_skew.pt"
        )
        age_skew_auc = age_skew_res["final_test_metrics"]["auc"]

        diag_skew_res = run_flower_fedavg_simulation(
            clients_dir="data/clients_diagnosis_skew",
            test_path="data/processed_test.csv",
            val_path="data/processed_val.csv",
            num_rounds=min(num_rounds, 15),
            num_clients=num_clients,
            local_epochs=2,
            metrics_output_path="results/fl_diagnosis_skew_metrics.json",
            model_output_path="results/federated_model_diag_skew.pt"
        )
        diag_skew_auc = diag_skew_res["final_test_metrics"]["auc"]
    else:
        print("\n[7/10] Skipping Non-IID benchmarks.")

    # Step 8: Algorithmic Fairness Audit
    print("\n[8/10] Evaluating Algorithmic Fairness using Fairlearn...")
    fairness_results = run_fairness_evaluation(
        model_path="results/federated_model.pt",
        test_proc_path="data/processed_test.csv",
        test_meta_path="data/processed_test_meta.csv",
        report_output_path="results/fairness_report.txt"
    )

    # Step 9: Visualization Plots
    print("\n[9/10] Generating publication-quality figures...")
    generate_all_figures()

    # Step 10: Comparison Table and Artifact Persistence
    print("\n[10/10] Compiling Final Comparison Results...")
    comparison_data = {
        "dataset_name": "Real IVF Treatment Dataset (Freeze-all IVF)",
        "total_records": schema["n_samples_total"],
        "num_features": schema["n_features_processed"],
        "num_clinics": num_clients,
        "federated_rounds": num_rounds,
        "centralized": centralized_metrics,
        "federated_iid": fl_metrics,
        "local_only_average": {
            "auc": local_summary["average_auc"],
            "f1": local_summary["average_f1"],
            "brier": local_summary["average_brier"],
            "accuracy": local_summary["average_accuracy"]
        },
        "non_iid_benchmarks": {
            "age_skew_auc": age_skew_auc,
            "diagnosis_skew_auc": diag_skew_auc
        },
        "best_federated_val_auc": best_fl_val_auc,
        "runtime_seconds": round(time.time() - start_time, 2)
    }

    with open("results/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    # Format final table text
    table_lines = []
    table_lines.append("")
    table_lines.append("========================================================")
    table_lines.append("             FINAL MODEL COMPARISON")
    table_lines.append("========================================================")
    table_lines.append(f"{'Model':<18} {'AUC':<9} {'F1':<9} {'Brier Score'}")
    table_lines.append("--------------------------------------------------------")
    table_lines.append(f"{'Centralized':<18} {centralized_metrics['auc']:<9.3f} {centralized_metrics['f1']:<9.3f} {centralized_metrics['brier']:.3f}")
    table_lines.append(f"{'Federated':<18} {fl_metrics['auc']:<9.3f} {fl_metrics['f1']:<9.3f} {fl_metrics['brier']:.3f}")
    table_lines.append(f"{'Local-only':<18} {local_summary['average_auc']:<9.3f} {local_summary['average_f1']:<9.3f} {local_summary['average_brier']:.3f}")
    table_lines.append("========================================================")
    table_lines.append("")
    table_lines.append(f"Dataset:            {comparison_data['dataset_name']}")
    table_lines.append(f"Number of samples:  {schema['n_samples_total']} ({schema['n_train']} train, {schema['n_val']} val, {schema['n_test']} test)")
    table_lines.append(f"Number of features: {schema['n_features_processed']}")
    table_lines.append(f"Number of clinics:  {num_clients}")
    table_lines.append(f"Federated rounds:   {num_rounds}")
    table_lines.append(f"Best federated AUC: {best_fl_val_auc:.3f}")
    if age_skew_auc and diag_skew_auc:
        table_lines.append(f"Non-IID Age-Skew AUC:       {age_skew_auc:.3f}")
        table_lines.append(f"Non-IID Diagnosis-Skew AUC: {diag_skew_auc:.3f}")
    table_lines.append("")
    table_lines.append("Artifact Locations:")
    table_lines.append("  - Dataset Summary:     results/dataset_summary.txt")
    table_lines.append("  - Centralized Results: results/centralized_results.json")
    table_lines.append("  - Local Results:       results/local_results.json")
    table_lines.append("  - FL Metrics History:  results/fl_metrics.json")
    table_lines.append("  - Fairness Report:     results/fairness_report.txt")
    table_lines.append("  - Saved Figures:       results/figures/")
    table_lines.append("  - Trained FL Weights:  results/federated_model.pt")
    table_lines.append("  - Web Demonstration:   python app.py")
    table_lines.append(f"Total Pipeline Runtime: {comparison_data['runtime_seconds']} seconds")
    table_lines.append("========================================================")

    final_table_text = "\n".join(table_lines)

    with open("results/final_comparison.txt", "w", encoding="utf-8") as f:
        f.write(final_table_text)

    print(final_table_text)
    return comparison_data


if __name__ == "__main__":
    run_pipeline(num_rounds=30, num_clients=10, run_non_iid_benchmarks=True)
