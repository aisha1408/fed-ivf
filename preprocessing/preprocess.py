"""Data inspection, cleaning, and preprocessing pipeline for IVF dataset.

Preserves privacy, eliminates data leakage, fits preprocessing strictly on training
splits, and saves processed feature representations for centralized and federated modeling.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# Clean column name mapping to ensure Python-friendly, standardized names
COLUMN_RENAME_MAP = {
    "Unnamed: 0": "row_index_or_date",
    "tarih": "cycle_start_date",
    "pno": "patient_id",
    "dtarih": "patient_birth_date",
    "age": "age",
    "inf yrs": "inf_yrs",
    "prime inf": "prime_inf",
    "prev ET numb": "prev_et_numb",
    "afc": "afc",
    "bmi": "bmi",
    "diag": "diag",
    "stim type": "stim_type",
    "trigger type": "trigger_type",
    "stim days": "stim_days",
    "FSH": "fsh",
    "hmg": "hmg",
    "tot FSH": "tot_fsh",
    "eggs": "eggs",
    "\u0131cs\u0131": "icsi",
    "icsi": "icsi",
    "2PN": "two_pn",
    "blast>=2bb": "blast_ge_2bb",
    "blast FZ>=2bb": "blast_fz_ge_2bb",
    "mat rate": "mat_rate",
    "fert rate": "fert_rate",
    "blast rate": "blast_rate",
    "note": "note",
    "date": "transfer_date",
    "FET#": "fet_n",
    "#blast transf": "n_blast_transf",
    "SET emb score": "set_emb_score",
    "endo mm": "endo_mm",
    "PR": "pregnancy_rate_leak",
    "LB 1": "live_birth"
}

IDENTIFIER_COLUMNS = [
    "row_index_or_date",
    "cycle_start_date",
    "patient_id",
    "patient_birth_date",
    "transfer_date",
    "note",
    "stim_type"  # 100% constant 'antagonist'
]

LEAKAGE_COLUMNS = [
    "pregnancy_rate_leak",
    "pr",
    "fh",
    "fetal_heartbeat",
    "number_of_live_births",
    "delivery_date"
]

FEATURE_LABELS = {
    "age": "Female Age (years)",
    "inf_yrs": "Infertility Duration (years)",
    "prime_inf": "Primary Infertility (0=Secondary, 1=Primary)",
    "prev_et_numb": "Previous Embryo Transfers",
    "afc": "Antral Follicle Count (AFC)",
    "bmi": "Body Mass Index (BMI kg/m\u00b2)",
    "diag": "Infertility Diagnosis Category",
    "trigger_type": "Ovulation Trigger Type",
    "stim_days": "Ovarian Stimulation Days",
    "fsh": "Daily FSH Dose (IU)",
    "hmg": "Daily HMG Dose (IU)",
    "tot_fsh": "Total FSH Administered (IU)",
    "eggs": "Total Oocytes Retrieved",
    "icsi": "Oocytes Injected (ICSI)",
    "two_pn": "2PN Fertilised Embryos",
    "blast_ge_2bb": "Good Quality Blastocysts (\u22652BB)",
    "blast_fz_ge_2bb": "Frozen Blastocysts (\u22652BB)",
    "mat_rate": "Oocyte Maturity Rate (0-1)",
    "fert_rate": "Fertilisation Rate (0-1)",
    "blast_rate": "Blastocyst Formation Rate (0-1)",
    "fet_n": "Frozen Embryo Transfer Number",
    "n_blast_transf": "Number of Blastocysts Transferred",
    "set_emb_score": "Single Embryo Transfer Score (1-8)",
    "endo_mm": "Endometrial Thickness (mm)"
}

DIAGNOSIS_LABELS = {
    "m": "Male Factor Infertility",
    "o": "Ovulatory Dysfunction",
    "u": "Unexplained Infertility",
    "dor": "Diminished Ovarian Reserve (DOR)",
    "a": "Advanced Maternal Age",
    "pcos": "Polycystic Ovary Syndrome (PCOS)",
    "e": "Endometriosis",
    "t": "Tubal Factor",
    "multi": "Multiple Etiologies"
}

TRIGGER_LABELS = {
    "a": "GnRH Agonist Trigger",
    "d": "Dual Trigger (Agonist + hCG)",
    "h": "hCG Trigger"
}


def load_data(excel_path="data/ivf_data.xlsx"):
    """Load the primary IVF treatment sheet with completed outcomes."""
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"IVF dataset not found at: {excel_path}")

    # Inspect sheet names
    excel_file = pd.ExcelFile(excel_path)
    sheet_names = excel_file.sheet_names

    # Select the sheet containing completed IVF cycles and outcomes
    target_sheet = None
    for name in sheet_names:
        lower_name = name.lower()
        if "lb outcome" in lower_name or "tot cycles" in lower_name or "ivf" in lower_name:
            target_sheet = name
            break
    if target_sheet is None:
        target_sheet = sheet_names[0]

    # Sheet layout inspection: row index 1 has the true column headers
    raw_preview = pd.read_excel(excel_path, sheet_name=target_sheet, header=None, nrows=5)
    header_row = 0
    for r in range(min(4, len(raw_preview))):
        row_vals = [str(v).lower() for v in raw_preview.iloc[r].dropna()]
        if any("age" in v for v in row_vals) and any("diag" in v or "bmi" in v for v in row_vals):
            header_row = r
            break

    df = pd.read_excel(excel_path, sheet_name=target_sheet, header=header_row)
    return df, target_sheet, sheet_names


def clean_data(df):
    """Clean column names, normalize Turkish letters, and remove duplicates."""
    df_clean = df.copy()

    # Map column names
    new_cols = []
    for c in df_clean.columns:
        c_str = str(c).strip()
        if c_str in COLUMN_RENAME_MAP:
            new_cols.append(COLUMN_RENAME_MAP[c_str])
        else:
            # Normalize characters
            sanitized = c_str.lower().replace("\u0131", "i").replace(" ", "_").replace("-", "_")
            new_cols.append(sanitized)
    df_clean.columns = new_cols

    # Remove duplicates
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()

    return df_clean


def identify_target(df):
    """Identify and validate the binary live birth target column."""
    target_candidates = ["live_birth", "lb_1", "live_birth_occurrence", "number_of_live_births"]
    target_col = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        for c in df.columns:
            if "live" in c and "birth" in c:
                target_col = c
                break

    if target_col is None:
        raise ValueError("Could not find a valid live birth target column in the IVF dataset.")

    # Convert to standard binary 0/1
    series = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
    # 1 if >= 1 live birth, 0 otherwise
    binary_target = (series >= 1).astype(int)

    return target_col, binary_target


def inspect_and_summarize_dataset(excel_path="data/ivf_data.xlsx", output_path="results/dataset_summary.txt"):
    """Inspect dataset, analyze data types, missingness, and write dataset_summary.txt."""
    df_raw, target_sheet, sheet_names = load_data(excel_path)
    df_clean = clean_data(df_raw)
    target_col, binary_target = identify_target(df_clean)
    df_clean["live_birth"] = binary_target

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("FEDERATED IVF: REAL CLINICAL DATASET INSPECTION & SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Source file: {excel_path}")
    summary_lines.append(f"Worksheets present: {sheet_names}")
    summary_lines.append(f"Selected worksheet for analysis: '{target_sheet}'")
    summary_lines.append(f"Total raw records: {len(df_clean)}")
    summary_lines.append(f"Total raw columns: {len(df_clean.columns)}")
    summary_lines.append("")

    summary_lines.append("-" * 80)
    summary_lines.append("1. TARGET VARIABLE: LIVE BIRTH OUTCOME")
    summary_lines.append("-" * 80)
    pos_count = int(binary_target.sum())
    neg_count = len(binary_target) - pos_count
    pos_rate = pos_count / len(binary_target)
    summary_lines.append(f"Target column name: '{target_col}' -> mapped to 'live_birth'")
    summary_lines.append(f"Live birth cases (Class 1): {pos_count} ({pos_rate * 100:.2f}%)")
    summary_lines.append(f"Non-live birth cases (Class 0): {neg_count} ({(1 - pos_rate) * 100:.2f}%)")
    summary_lines.append("Class balance: Moderately balanced (~55% positive). Stratification applied.")
    summary_lines.append("")

    summary_lines.append("-" * 80)
    summary_lines.append("2. COLUMNS REMOVED (IDENTIFIERS & DATA LEAKAGE)")
    summary_lines.append("-" * 80)
    summary_lines.append(f"Identifiers / Non-predictive metadata: {IDENTIFIER_COLUMNS}")
    summary_lines.append(f"Data leakage columns removed: {LEAKAGE_COLUMNS}")
    summary_lines.append("Leakage Rationale: 'pregnancy_rate_leak' (PR) represents post-transfer hCG pregnancy test,")
    summary_lines.append("which occurs after embryo transfer and leaks outcome state. All post-delivery metrics excluded.")
    summary_lines.append("")

    summary_lines.append("-" * 80)
    summary_lines.append("3. FEATURE COLUMN INVENTORY & MISSINGNESS")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'Feature Name':<22} | {'Data Type':<10} | {'Missing %':<10} | {'Unique':<8} | {'Sample Values'}")
    summary_lines.append("-" * 80)

    for c in df_clean.columns:
        if c in IDENTIFIER_COLUMNS or c in LEAKAGE_COLUMNS or c == target_col or c == "live_birth":
            continue
        col_dtype = str(df_clean[c].dtype)
        null_pct = df_clean[c].isnull().mean() * 100
        n_unique = df_clean[c].nunique(dropna=True)
        samples = str(df_clean[c].dropna().unique()[:4].tolist())
        summary_lines.append(f"{c:<22} | {col_dtype:<10} | {null_pct:>8.2f}% | {n_unique:>8d} | {samples}")

    summary_lines.append("")
    summary_lines.append("-" * 80)
    summary_lines.append("4. CATEGORICAL FEATURES & UNIQUE VALUES")
    summary_lines.append("-" * 80)
    cat_cols = ["diag", "trigger_type"]
    for cat in cat_cols:
        if cat in df_clean.columns:
            counts = df_clean[cat].value_counts(dropna=False).to_dict()
            summary_lines.append(f"Column '{cat}':")
            for k, v in counts.items():
                label = DIAGNOSIS_LABELS.get(k, TRIGGER_LABELS.get(k, "Unknown"))
                summary_lines.append(f"   - {k} ({label}): {v} patients")
    summary_lines.append("=" * 80)

    summary_text = "\n".join(summary_lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    return summary_text


def split_data(df, target_col="live_birth", train_size=0.7, val_size=0.1, test_size=0.2, random_state=42):
    """Split data into 70% train, 10% val, 20% test stratified on live birth."""
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-6, "Splits must sum to 1.0"

    # First split off test set (20%)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state
    )

    # From remaining 80%, take validation set (10% of total -> 0.1 / 0.8 = 0.125)
    val_ratio_of_train_val = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_of_train_val,
        stratify=train_val_df[target_col],
        random_state=random_state
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


def fit_preprocessor(train_df, numeric_cols, categorical_cols):
    """Fit median imputer + standard scaler for numeric features, and OneHotEncoder for categoricals."""
    # Numeric pipeline
    num_imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Fit numeric imputer and scaler
    train_num_imputed = num_imputer.fit_transform(train_df[numeric_cols])
    scaler.fit(train_num_imputed)

    # Categorical pipeline
    cat_imputer = SimpleImputer(strategy="constant", fill_value="Unknown")
    train_cat_imputed = cat_imputer.fit_transform(train_df[categorical_cols].astype(str))

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_cat_imputed)

    cat_feature_names = list(encoder.get_feature_names_out(categorical_cols))

    preprocessor = {
        "num_imputer": num_imputer,
        "scaler": scaler,
        "cat_imputer": cat_imputer,
        "encoder": encoder,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "cat_feature_names": cat_feature_names,
        "all_feature_names": numeric_cols + cat_feature_names
    }
    return preprocessor


def transform_data(df, preprocessor, target_col="live_birth"):
    """Transform data using fitted preprocessor. Pure transform, strictly avoiding data leakage."""
    num_cols = preprocessor["numeric_cols"]
    cat_cols = preprocessor["categorical_cols"]

    # Transform numeric
    num_imputed = preprocessor["num_imputer"].transform(df[num_cols])
    num_scaled = preprocessor["scaler"].transform(num_imputed)

    # Transform categorical
    cat_imputed = preprocessor["cat_imputer"].transform(df[cat_cols].astype(str))
    cat_encoded = preprocessor["encoder"].transform(cat_imputed)

    # Combine features
    x_transformed = np.hstack([num_scaled, cat_encoded])
    feature_names = preprocessor["all_feature_names"]

    df_transformed = pd.DataFrame(x_transformed, columns=feature_names)

    if target_col in df.columns:
        df_transformed[target_col] = df[target_col].values.astype(int)

    return df_transformed


def prepare_dataset(
    excel_path="data/ivf_data.xlsx",
    output_dir="data",
    random_state=42
):
    """End-to-end dataset preparation: loads, cleans, splits, fits preprocessor, and saves."""
    os.makedirs(output_dir, exist_ok=True)

    df_raw, target_sheet, _ = load_data(excel_path)
    df_clean = clean_data(df_raw)
    target_col, binary_target = identify_target(df_clean)
    df_clean["live_birth"] = binary_target

    # Filter out identifier, leakage, and target columns to get raw feature list
    excluded = set(IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS + [target_col, "live_birth"])
    feature_cols = [c for c in df_clean.columns if c not in excluded]

    categorical_cols = ["diag", "trigger_type"]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Convert numeric columns to float/numeric
    for c in numeric_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    # Split dataset
    train_df, val_df, test_df = split_data(
        df_clean,
        target_col="live_birth",
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
        random_state=random_state
    )

    # Fit preprocessor strictly on training data
    preprocessor = fit_preprocessor(train_df, numeric_cols, categorical_cols)

    # Transform all splits
    train_proc = transform_data(train_df, preprocessor, target_col="live_birth")
    val_proc = transform_data(val_df, preprocessor, target_col="live_birth")
    test_proc = transform_data(test_df, preprocessor, target_col="live_birth")

    # Save processed CSVs
    train_path = os.path.join(output_dir, "processed_train.csv")
    val_path = os.path.join(output_dir, "processed_val.csv")
    test_path = os.path.join(output_dir, "processed_test.csv")

    train_proc.to_csv(train_path, index=False)
    val_proc.to_csv(val_path, index=False)
    test_proc.to_csv(test_path, index=False)

    # Save meta dataframes for unscaled demographic fairness evaluation
    meta_cols = ["age", "bmi", "diag", "live_birth"]
    train_df[meta_cols].to_csv(os.path.join(output_dir, "processed_train_meta.csv"), index=False)
    val_df[meta_cols].to_csv(os.path.join(output_dir, "processed_val_meta.csv"), index=False)
    test_df[meta_cols].to_csv(os.path.join(output_dir, "processed_test_meta.csv"), index=False)

    # Save preprocessor artifact
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)

    # Save feature schema for inference and web application
    default_vals = {}
    for col in numeric_cols:
        default_vals[col] = float(train_df[col].median())
    for col in categorical_cols:
        default_vals[col] = str(train_df[col].mode().iloc[0])

    schema = {
        "target": "live_birth",
        "sheet": target_sheet,
        "raw_feature_columns": feature_cols,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "categorical_options": {
            "diag": sorted([str(x) for x in train_df["diag"].dropna().unique()]),
            "trigger_type": sorted([str(x) for x in train_df["trigger_type"].dropna().unique()])
        },
        "default_values": default_vals,
        "processed_feature_columns": preprocessor["all_feature_names"],
        "n_features_processed": len(preprocessor["all_feature_names"]),
        "n_samples_total": len(df_clean),
        "n_train": len(train_proc),
        "n_val": len(val_proc),
        "n_test": len(test_proc),
        "positive_rate": float(binary_target.mean()),
        "friendly_labels": FEATURE_LABELS
    }

    schema_path = os.path.join(output_dir, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    return {
        "train_proc": train_proc,
        "val_proc": val_proc,
        "test_proc": test_proc,
        "preprocessor": preprocessor,
        "schema": schema
    }


if __name__ == "__main__":
    inspect_and_summarize_dataset()
    results = prepare_dataset()
    print("Preprocessing completed successfully.")
