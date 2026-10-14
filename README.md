
# Federated Learning for IVF Live-Birth Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-FedAvg-green.svg)](https://flower.ai/)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-Fairness-blueviolet.svg)](https://fairlearn.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Demo-red.svg)](https://flask.palletsprojects.com/)

An end-to-end privacy-preserving machine learning pipeline for in-vitro fertilisation (IVF) live-birth outcome prediction. Built as an academic college capstone project demonstrating how decentralized healthcare institutions can collaboratively train high-performing clinical neural networks without ever transmitting sensitive electronic health records (EHR).

---

## 1. Project Objective

In-vitro fertilisation (IVF) success depends on a complex interplay of patient demographics, ovarian reserve markers, stimulation protocols, and embryology quality. While artificial intelligence can assist reproductive specialists in prognostic estimation, healthcare privacy regulations (e.g., HIPAA, GDPR) strictly prohibit centralizing identifiable patient data across clinics.

This project implements **Federated Learning using FedAvg (Federated Averaging)** across **10 virtual IVF clinics**, benchmarking it against both a **Centralized Pooled Baseline** and **Isolated Local-Only Baselines**, while evaluating algorithmic fairness across patient age cohorts, BMI classifications, and infertility diagnoses.

---

## 2. Why Federated Learning?

> *"Instead of sending patient data to a central server, each clinic trains a model locally and only sends model updates. The server aggregates these updates using FedAvg."*

### The Core Healthcare Dilemma:
- **Small Clinic Data Silos:** An individual clinic often treats only a few hundred patients per year. Models trained on small local cohorts overfit, suffer high variance, and generalize poorly to external patient populations.
- **Privacy & Regulatory Barriers:** Hospitals cannot pool raw IVF records, patient ages, cycle dates, or medical notes into a central database.

### The Federated Learning Guarantee:
1. **Data Never Moves:** Patient data remains strictly locked inside each clinic's private database.
2. **Only Model Weights Move:** Clinics receive the global model, train locally on their private records for 2 epochs, and send back only updated mathematical parameters (NumPy weight matrices).
3. **Federated Aggregation:** The coordinating server computes a sample-weighted average (FedAvg) of the weights and disseminates an improved global model back to all clinics.
4. **Result:** The federated model learns from the collective clinical experience of all 10 clinics without any clinic ever observing another clinic's patients.

---

## 3. Real Clinical IVF Dataset

The pipeline utilizes a real clinical IVF dataset of freeze-all embryo transfer cycles:
- **Source file:** `data/ivf_data.xlsx` (Worksheet: `'tot cycles LB outcome'`)
- **Total records:** 1,578 completed IVF cycles
- **Target Variable:** `live_birth` (Binary: `1` = one or more live births, `0` = no live birth)
- **Class Balance:** 871 positive cases (55.20%) and 707 negative cases (44.80%)
- **Stratified Split:** 70% Training (1,104 samples), 10% Validation (158 samples), 20% Unseen Test (316 samples)

### Strict Data Leakage Safeguards:
- **Excluded Identifiers:** Cycle dates (`tarih`, `date`), patient ID (`pno`), date of birth (`dtarih`), clinical text notes (`note`).
- **Excluded Leakage Outcome Fields:** `PR` (post-implantation pregnancy test / pregnancy rate) was removed because it directly reveals chemical pregnancy status prior to delivery, creating unrealistic artificial accuracy.
- **No Synthetic Fabrication:** All 1,578 records are real clinical observations; no artificial rows were synthesized.

### Feature Inventory (24 Clinical Predictors -> 35 Encoded Features):
- **Demographics & Reserve:** Age, Infertility duration (years), Primary vs. Secondary infertility, Prior transfer count, Antral Follicle Count (AFC), BMI.
- **Diagnosis & Protocol:** Diagnostic code (`m` male factor, `pcos`, `dor`, `u` unexplained, `t` tubal, `o` ovulatory, etc.), Ovulation trigger type (GnRH Agonist, Dual Trigger, hCG).
- **Ovarian Stimulation:** Stimulation duration (days), Daily FSH dose, Daily HMG dose, Total FSH administered.
- **Embryology Laboratory:** Total eggs retrieved, ICSI injected oocytes, 2PN fertilised embryos, Blastocysts >= 2BB, Frozen blastocysts >= 2BB, Oocyte maturity rate, Fertilisation rate, Blastocyst formation rate.
- **Transfer & Endometrium:** FET attempt number, Number of blastocysts transferred, SET embryo score, Endometrial thickness (mm).

---

## 4. System Architecture

```
                  ┌───────────────────────────────────────────────┐
                  │          FLOWER FEDERATED SERVER              │
                  │        FedAvg Parameter Aggregation           │
                  └──────┬─────────────────▲──────────────────────┘
                         │                 │
           Global Model  │                 │ Local Weight Updates
           Weights W_t   │                 │ ΔW_i (NumPy Arrays)
                         ▼                 │
      ┌─────────────────────────────────────────────────────────────────┐
      │  CLINIC 0      CLINIC 1      CLINIC 2  ...      CLINIC 9        │
      │  111 records   111 records   111 records        110 records     │
      │  Local IVFNet  Local IVFNet  Local IVFNet       Local IVFNet    │
      │  🔒 Private    🔒 Private    🔒 Private         🔒 Private      │
      │     EHR Data      EHR Data      EHR Data           EHR Data     │
      └─────────────────────────────────────────────────────────────────┘
```

### Deep Neural Network (`models/neural_net.py`):
```
Input Layer (35 features)
       ↓
Linear (128) → ReLU → Dropout (0.3)
       ↓
Linear (64)  → ReLU → Dropout (0.3)
       ↓
Linear (32)  → ReLU
       ↓
Linear (1)   → Sigmoid → Live-Birth Probability [0.0, 1.0]
```
- **Loss Function:** Binary Cross-Entropy (`BCELoss`)
- **Optimizer:** Adam (Learning Rate = 0.001, Weight Decay = 1e-4)

---

## 5. Project Directory Structure

```
federated_ivf/
│
├── data/                                # Dataset storage and partitions
│   ├── ivf_data.xlsx                    # Original clinical Excel dataset
│   ├── processed_train.csv              # 70% training split (1,104 samples)
│   ├── processed_val.csv                # 10% validation split (158 samples)
│   ├── processed_test.csv               # 20% test split (316 samples)
│   ├── processed_test_meta.csv          # Raw demographic metadata for fairness
│   ├── preprocessor.joblib              # Fitted imputation and scaling pipeline
│   ├── feature_schema.json              # Feature metadata and default values
│   ├── clients/                         # 10 IID virtual clinic datasets (client_0 to 9)
│   ├── clients_age_skew/                # 10 Non-IID age-skewed clinic datasets
│   └── clients_diagnosis_skew/          # 10 Non-IID diagnosis-skewed clinic datasets
│
├── preprocessing/                       # Data processing modules
│   ├── __init__.py
│   ├── preprocess.py                    # Inspection, leakage removal, pipeline fitting
│   └── partition.py                     # IID, Age-skew, and Diagnosis-skew partitioning
│
├── models/                              # Neural network architecture
│   ├── __init__.py
│   └── neural_net.py                    # PyTorch IVFNet & Flower weight conversion
│
├── baselines/                           # Benchmark models
│   ├── __init__.py
│   ├── centralized.py                   # Centralized model on full pooled data
│   └── local_only.py                    # 10 isolated clinic models (no collaboration)
│
├── fl_simulation/                       # Federated Learning framework
│   ├── __init__.py
│   ├── client.py                        # Flower NumPyClient implementation
│   ├── server.py                        # FedAvg strategy & metric aggregation
│   └── run_simulation.py                # 30-round simulation orchestrator
│
├── evaluation/                          # Evaluation and auditing
│   ├── __init__.py
│   ├── metrics.py                       # AUC-ROC, F1, Brier Score, ECE calibration
│   └── fairness.py                      # Fairlearn demographic parity & equalized odds
│
├── results/                             # Saved experiment results & figures
│   ├── figures/                         # Presentation-ready plots
│   │   ├── federated_auc_convergence.png
│   │   ├── model_comparison.png
│   │   └── fairness_metrics.png
│   ├── dataset_summary.txt              # Complete clinical dataset breakdown
│   ├── centralized_results.json         # Centralized benchmark scores
│   ├── local_results.json               # Per-clinic & average local scores
│   ├── fl_metrics.json                  # Federated Learning convergence log
│   ├── fl_iid_metrics.json              # IID federated run log
│   ├── fl_age_skew_metrics.json         # Non-IID age-skew run log
│   ├── fl_diagnosis_skew_metrics.json   # Non-IID diagnosis-skew run log
│   ├── comparison_results.json          # Consolidated experimental results
│   ├── fairness_report.txt              # Fairlearn demographic parity audit
│   ├── final_comparison.txt             # Final console summary table
│   └── federated_model.pt               # Trained global federated model weights
│
├── templates/                           # Flask UI templates
│   └── index.html                       # Responsive clinical prediction interface
├── app.py                               # Interactive Flask web demonstration
├── main.py                              # End-to-end pipeline execution script
├── requirements.txt                     # Python package dependencies
└── README.md                            # Comprehensive project documentation
```

---

## 6. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Windows, macOS, or Linux

### Step 1: Clone or Navigate to Project
```bash
cd federated_ivf
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 7. How to Run

### Execute the Complete Pipeline:
```bash
python main.py
```
This single command automatically runs all 10 stages:
1. Dataset inspection (`results/dataset_summary.txt`)
2. Leakage-free preprocessing (`data/processed_*.csv`, `data/preprocessor.joblib`)
3. Virtual clinic partitioning (IID + Non-IID)
4. Centralized baseline training (`results/centralized_results.json`)
5. Local-only baseline training on 10 isolated clinics (`results/local_results.json`)
6. Federated learning simulation for 30 communication rounds (`results/fl_metrics.json`, `results/federated_model.pt`)
7. Non-IID benchmarking across age and diagnosis skews
8. Fairlearn algorithmic fairness evaluation (`results/fairness_report.txt`)
9. Visualization generation (`results/figures/`)
10. Final comparison table (`results/final_comparison.txt`)

### Launch the Interactive Web Demonstration:
```bash
python app.py
```
Open your browser at:
```
http://127.0.0.1:5000
```
Enter patient parameters (e.g., female age, AFC, embryo quality) to test live inference using the privacy-preserving federated model.

---

## 8. Experimental Results & Viva Discussion

### Final Model Comparison (Unseen Test Set, n=316)

```
========================================================
             FINAL MODEL COMPARISON
========================================================
Model              AUC       F1       Brier Score
--------------------------------------------------------
Centralized        0.657     0.688    0.237
Federated          0.647     0.718    0.236
Local-only         0.587     0.623    0.338
========================================================

Dataset:            Real IVF Treatment Dataset (Freeze-all IVF)
Number of samples:  1,578 (1,104 train, 158 val, 316 test)
Number of features: 35
Number of clinics:  10
Federated rounds:   30
Best federated AUC: 0.647
```

### Key Analytical Takeaways for Viva:
1. **The Isolated Silo Penalty:** When clinics train solely on their own private data (Local-Only Baseline), average AUC drops to **0.587** with an elevated Brier error of **0.338**. Small clinical sample sizes (~110 patients) lead to severe generalization error.
2. **Federated Parity:** Federated Learning achieves an AUC of **0.647**, closely approaching the centralized ceiling (**0.657**), while maintaining superior calibration (Brier score **0.236**).
3. **Privacy Without Sacrifice:** Collaborative learning achieves ~98.5% of centralized model discrimination without transmitting a single patient record outside hospital firewalls.

---

## 9. Non-IID Heterogeneity Benchmarks

In real-world medicine, patient distributions are never identically distributed across clinics. We evaluated two realistic clinical divergence scenarios:

1. **Age-Skew Non-IID:** Clinics represent age-specialized facilities (e.g., Clinic 0 treats younger patients median age 22.9 yrs, whereas Clinic 9 treats advanced maternal age patients median age 40.6 yrs).
2. **Diagnosis-Skew Non-IID:** Clinics specialize in specific etiologies (e.g., Male factor clinics, PCOS centers, Endometriosis/Tubal centers, and Unexplained infertility centers).

### Benchmark Comparison:
| Partitioning Strategy | Client Distribution | Final Test AUC | Observations |
|-----------------------|----------------------|----------------|--------------|
| **IID (Homogeneous)** | Uniform random split | **0.647** | Smooth, steady convergence curve |
| **Age-Skew Non-IID** | Binned by maternal age | **0.638** | Slight convergence variance due to demographic shift |
| **Diagnosis-Skew Non-IID** | Clustered by etiology | **0.635** | Moderate parameter drift mitigated by FedAvg |

---

## 10. Algorithmic Fairness Audit (`evaluation/fairness.py`)

Using the **Fairlearn** framework, the global federated model was audited across sensitive demographic groups:

### 1. Maternal Age Cohorts:
- **18–34 yrs (Young):** Selection Rate = 88.9%, TPR (Recall) = 92.5%
- **35–37 yrs (Intermediate):** Selection Rate = 73.3%, TPR (Recall) = 90.5%
- **38–39 yrs (Advanced 1):** Selection Rate = 40.6%, TPR (Recall) = 50.0%
- **40+ yrs (Advanced 2):** Selection Rate = 59.1%, TPR (Recall) = 66.7%
- **Demographic Parity Difference:** 0.483
- *Clinical Insight:* In reproductive endocrinology, live-birth rates naturally decline with oocyte aneuploidy at advanced maternal age. The model's lower selection rate for older cohorts reflects underlying biological fecundity rather than purely artificial algorithmic bias.

### 2. BMI Classifications:
- **Normal (<25 kg/m²):** Selection Rate = 85.4%
- **Overweight (25–29.9 kg/m²):** Selection Rate = 78.0%
- **Obese (>=30 kg/m²):** Selection Rate = 72.7%
- **Demographic Parity Difference:** 0.126 (Relatively equitable parity across weight classes).

---

## 11. Clinical Disclaimer & Limitations

> **EDUCATIONAL RESEARCH DISCLAIMER:**  
> This project is developed solely for academic research and computer science educational demonstration. It is NOT medical software, a certified medical device, or a clinical decision support tool. It does not replace the expert clinical judgment of licensed reproductive endocrinologists, embryologists, or fertility specialists.

### Limitations:
1. **Single-Center Retrospective Origin:** While partitioned into 10 virtual clinics, the raw dataset originates from a single clinical source. Prospective evaluation across geographically diverse health systems is required for real-world translation.
2. **Communication Overhead:** Federated learning incurs network communication rounds, although transmitting lightweight weight vectors (35-dimension neural net, ~50 KB) requires minimal bandwidth.
3. **Biological Variance in Reproductive Outcomes:** Embryo implantation involves unmeasured genetic and immunological variables not captured in standard clinical charts, which establishes a natural ceiling on tabular IVF predictive accuracy (~0.65–0.70 AUC).

---

## 12. Future Work

- **Differential Privacy (DP):** Integrating DP-SGD to mathematically bound information leakage against model inversion attacks.
- **Personalized Federated Learning:** Exploring pFedMe or FedProx to provide local personalization heads adapted to individual clinic demographics.
- **Multimodal FL:** Combining structured tabular clinical records with time-lapse blastocyst microscopy imagery via vision transformers.

---

