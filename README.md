# Combined-Cycle Power Plant Output Prediction

A machine learning regression workflow, Dockerized service, and Flask web application that predicts the net hourly electrical energy output (**PE**) of a Combined-Cycle Power Plant (CCPP) based on ambient operating conditions.

## Overview

Combined-Cycle Power Plants combine gas and steam turbines to generate electricity efficiently. Predicting power output accurately helps grid operators optimize energy dispatch and plant performance.

This project:
1. Explores and validates a cleaned dataset of **9,527 observations**.
2. Evaluates linear, polynomial, and tree-based regression algorithms using **5-fold cross-validation** inside a leakage-free `scikit-learn` pipeline.
3. Balances predictive accuracy against serialized file size to export a lightweight, production-ready model (`2.25 MB`) for a Flask web application.

---

## Dataset Features

The dataset (`data/cleaned_data.csv`) contains **9,527 rows** and **5 columns** with zero missing values:

| Feature | Description | Unit | Range |
| :--- | :--- | :--- | :--- |
| **`AT`** | Ambient Temperature | °C | `1.81` – `37.11` |
| **`V`** | Exhaust Vacuum | cm Hg | `25.36` – `81.56` |
| **`AP`** | Ambient Pressure | millibar | `992.89` – `1033.30` |
| **`RH`** | Relative Humidity | % | `25.56` – `100.16` |
| **`PE`** *(Target)* | Net Hourly Electrical Energy Output | MW | `420.26` – `495.76` |

---

## Model Selection & Results

An 80/20 train-test split (`7,621` training rows, `1,906` holdout test rows) was used to keep evaluation completely isolated from hyperparameter tuning.

### 5-Fold Cross-Validation Comparison

| Model | Feature Transform | Train RMSE (MW) | CV RMSE (MW) |
| :--- | :--- | :---: | :---: |
| **Random Forest Regressor** | Original features | 1.2791 | **3.4175** |
| **XGBoost Regressor** | Original features | 2.9762 | 3.5396 |
| **Ridge Regression** | Polynomial (Degree 3) | 4.1819 | 4.1955 |
| **Ridge Regression** | Polynomial (Degree 2) | 4.2473 | 4.2560 |
| **Linear Regression** | Original features | 4.5508 | 4.5524 |
| **Ridge Regression** | Original features | 4.5508 | 4.5524 |
| **Lasso Regression** | Original features | 4.5508 | 4.5525 |

### Deployment Size vs. Accuracy Trade-Off
Because unrestricted Random Forests produce large serialized files, a constrained Random Forest (`n_estimators=50`, `max_depth=12`, `min_samples_leaf=2`) was trained for web deployment:

| Model Variant | CV RMSE | Test MAE | Test RMSE | Test R² | Compressed Size |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full CV Winner** | 3.4175 MW | 2.4168 MW | 3.4484 MW | 0.9595 | 14.37 MB |
| **Compact Random Forest (Deployed)** | 3.5149 MW | 2.5503 MW | 3.5570 MW | 0.9569 | **2.25 MB** |

The compact pipeline reduces storage size by **~84%** while maintaining an **R² score of 0.9569** on unseen test data.

---

## Project Structure

```text
├── app/                             # Flask web application and UI templates
├── data/
│   └── cleaned_data.csv             # Cleaned CCPP operating dataset
├── model/
│   └── best_model_compact.joblib    # Compressed scikit-learn deployment pipeline
├── notebooks/                       # EDA, model comparison, tuning, and export
├── Dockerfile                       # Container setup for deployment
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd Combined-cycle-power-plant
```

### 2. Local Setup
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Run with Docker (Optional)
```bash
docker build -t ccpp-predictor .
docker run -p 5000:5000 ccpp-predictor
```
