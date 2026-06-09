# Iris Flower Classification

A complete **Machine Learning** project in Python that classifies iris flowers into three species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*) using measurements of sepal and petal dimensions. The project uses the classic **Iris dataset** (Kaggle-style CSV format), **Pandas** for data handling, **Matplotlib** and **Seaborn** for visualization, and a **Random Forest Classifier** for prediction.

---

## Project Overview

| Aspect | Details |
|--------|---------|
| **Dataset** | 150 samples, 4 features, 3 classes |
| **Algorithm** | Random Forest Classifier |
| **Train/Test Split** | 80% train / 20% test (stratified) |
| **Evaluation** | Accuracy, Confusion Matrix, Classification Report |
| **Model Persistence** | Joblib (`.pkl`) |

The pipeline covers:

1. Data loading and preprocessing (missing values, label encoding)
2. Exploratory Data Analysis (EDA) with visualizations
3. Model training and evaluation
4. Saving and loading the trained model for inference

---

## Project Structure

```
iris-flower-classification/
│
├── data/
│   └── Iris.csv              # Iris dataset (Kaggle format)
│
├── notebooks/
│   └── iris_analysis.ipynb   # Interactive EDA and analysis
│
├── src/
│   ├── data_preprocessing.py   # Load, clean, encode, split data
│   ├── train_model.py          # Train, evaluate, save model
│   └── predict.py              # Load model and predict species
│
├── models/
│   └── iris_model.pkl          # Saved model (created after training)
│
├── outputs/
│   └── plots/                  # Generated EDA and evaluation plots
│
├── requirements.txt
├── README.md
└── main.py                     # Full pipeline entry point
```

---

## Prerequisites

- **Python 3.9+** (recommended 3.10 or 3.11)
- `pip` package manager

---

## Setup Instructions

### 1. Clone or download the project

Place the project folder on your machine, for example:

```
Desktop/Iris Flower Classification/
```

### 2. Create a virtual environment (recommended)

**Windows (PowerShell):**

```powershell
cd "path\to\Iris Flower Classification"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
cd path/to/Iris\ Flower\ Classification
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset

The file `data/Iris.csv` is included in the project with Kaggle-compatible columns:

- `Id`, `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`, `Species`

To use the official Kaggle file instead, download from [Kaggle – Iris Species](https://www.kaggle.com/datasets/uciml/iris) and replace `data/Iris.csv`.

---

## How to Run

### Option A: Full pipeline (`main.py`)

Runs preprocessing, EDA, training, evaluation, and sample predictions in one command:

```bash
python main.py
```

**Expected output includes:**

- Dataset info and missing value check
- First 5 rows and class distribution
- Model accuracy (typically **95–100%**)
- Confusion matrix and classification report
- Saved model at `models/iris_model.pkl`
- Plots under `outputs/plots/`

### Option B: Train model only

```bash
cd src
python train_model.py
```

### Option C: Run predictions

After training, from the `src` directory:

```bash
python predict.py
```

### Option D: Jupyter notebook

```bash
jupyter notebook notebooks/iris_analysis.ipynb
```

Run all cells for step-by-step EDA and model training in the notebook.

---

## Module Reference

### `data_preprocessing.py`

- `load_dataset()` – Load CSV with Pandas
- `check_missing_values()` – Missing value audit
- `display_dataset_info()` – Shape, dtypes, summary stats
- `encode_target_labels()` – LabelEncoder for `Species`
- `split_train_test()` – 80/20 stratified split
- `preprocess_pipeline()` – End-to-end preprocessing

### `train_model.py`

- `train_random_forest()` – Fit Random Forest
- `evaluate_model()` – Accuracy, confusion matrix, report
- `save_model()` – Persist with Joblib
- `run_training_pipeline()` – Full train + evaluate + save

### `predict.py`

- `load_model()` – Load `.pkl` from `models/`
- `predict_single()` – Predict one flower by measurements
- `predict_batch()` – Predict from a DataFrame

---

## Example: Custom Prediction

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src")))
from predict import load_model, predict_single
from data_preprocessing import encode_target_labels, load_dataset

model = load_model("models/iris_model.pkl")
_, le = encode_target_labels(load_dataset())

species = predict_single(
    model,
    sepal_length=5.1,
    sepal_width=3.5,
    petal_length=1.4,
    petal_width=0.2,
    label_encoder=le,
)
print(f"Predicted species: {species}")
```

---

## Results

On the standard Iris dataset, the Random Forest classifier typically achieves:

- **Accuracy:** 95%–100% on the test set
- Clear separation of *setosa* vs. *versicolor* / *virginica* in feature space

Evaluation artifacts:

- Console: accuracy percentage, confusion matrix, classification report
- File: `outputs/plots/confusion_matrix.png`

---

## Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Model, metrics, preprocessing |
| matplotlib | Plotting |
| seaborn | Statistical visualizations |
| joblib | Model serialization |
| jupyter | Notebook environment |

---

## Author & License

Educational ML project for iris flower species classification. The Iris dataset is in the public domain (Fisher, 1936; UCI ML Repository).

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError` for CSV | Ensure `data/Iris.csv` exists |
| `Model not found` | Run `python main.py` or `python src/train_model.py` first |
| Plots not showing in notebook | Use `%matplotlib inline` at the top of the notebook |
