"""
Model training and evaluation module for Iris Flower Classification.

Trains a Random Forest Classifier and evaluates it with standard metrics.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import (
    DEFAULT_DATA_PATH,
    PROJECT_ROOT,
    preprocess_pipeline,
)

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "iris_model.pkl"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier on the training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        n_estimators: Number of trees in the forest.
        random_state: Random seed for reproducibility.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_encoder=None) -> dict:
    """
    Evaluate the classifier using accuracy, confusion matrix, and classification report.

    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: True test labels.
        label_encoder: Optional encoder to map class indices to species names.

    Returns:
        Dictionary with accuracy, predictions, confusion matrix, and report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    target_names = None
    if label_encoder is not None:
        target_names = label_encoder.classes_.tolist()

    report = classification_report(
        y_test, y_pred, target_names=target_names
    )

    return {
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def print_evaluation_results(results: dict) -> None:
    """Print model evaluation metrics to the console."""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"\nAccuracy: {results['accuracy_percent']:.2f}%")
    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])
    print("\nClassification Report:")
    print(results["classification_report"])


def save_model(model, path: Path | str = DEFAULT_MODEL_PATH) -> Path:
    """
    Persist the trained model to disk using Joblib.

    Args:
        model: Fitted classifier.
        path: Output file path (.pkl).

    Returns:
        Path where the model was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved to: {path}")
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list | None = None,
    save_path: Path | None = None,
) -> None:
    """
    Visualize the confusion matrix using Seaborn heatmap.

    Args:
        cm: Confusion matrix array.
        class_names: Labels for each class.
        save_path: Optional path to save the figure.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - Iris Flower Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix plot saved to: {save_path}")
    plt.close()


def run_training_pipeline(
    data_path: Path | str = DEFAULT_DATA_PATH,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    save_plots: bool = True,
) -> dict:
    """
    Execute the full training and evaluation pipeline.

    Returns:
        Dictionary with model, evaluation results, and preprocessing artifacts.
    """
    data = preprocess_pipeline(data_path=data_path, verbose=True)

    print("\n" + "=" * 50)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 50)

    model = train_random_forest(data["X_train"], data["y_train"])
    results = evaluate_model(
        model,
        data["X_test"],
        data["y_test"],
        label_encoder=data["label_encoder"],
    )
    print_evaluation_results(results)

    save_model(model, model_path)

    if save_plots:
        plot_path = PLOTS_DIR / "confusion_matrix.png"
        plot_confusion_matrix(
            results["confusion_matrix"],
            class_names=data["label_encoder"].classes_.tolist(),
            save_path=plot_path,
        )

    return {
        "model": model,
        "results": results,
        **data,
    }


if __name__ == "__main__":
    run_training_pipeline()
