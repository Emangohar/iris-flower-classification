"""
Prediction module for Iris Flower Classification.

Loads the saved model and predicts species for new flower measurements.
"""

from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

from data_preprocessing import FEATURE_COLUMNS, PROJECT_ROOT

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "iris_model.pkl"


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH):
    """
    Load a trained model from disk using Joblib.

    Args:
        model_path: Path to the saved .pkl file.

    Returns:
        Loaded classifier object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_model.py or main.py first."
        )
    return joblib.load(path)


def predict_single(
    model,
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    label_encoder=None,
) -> Union[int, str]:
    """
    Predict the iris species for a single flower sample.

    Args:
        model: Trained classifier.
        sepal_length: Sepal length in cm.
        sepal_width: Sepal width in cm.
        petal_length: Petal length in cm.
        petal_width: Petal width in cm.
        label_encoder: Optional encoder to return species name instead of index.

    Returns:
        Predicted class index or species name.
    """
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]

    if label_encoder is not None:
        return label_encoder.inverse_transform([prediction])[0]
    return prediction


def predict_batch(model, df: pd.DataFrame) -> np.ndarray:
    """
    Predict species for multiple samples in a DataFrame.

    Args:
        model: Trained classifier.
        df: DataFrame with feature columns (FEATURE_COLUMNS).

    Returns:
        Array of predicted class indices.
    """
    X = df[FEATURE_COLUMNS]
    return model.predict(X)


def run_sample_predictions(model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
    """
    Demonstrate predictions on example iris measurements.
    """
    from data_preprocessing import load_dataset, encode_target_labels

    model = load_model(model_path)
    df_raw = load_dataset()
    _, label_encoder = encode_target_labels(df_raw)

    # Example: typical Setosa measurements
    examples = [
        ("Setosa-like", 5.1, 3.5, 1.4, 0.2),
        ("Versicolor-like", 6.4, 3.2, 4.5, 1.5),
        ("Virginica-like", 6.3, 3.3, 6.0, 2.5),
    ]

    print("\n" + "=" * 50)
    print("SAMPLE PREDICTIONS")
    print("=" * 50)

    for name, sl, sw, pl, pw in examples:
        species = predict_single(
            model, sl, sw, pl, pw, label_encoder=label_encoder
        )
        print(f"\n{name}:")
        print(f"  Measurements: SL={sl}, SW={sw}, PL={pl}, PW={pw}")
        print(f"  Predicted species: {species}")


if __name__ == "__main__":
    run_sample_predictions()
