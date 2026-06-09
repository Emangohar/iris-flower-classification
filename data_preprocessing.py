"""
Data preprocessing module for the Iris Flower Classification project.

Handles loading, cleaning, encoding, and train-test splitting of the Iris dataset.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "Iris.csv"

# Feature and target column names (Kaggle Iris.csv format)
FEATURE_COLUMNS = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm",
]
TARGET_COLUMN = "Species"
ID_COLUMN = "Id"


def load_dataset(data_path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the Iris dataset from a CSV file using Pandas.

    Args:
        data_path: Path to the Iris.csv file.

    Returns:
        DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Check for missing values in each column.

    Args:
        df: Input DataFrame.

    Returns:
        Series with missing value counts per column.
    """
    return df.isnull().sum()


def display_dataset_info(df: pd.DataFrame) -> None:
    """
    Display dataset shape, column types, and memory usage.

    Args:
        df: Input DataFrame.
    """
    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"\nColumn names: {df.columns.tolist()}")
    print("\nData types:")
    print(df.dtypes)
    print("\nDetailed info:")
    df.info()
    print("\nStatistical summary:")
    print(df.describe(include="all"))


def encode_target_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode categorical species labels into numeric values.

    Args:
        df: DataFrame with a Species column.

    Returns:
        Tuple of (DataFrame with encoded 'Species' column, fitted LabelEncoder).
    """
    df = df.copy()
    label_encoder = LabelEncoder()
    df[TARGET_COLUMN] = label_encoder.fit_transform(df[TARGET_COLUMN])
    return df, label_encoder


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from the DataFrame.

    Drops the Id column if present.

    Args:
        df: Preprocessed DataFrame with encoded labels.

    Returns:
        Tuple of (features DataFrame, target Series).
    """
    df = df.copy()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data for testing (default 20%).
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def preprocess_pipeline(
    data_path: Path | str = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the full preprocessing pipeline.

    Args:
        data_path: Path to Iris.csv.
        test_size: Test set proportion.
        random_state: Random seed.
        verbose: Whether to print preprocessing steps.

    Returns:
        Dictionary with keys: df_raw, df_encoded, X_train, X_test,
        y_train, y_test, label_encoder.
    """
    df_raw = load_dataset(data_path)

    if verbose:
        print("\n--- Missing Values ---")
        missing = check_missing_values(df_raw)
        print(missing)
        if missing.sum() == 0:
            print("No missing values found.")
        display_dataset_info(df_raw)

    df_encoded, label_encoder = encode_target_labels(df_raw)
    X, y = prepare_features_and_target(df_encoded)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=random_state
    )

    if verbose:
        print("\n--- Train-Test Split ---")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples:  {len(X_test)}")

    return {
        "df_raw": df_raw,
        "df_encoded": df_encoded,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
    }
