"""
Iris Flower Classification - Main Entry Point

Orchestrates data loading, preprocessing, EDA, training, evaluation, and prediction.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    check_missing_values,
    display_dataset_info,
    encode_target_labels,
    load_dataset,
    preprocess_pipeline,
)
from predict import run_sample_predictions
from train_model import run_training_pipeline


def run_eda(df, output_dir: Path | None = None) -> None:
    """
    Perform Exploratory Data Analysis with visualizations.

    Args:
        df: Raw Iris DataFrame.
        output_dir: Directory to save plot images (optional).
    """
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 50)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Class Distribution ---")
    class_counts = df[TARGET_COLUMN].value_counts()
    print(class_counts)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # 1. Class distribution bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    class_counts.plot(kind="bar", ax=ax, color=["#5B9BD5", "#ED7D31", "#70AD47"])
    ax.set_title("Iris Species Class Distribution")
    ax.set_xlabel("Species")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "class_distribution.png", dpi=150)
    plt.close()

    # 2. Pairplot of features colored by species
    feature_cols = [
        c
        for c in df.columns
        if c not in ("Id", TARGET_COLUMN)
    ]
    g = sns.pairplot(
        df,
        vars=feature_cols,
        hue=TARGET_COLUMN,
        palette="Set2",
        diag_kind="kde",
    )
    g.fig.suptitle("Feature Pairplot by Species", y=1.02)
    if output_dir:
        g.savefig(output_dir / "pairplot.png", dpi=150)
    plt.close("all")

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    numeric_df = df.drop(columns=["Id", TARGET_COLUMN], errors="ignore")
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, fmt=".2f")
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    # 4. Box plots per feature by species
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, col in enumerate(feature_cols):
        sns.boxplot(data=df, x=TARGET_COLUMN, y=col, ax=axes[idx], palette="Set2")
        axes[idx].set_title(f"{col} by Species")
        axes[idx].tick_params(axis="x", rotation=45)
    plt.suptitle("Feature Distributions by Species", y=1.02)
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / "boxplots_by_species.png", dpi=150)
    plt.close()

    print("\nEDA visualizations completed.")
    if output_dir:
        print(f"Plots saved to: {output_dir}")


def main() -> None:
    """Run the complete Iris Flower Classification pipeline."""
    print("=" * 60)
    print("  IRIS FLOWER CLASSIFICATION - MACHINE LEARNING PROJECT")
    print("=" * 60)

    plots_dir = PROJECT_ROOT / "outputs" / "plots"

    # Step 1: Load data
    print("\n[1/5] Loading dataset...")
    df = load_dataset(DEFAULT_DATA_PATH)

    # Step 2: Preprocessing checks
    print("\n[2/5] Data preprocessing...")
    print("\n--- Missing Values ---")
    missing = check_missing_values(df)
    print(missing)
    display_dataset_info(df)

    # Step 3: EDA
    print("\n[3/5] Exploratory Data Analysis...")
    run_eda(df, output_dir=plots_dir)

    # Step 4: Train and evaluate
    print("\n[4/5] Training and evaluating model...")
    pipeline_result = run_training_pipeline(
        data_path=DEFAULT_DATA_PATH,
        save_plots=True,
    )

    accuracy_pct = pipeline_result["results"]["accuracy_percent"]
    print(f"\n>>> Final Model Accuracy: {accuracy_pct:.2f}% <<<")

    # Step 5: Sample predictions
    print("\n[5/5] Running sample predictions...")
    run_sample_predictions()

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
