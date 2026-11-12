import os
import pandas as pd
import numpy as np

# Dataset paths
INPUT_PATH = "app_user_behavior_dataset.csv"
OUTPUT_PATH = "outputs/cleaned_app_user_data.csv"


def load_dataset(file_path):
    """Load the original dataset."""

    df = pd.read_csv(file_path)

    print("Original dataset loaded successfully.")
    return df


def remove_duplicates(df):
    """Remove duplicate records."""

    initial_count = len(df)

    df = df.drop_duplicates().copy()

    removed_count = initial_count - len(df)

    print(f"Duplicate records removed: {removed_count}")

    return df


def handle_missing_values(df):
    """Handle missing values without removing user records."""

    missing_before = df.isnull().sum()

    print("\nMissing values before treatment:")
    print(missing_before[missing_before > 0])

    # Preserve missing ratings because they may indicate
    # that the user has not submitted an app rating.

    print("\nMissing rating values will be preserved.")
    print("Other missing values will be investigated separately.")

    return df


def validate_data(df):
    """Validate dataset values."""

    print("\nData validation:")

    print("Duplicate user IDs:", df["user_id"].duplicated().sum())

    print("Invalid ages:", ((df["age"] < 0) | (df["age"] > 100)).sum())

    print(
        "Negative session counts:",
        (df["sessions_per_week"] < 0).sum()
    )

    print(
        "Invalid rating values:",
        (~df["rating_given"].dropna().between(1, 5)).sum()
    )

    print(
        "Invalid churn scores:",
        (~df["churn_risk_score"].between(0, 1)).sum()
    )


def save_cleaned_data(df, file_path):
    """Save the cleaned dataset."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)

    print("\nCleaned dataset saved successfully.")
    print("Output file:", file_path)
    print("Final dataset shape:", df.shape)


def main():

    print("========== DATA CLEANING ==========")

    df = load_dataset(INPUT_PATH)

    df = remove_duplicates(df)

    df = handle_missing_values(df)

    validate_data(df)

    save_cleaned_data(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
