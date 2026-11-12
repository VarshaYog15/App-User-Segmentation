print("DATA UNDERSTANDING SCRIPT STARTED")
import pandas as pd

# Dataset path
DATA_PATH = "app_user_behavior_dataset.csv"


def load_dataset(file_path):
    """Load the app user behavior dataset."""
    df = pd.read_csv(file_path)
    return df


def inspect_dataset(df):
    """Display basic information about the dataset."""

    print("\n========== DATASET OVERVIEW ==========")

    print("\n1. Dataset Shape:")
    print(df.shape)

    print("\n2. First Five Records:")
    print(df.head())

    print("\n3. Column Names:")
    print(df.columns.tolist())

    print("\n4. Data Types:")
    print(df.dtypes)

    print("\n5. Dataset Information:")
    df.info()

    print("\n6. Statistical Summary:")
    print(df.describe(include="all").T)

    print("\n7. Missing Values:")
    print(df.isnull().sum())

    print("\n8. Duplicate Records:")
    print(df.duplicated().sum())

    print("\n9. Unique Values in Categorical Columns:")

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:
        print(f"\n{column}:")
        print(df[column].value_counts())


def main():
    df = load_dataset(DATA_PATH)
    inspect_dataset(df)


if __name__ == "__main__":
    main()
