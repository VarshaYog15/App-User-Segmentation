import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

INPUT_PATH = "outputs/engineered_app_user_data.csv"
OUTPUT_PATH = "outputs/scaled_app_user_data.csv"
SCALER_PATH = "models/standard_scaler.pkl"


def load_data():
    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df



def validate_data(df):
    print("\n========== DATA VALIDATION ==========")

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nInfinite values:")
    print(np.isinf(df.select_dtypes(include=np.number)).sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nFeature summary:")
    print(df.describe().T)


def analyze_outliers(df):
    print("\n========== OUTLIER ANALYSIS ==========")

    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_limit) |
            (df[column] > upper_limit)
        ]

        print(f"\n{column}")
        print("Outlier count:", len(outliers))
        print("Lower limit:", round(lower_limit, 2))
        print("Upper limit:", round(upper_limit, 2))


def scale_features(df):
    print("\n========== FEATURE SCALING ==========")

    scaler = StandardScaler()

    scaled_array = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=df.columns
    )

    print("\nScaling completed successfully.")

    print("\nScaled feature summary:")
    print(scaled_df.describe().T)

    return scaled_df, scaler

def save_preprocessing_outputs(scaled_df, scaler):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    scaled_df.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    print("\nScaled dataset saved:", OUTPUT_PATH)
    print("Scaler saved:", SCALER_PATH)


def main():
    print("\n========== DATA PREPROCESSING ==========")

    df = load_data()

    validate_data(df)

    analyze_outliers(df)

    scaled_df, scaler = scale_features(df)

    save_preprocessing_outputs(scaled_df, scaler)

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()