
import os
import pandas as pd
import numpy as np

# File paths
INPUT_PATH = "outputs/cleaned_app_user_data.csv"
OUTPUT_PATH = "outputs/engineered_app_user_data.csv"


# Features selected for behavioral segmentation
BEHAVIORAL_FEATURES = [
    "sessions_per_week",
    "avg_session_duration_min",
    "daily_active_minutes",
    "feature_clicks_per_session",
    "notifications_opened_per_week",
    "in_app_search_count",
    "pages_viewed_per_session",
    "crash_events_last_30_days",
    "support_tickets_raised",
    "days_since_last_login",
    "ads_clicked_last_30_days",
    "content_downloads",
    "social_shares",
    "account_age_days"
]

ENGINEERED_FEATURES = [
    "usage_intensity",
    "interaction_intensity",
    "content_engagement",
    "app_friction_score"
]

def load_data():
    """Load cleaned dataset."""

    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully.")
    print("Original shape:", df.shape)

    return df


def investigate_features(df):
    """Investigate feature relationships and redundancy."""

    print("\n========== FEATURE INVESTIGATION ==========")

    print("\n1. Engagement Score Correlation:")

    print(
        df[BEHAVIORAL_FEATURES + ["engagement_score"]]
        .corr()["engagement_score"]
        .sort_values(ascending=False)
    )

    print("\n2. Churn Risk Score Correlation:")

    print(
        df[BEHAVIORAL_FEATURES + ["churn_risk_score"]]
        .corr()["churn_risk_score"]
        .sort_values(ascending=False)
    )

    print("\n3. Highly Correlated Behavioral Features:")

    correlation_matrix = df[BEHAVIORAL_FEATURES].corr()

    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )

    high_correlations = (
        upper_triangle.stack()
        .loc[lambda x: x.abs() >= 0.80]
    )

    if high_correlations.empty:
        print("No feature pairs with absolute correlation >= 0.80.")
    else:
        print(high_correlations)


def create_behavioral_features(df):
    print("\n========== CREATING BEHAVIORAL FEATURES ==========")

    df["usage_intensity"] = (
        df["sessions_per_week"] *
        df["avg_session_duration_min"]
    )

    df["interaction_intensity"] = (
        df["feature_clicks_per_session"] +
        df["in_app_search_count"] +
        df["pages_viewed_per_session"]
    )

    df["content_engagement"] = (
        df["content_downloads"] +
        df["social_shares"]
    )

    df["app_friction_score"] = (
        df["crash_events_last_30_days"] +
        df["support_tickets_raised"]
    )

    print("New behavioural features created.")

    print("\nNew feature summary:")
    print(df[ENGINEERED_FEATURES].describe().T)

    return df

def analyze_engineered_correlations(df):
    print("\n========== ENGINEERED FEATURE CORRELATIONS ==========")

    columns = BEHAVIORAL_FEATURES + ENGINEERED_FEATURES

    correlation_matrix = df[columns].corr()

    print("\nCorrelation with engineered features:")

    print(
        correlation_matrix[ENGINEERED_FEATURES]
        .round(3)
    )

    print("\nCorrelation with engagement score:")

    print(
        df[columns + ["engagement_score"]]
        .corr()["engagement_score"]
        .sort_values(ascending=False)
    )

    print("\nCorrelation with churn risk score:")

    print(
        df[columns + ["churn_risk_score"]]
        .corr()["churn_risk_score"]
        .sort_values(ascending=False)
    )


def select_final_features(df):
    print("\n========== FINAL FEATURE SELECTION ==========")

    final_features = [
        "usage_intensity",
        "interaction_intensity",
        "content_engagement",
        "notifications_opened_per_week",
        "days_since_last_login",
        "app_friction_score"
    ]

    selected_df = df[final_features].copy()

    print("Selected features:", final_features)
    print("Dataset shape:", selected_df.shape)

    print("\nCorrelation matrix:")
    print(selected_df.corr().round(3))

    return selected_df

def inspect_feature_variation(df):
    print("\n========== FEATURE VARIATION ANALYSIS ==========")

    for column in BEHAVIORAL_FEATURES:
        print(f"\n{column}")
        print("Unique values:", df[column].nunique())
        print("Standard deviation:", round(df[column].std(), 4))
        print("Minimum:", df[column].min())
        print("Maximum:", df[column].max())
def inspect_feature_relationships(df):
    print("\n========== FEATURE RELATIONSHIP ANALYSIS ==========")

    correlation_matrix = df[BEHAVIORAL_FEATURES].corr()

    print("\nCorrelation with sessions_per_week:")
    print(
        correlation_matrix["sessions_per_week"]
        .sort_values(ascending=False)
    )

    print("\nCorrelation with daily_active_minutes:")
    print(
        correlation_matrix["daily_active_minutes"]
        .sort_values(ascending=False)
    )

    print("\nCorrelation with days_since_last_login:")
    print(
        correlation_matrix["days_since_last_login"]
        .sort_values(ascending=False)
    )

def select_final_features(df):
    print("\n========== FINAL FEATURE SELECTION ==========")

    final_features = [
        "usage_intensity",
        "interaction_intensity",
        "content_engagement",
        "notifications_opened_per_week",
        "days_since_last_login",
        "app_friction_score"
    ]

    selected_df = df[final_features].copy()

    print("Selected features:", final_features)
    print("Dataset shape:", selected_df.shape)

    print("\nCorrelation matrix:")
    print(selected_df.corr().round(3))

    selected_df.to_csv(
        "outputs/candidate_app_user_data.csv",
        index=False
    )

    print("\nSix-feature candidate dataset saved successfully.")

    return selected_df

def validate_features(features):

    print("\n========== FEATURE VALIDATION ==========")

    print("Feature dataset shape:", features.shape)

    print("\nMissing values:")
    print(features.isnull().sum())

    print("\nInfinite values:")
    print(np.isinf(features.select_dtypes(include=np.number)).sum())

    print("\nStatistical summary:")
    print(features.describe().T)


def save_features(features):

    os.makedirs("outputs", exist_ok=True)

    features.to_csv(OUTPUT_PATH, index=False)

    print("\nEngineered feature dataset saved successfully.")
    print("Saved location:", OUTPUT_PATH)


def main():

    print("\n========== FEATURE ENGINEERING ==========")

    df = load_data()

    df = create_behavioral_features(df)

    analyze_engineered_correlations(df)

    final_features = select_final_features(df)

    investigate_features(df)

    inspect_feature_variation(df)

    inspect_feature_relationships(df)

    validate_features(final_features)
    save_features(final_features)

    print("\nFeature engineering completed successfully.")


if __name__ == "__main__":
    main()
