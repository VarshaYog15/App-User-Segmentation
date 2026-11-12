import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INPUT_PATH = "outputs/candidate_app_user_data.csv"
OUTPUT_PATH = "outputs/clustered_app_user_data.csv"
ELBOW_PATH = "visualizations/elbow_method.png"

MODEL_PATH = "models/kmeans_model.pkl"

def load_data():
    df = pd.read_csv(INPUT_PATH)

    print("Candidate dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # Check for missing values
    if df.isnull().sum().sum() > 0:
        raise ValueError("Missing values found in candidate dataset.")

    # Apply Standard Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Preserve feature names
    scaled_df = pd.DataFrame(
        scaled_data,
        columns=df.columns
    )

    # Save the fitted scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/standard_scaler.pkl")

    print("\nStandard Scaling completed successfully.")
    print("Scaled dataset shape:", scaled_df.shape)
    print("Scaler saved: models/standard_scaler.pkl")

    return scaled_df


def calculate_elbow(df):
    print("\n========== ELBOW METHOD ==========")

    inertia_values = []

    for k in range(2, 11):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        kmeans.fit(df)

        inertia_values.append(kmeans.inertia_)

        print(f"K = {k}, Inertia = {kmeans.inertia_:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), inertia_values, marker="o")

    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.xticks(range(2, 11))
    plt.grid(True)

    os.makedirs("visualizations", exist_ok=True)
    plt.savefig(ELBOW_PATH, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\nElbow graph saved:", ELBOW_PATH)


def calculate_silhouette_scores(df):
    print("\n========== SILHOUETTE SCORE ANALYSIS ==========")

    silhouette_values = []

    for k in range(2, 11):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(df)

        score = silhouette_score(
            df,
            labels,
            sample_size=10000,
            random_state=42
        )

        silhouette_values.append(score)

        print(f"K = {k}, Silhouette Score = {score:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 11), silhouette_values, marker="o")

    plt.title("Silhouette Score Analysis")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(2, 11))
    plt.grid(True)

    plt.savefig(
        "visualizations/silhouette_scores.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    print("\nSilhouette graph saved successfully.")


def train_kmeans(df, optimal_k):
    print("\n========== K-MEANS MODEL TRAINING ==========")

    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(df)

    df_clustered = df.copy()
    df_clustered["cluster"] = cluster_labels

    print("K-Means model trained successfully.")
    print("Number of clusters:", optimal_k)

    print("\nCluster distribution:")
    print(df_clustered["cluster"].value_counts().sort_index())

    return df_clustered, kmeans

def save_clustered_data(df_clustered):
    os.makedirs("outputs", exist_ok=True)

    df_clustered.to_csv(OUTPUT_PATH, index=False)

    print("\nClustered dataset saved:", OUTPUT_PATH)


def compare_cluster_solutions(scaled_df):
    print("\n========== K = 2 vs K = 4 COMPARISON ==========")

    original_df = pd.read_csv(
        "outputs/cleaned_app_user_data.csv"
    )

    profile_features = [
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
    "account_age_days",
    "engagement_score",
    "churn_risk_score"
]

    for k in [2, 4]:
        print(f"\n========== RESULTS FOR K = {k} ==========")

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_df)

        comparison_df = original_df.copy()
        comparison_df["cluster"] = labels

        print("\nCluster sizes:")
        print(
            comparison_df["cluster"]
            .value_counts()
            .sort_index()
        )

        print("\nCluster behavioural profiles:")
        print(
            comparison_df.groupby("cluster")[profile_features]
            .mean()
            .round(2)
        )


def main():
    print("\n========== APP USER SEGMENTATION ==========")

    # Load and scale candidate features
    scaled_df = load_data()

    # Analyse the optimal number of clusters
    calculate_elbow(scaled_df)
    calculate_silhouette_scores(scaled_df)
    compare_cluster_solutions(scaled_df)

    # Train final K-Means model
    clustered_scaled_df, kmeans = train_kmeans(
        scaled_df,
        optimal_k=4
    )

    # Save trained K-Means model
    joblib.dump(kmeans, MODEL_PATH)
    print("\nK-Means model saved:", MODEL_PATH)

    # Load original candidate features
    candidate_df = pd.read_csv(INPUT_PATH)

    # Load user IDs from cleaned dataset
    original_df = pd.read_csv(
        "outputs/cleaned_app_user_data.csv"
    )

    if len(candidate_df) != len(original_df):
        raise ValueError("Candidate data and original data row counts do not match.")

    # Attach cluster assignments to user IDs
    final_df = candidate_df.copy()
    final_df.insert(0, "user_id", original_df["user_id"].values)
    final_df["cluster"] = clustered_scaled_df["cluster"].values

    # Save final cluster assignments
    save_clustered_data(final_df)


if __name__ == "__main__":
    main()