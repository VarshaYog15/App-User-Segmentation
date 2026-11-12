
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


BASELINE_PATH = "outputs/cleaned_app_user_data.csv"
CANDIDATE_PATH = "outputs/candidate_app_user_data.csv"

RESULT_PATH = "outputs/feature_comparison_results.csv"

BASELINE_FEATURES = [
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


def load_feature_sets():

    baseline_df = pd.read_csv(BASELINE_PATH)
    candidate_df = pd.read_csv(CANDIDATE_PATH)

    baseline = baseline_df[BASELINE_FEATURES].copy()

    print("Baseline dataset shape:", baseline.shape)
    print("Candidate dataset shape:", candidate_df.shape)

    return baseline, candidate_df


def evaluate_feature_set(df, feature_name):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    results = []

    print(f"\n========== {feature_name} ==========")

    for k in [2, 3, 4]:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        score = silhouette_score(
            scaled_data,
            labels,
            sample_size=10000,
            random_state=42
        )

        cluster_sizes = pd.Series(labels).value_counts().sort_index()

        print(f"\nK = {k}")
        print(f"Silhouette Score: {score:.4f}")
        print("Cluster Sizes:")
        print(cluster_sizes.to_dict())

        results.append({
            "Feature_Set": feature_name,
            "K": k,
            "Silhouette_Score": score,
            "Cluster_Sizes": str(cluster_sizes.to_dict())
        })

    return results


def main():

    baseline, candidate = load_feature_sets()

    results = []

    results.extend(
        evaluate_feature_set(baseline, "Original_14_Features")
    )

    results.extend(
        evaluate_feature_set(candidate, "Engineered_6_Features")
    )

    results_df = pd.DataFrame(results)

    os.makedirs("outputs", exist_ok=True)

    results_df.to_csv(RESULT_PATH, index=False)

    print("\n========== FINAL COMPARISON ==========")
    print(results_df.to_string(index=False))

    print("\nComparison results saved:", RESULT_PATH)


if __name__ == "__main__":
    main()
