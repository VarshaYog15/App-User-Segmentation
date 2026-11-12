
import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


DATA_PATH = "outputs/candidate_app_user_data.csv"
OUTPUT_PATH = "outputs/cluster_stability_results.csv"


def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


def evaluate_cluster_quality(df):

    scaler = joblib.load("models/standard_scaler.pkl")
    model = joblib.load("models/kmeans_model.pkl")

    scaled_data = scaler.transform(df)

    scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns
)

    labels = model.predict(scaled_df)   

    silhouette = silhouette_score(
        scaled_data,
        labels,
        sample_size=10000,
        random_state=42
    )

    db_index = davies_bouldin_score(
        scaled_data,
        labels
    )

    ch_score = calinski_harabasz_score(
        scaled_data,
        labels
    )

    print("\n========== CLUSTER QUALITY EVALUATION ==========")

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Calinski-Harabasz Score: {ch_score:.4f}")

def evaluate_cluster_stability(df):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    random_seeds = [21, 42, 100, 123, 2026]

    cluster_labels = {}

    print("\n========== CLUSTER STABILITY EVALUATION ==========")

    for seed in random_seeds:

        model = KMeans(
            n_clusters=4,
            random_state=seed,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        cluster_labels[seed] = labels

        print(f"\nRandom seed: {seed}")

        print(
            "Cluster sizes:",
            pd.Series(labels).value_counts().sort_index().to_dict()
        )

    # Compare cluster assignments between random seeds
    results = []

    seeds = list(cluster_labels.keys())

    for i in range(len(seeds)):

        for j in range(i + 1, len(seeds)):

            seed_1 = seeds[i]
            seed_2 = seeds[j]

            ari_score = adjusted_rand_score(
                cluster_labels[seed_1],
                cluster_labels[seed_2]
            )

            results.append({
                "Seed_1": seed_1,
                "Seed_2": seed_2,
                "Adjusted_Rand_Index": ari_score
            })

    results_df = pd.DataFrame(results)

    print("\n========== STABILITY COMPARISON ==========")
    print(results_df.round(4).to_string(index=False))

    print(
        "\nAverage stability score:",
        round(results_df["Adjusted_Rand_Index"].mean(), 4)
    )

    os.makedirs("outputs", exist_ok=True)

    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\nStability results saved:", OUTPUT_PATH)


def main():

    df = load_data()

    evaluate_cluster_stability(df)

    evaluate_cluster_quality(df)

    print("\nCluster stability evaluation completed.")


if __name__ == "__main__":
    main()
