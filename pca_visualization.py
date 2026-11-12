
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

DATA_PATH = "outputs/clustered_app_user_data.csv"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD CLUSTERED DATA
# --------------------------------------------------

def load_clustered_data():

    df = pd.read_csv(DATA_PATH)

    print("Clustered dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("\nAvailable columns:")
    print(df.columns.tolist())

    return df


# --------------------------------------------------
# PCA DIMENSIONALITY REDUCTION
# --------------------------------------------------

def perform_pca(df):

    # Select the six engineered features
    feature_columns = [
        "usage_intensity",
        "interaction_intensity",
        "content_engagement",
        "notifications_opened_per_week",
        "days_since_last_login",
        "app_friction_score"
    ]

    X = df[feature_columns]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce six dimensions into two principal components
    pca = PCA(n_components=2)

    pca_components = pca.fit_transform(X_scaled)

    # Create PCA dataframe
    pca_df = pd.DataFrame(
        pca_components,
        columns=["PC1", "PC2"]
    )

    # Preserve cluster labels
    pca_df["cluster"] = df["cluster"].values

    # Print explained variance
    print("\nPCA Transformation Completed!")
    print("Original number of features:", X.shape[1])
    print("Reduced number of components:", pca.n_components_)

    print("\nExplained Variance Ratio:")
    print(pca.explained_variance_ratio_)

    print(
        "\nTotal Variance Explained:",
        round(sum(pca.explained_variance_ratio_) * 100, 2),
        "%"
    )

    return pca_df, pca


# --------------------------------------------------
# PCA CLUSTER VISUALIZATION
# --------------------------------------------------

def visualize_clusters(pca_df):

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="viridis",
        alpha=0.5,
        s=25
    )

    plt.title(
        "App User Segmentation - PCA Visualization",
        fontsize=16
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.legend(title="User Cluster")

    plt.grid(alpha=0.2)
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "pca_cluster_visualization.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.show()

    print("\nPCA visualization saved successfully!")
    print("Saved at:", output_path)



if __name__ == "__main__":
    df = load_clustered_data()
    pca_df, pca = perform_pca(df)
    visualize_clusters(pca_df)