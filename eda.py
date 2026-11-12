
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
DATA_PATH = "outputs/cleaned_app_user_data.csv"
OUTPUT_DIR = "visualizations"

# Visualization settings
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def load_data():
    """Load cleaned dataset."""

    df = pd.read_csv(DATA_PATH)
    print("Cleaned dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    return df


def plot_user_demographics(df):
    """Analyse user demographics."""

    # Age distribution
    plt.figure()
    sns.histplot(data=df, x="age", bins=20, kde=True)
    plt.title("Distribution of User Age")
    plt.xlabel("Age")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/age_distribution.png")
    plt.close()

    # Gender distribution
    plt.figure()
    sns.countplot(data=df, x="gender")
    plt.title("User Distribution by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gender_distribution.png")
    plt.close()


def plot_app_usage(df):
    """Analyse app usage patterns."""

    # Sessions per week
    plt.figure()
    sns.histplot(data=df, x="sessions_per_week", bins=20, kde=True)
    plt.title("Weekly App Session Frequency")
    plt.xlabel("Sessions per Week")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sessions_distribution.png")
    plt.close()

    # Session duration
    plt.figure()
    sns.histplot(data=df, x="avg_session_duration_min", bins=30, kde=True)
    plt.title("Average Session Duration")
    plt.xlabel("Duration (Minutes)")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/session_duration_distribution.png")
    plt.close()

    # Daily active minutes
    plt.figure()
    sns.histplot(data=df, x="daily_active_minutes", bins=30, kde=True)
    plt.title("Daily Active Minutes")
    plt.xlabel("Active Minutes")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/daily_active_minutes.png")
    plt.close()


def plot_engagement_analysis(df):
    """Analyse engagement and churn risk."""

    # Engagement score distribution
    plt.figure()
    sns.histplot(data=df, x="engagement_score", bins=30, kde=True)
    plt.title("User Engagement Score Distribution")
    plt.xlabel("Engagement Score")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/engagement_distribution.png")
    plt.close()

    # Churn risk distribution
    plt.figure()
    sns.histplot(data=df, x="churn_risk_score", bins=30, kde=True)
    plt.title("User Churn Risk Distribution")
    plt.xlabel("Churn Risk Score")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/churn_risk_distribution.png")
    plt.close()

    # Engagement vs churn risk
    
    # Engagement vs Churn Risk - Improved Visualization

    plt.figure(figsize=(12, 7))

    sns.kdeplot(
        data=df,
        x="engagement_score",
        y="churn_risk_score",
        fill=True,
        cmap="viridis",
        levels=15,
        thresh=0.05
    )

    sns.regplot(
        data=df.sample(min(3000, len(df)), random_state=42),
        x="engagement_score",
        y="churn_risk_score",
        scatter=False,
        lowess=False,
        color="red",
        line_kws={"linewidth": 2.5, "label": "Linear Trend"}
    )

    plt.title(
        "Relationship Between User Engagement and Churn Risk",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    plt.xlabel("Engagement Score", fontsize=12)
    plt.ylabel("Churn Risk Score", fontsize=12)

    plt.xlim(0, 100)
    plt.ylim(0, 1)

    plt.legend(
    handles=[
        plt.Line2D(
            [0], [0],
            color="red",
            linewidth=2.5,
            label="Linear Trend"
        )
    ]
)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/engagement_vs_churn.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



def plot_subscription_analysis(df):
    """Analyse subscription patterns."""

    plt.figure()
    sns.countplot(
        data=df,
        x="subscription_type",
        order=df["subscription_type"].value_counts().index
    )
    plt.title("Users by Subscription Type")
    plt.xlabel("Subscription Type")
    plt.ylabel("Number of Users")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/subscription_distribution.png")
    plt.close()


def plot_correlation_matrix(df):
    """Analyse correlations between numerical features."""

    numerical_df = df.select_dtypes(include="number").drop(
        columns=["user_id", "app_version"],
        errors="ignore"
    )

    correlation = numerical_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        annot=False
    )
    plt.title("Correlation Matrix of Numerical Features")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png")
    plt.close()


def main():

    print("\n========== EXPLORATORY DATA ANALYSIS ==========")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()

    plot_user_demographics(df)
    plot_app_usage(df)
    plot_engagement_analysis(df)
    plot_subscription_analysis(df)
    plot_correlation_matrix(df)

    print("\nEDA completed successfully.")
    print("Visualizations saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
