
import pandas as pd
import joblib



DATA_PATH = "outputs/candidate_app_user_data.csv"



def main():

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    # Load six engineered features
    df = pd.read_csv("outputs/candidate_app_user_data.csv")

    # Load original dataset for business profiling
    original_df = pd.read_csv("app_user_behavior_dataset.csv")


   # Load saved scaler and K-Means model
    scaler = joblib.load("models/standard_scaler.pkl")
    model = joblib.load("models/kmeans_model.pkl")

# Transform features using the saved scaler
    scaled_data = scaler.transform(df)

# Convert scaled data into DataFrame
    scaled_df = pd.DataFrame(
    scaled_data,
    columns=df.columns,
    index=df.index
)

    # Predict cluster assignments
    df["cluster"] = model.predict(scaled_df)    

    # Attach cluster labels to original dataset
    original_df["cluster"] = df["cluster"]

    print("\n========== CLUSTER SIZES ==========")
    print(original_df["cluster"].value_counts().sort_index())

    # Numerical business profile
    numerical_features = [
        "engagement_score",
        "churn_risk_score",
        "age",
        "sessions_per_week",
        "daily_active_minutes",
        "days_since_last_login",
        "crash_events_last_30_days",
        "support_tickets_raised"
    ]

    print("\n========== BUSINESS METRICS BY CLUSTER ==========")

    business_profile = (
        original_df.groupby("cluster")[numerical_features]
        .mean()
        .round(2)
    )

    print(business_profile)

    # Categorical business profile
    categorical_features = [
        "subscription_type",
        "device_type"
    ]

    for feature in categorical_features:

        print(f"\n========== {feature.upper()} DISTRIBUTION ==========")

        distribution = pd.crosstab(
            original_df["cluster"],
            original_df[feature],
            normalize="index"
        ) * 100

        print(distribution.round(2))

    # Save results
    business_profile.to_csv(
        "outputs/business_cluster_profiles.csv"
    )
        # Business recommendations for each cluster

    recommendations = {
        0: {
            "segment_name": "Recently Active Users",
            "key_finding": "Users logged in more recently, averaging 10.57 days since last login.",
            "business_action": "Maintain engagement through relevant content and feature updates."
        },
        1: {
            "segment_name": "Less Recently Active Users",
            "key_finding": "Users have an average of 33.43 days since last login.",
            "business_action": "Test re-engagement notifications and targeted campaigns."
        },
        2: {
            "segment_name": "High Usage Frequency Users",
            "key_finding": "Users average 10.27 sessions per week.",
            "business_action": "Collect feedback and improve the frequent-use experience."
        },
        3: {
            "segment_name": "High App Friction Users",
            "key_finding": "Users average 1.49 crashes and 0.74 support tickets.",
            "business_action": "Prioritise crash investigation, bug fixes and support improvements."
        }
    }

    recommendation_df = pd.DataFrame.from_dict(
        recommendations,
        orient="index"
    )

    recommendation_df.index.name = "cluster"

    recommendation_df.to_csv(
        "outputs/cluster_business_recommendations.csv"
    )

    print("\n========== BUSINESS RECOMMENDATIONS ==========")
    print(recommendation_df.to_string())

    print("\nBusiness recommendations saved successfully.")
    print("\nBusiness cluster profiles saved successfully.")


if __name__ == "__main__":
    main()





