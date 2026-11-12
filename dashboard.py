
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Import SQL functions
from sql import (
    get_cluster_summary,
    get_cluster_profiles,
    get_subscription_distribution,
    get_device_distribution,
    get_high_friction_users,
    get_inactive_users,
    get_business_recommendations
)


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="App User Behavior Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. DASHBOARD HEADER
# ==================================================

st.title("📊 App User Behavior Segmentation Dashboard")

st.markdown(
    """
    ### User Segmentation Using Unsupervised Machine Learning

    Analyse application user behaviour using K-Means Clustering
    and explore actionable business insights.
    """
)

st.divider()


# ==================================================
# 3. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "Overview",
        "Cluster Analysis",
        "User Insights",
        "Business Recommendations"
    ]
)


# ==================================================
# 4. LOAD DATA FROM MYSQL
# ==================================================

@st.cache_data(ttl=300)
def load_dashboard_data():

    cluster_summary = get_cluster_summary()

    cluster_profiles = get_cluster_profiles()

    subscription_data = get_subscription_distribution()

    device_data = get_device_distribution()

    friction_users = get_high_friction_users()

    inactive_users = get_inactive_users()

    recommendations = get_business_recommendations()

    return (
        cluster_summary,
        cluster_profiles,
        subscription_data,
        device_data,
        friction_users,
        inactive_users,
        recommendations
    )


try:

    (
        cluster_summary,
        cluster_profiles,
        subscription_data,
        device_data,
        friction_users,
        inactive_users,
        recommendations
    ) = load_dashboard_data()

    st.sidebar.success("MySQL Connected")

except Exception as e:

    st.error(f"Unable to load dashboard data: {e}")

    st.stop()


# ==================================================
# 5. OVERVIEW
# ==================================================

if page == "Overview":

    st.header("Application Overview")

    total_users = int(
        cluster_summary["total_users"].sum()
    )

    total_clusters = cluster_summary["cluster"].nunique()

    largest_cluster = cluster_summary.loc[
        cluster_summary["total_users"].idxmax()
    ]

    avg_friction = (
        cluster_summary["avg_app_friction"]
        * cluster_summary["total_users"]
    ).sum() / total_users

    # KPI Cards

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Users",
        f"{total_users:,}"
    )

    col2.metric(
        "Total Clusters",
        total_clusters
    )

    col3.metric(
        "Largest Segment",
        f"Cluster {int(largest_cluster['cluster'])}",
        f"{largest_cluster['user_percentage']}% of users"
    )

    col4.metric(
        "Average App Friction",
        f"{avg_friction:.2f}"
    )

    st.divider()

    # Cluster Summary Table

    st.subheader("Cluster Summary")

    display_summary = cluster_summary.rename(
        columns={
            "cluster": "Cluster",
            "total_users": "Total Users",
            "user_percentage": "User Percentage (%)",
            "avg_usage_intensity": "Avg Usage Intensity",
            "avg_days_since_login": "Avg Days Since Login",
            "avg_app_friction": "Avg App Friction"
        }
    )

    st.dataframe(
        display_summary,
        width="stretch",
        hide_index=True
    )


# ==================================================
# 6. CLUSTER ANALYSIS
# ==================================================

elif page == "Cluster Analysis":

    st.header("Cluster Analysis")

    # --------------------------------------------------
    # Cluster Distribution
    # --------------------------------------------------

    st.subheader("User Distribution Across Clusters")

    fig = px.bar(
        cluster_summary,
        x="cluster",
        y="total_users",
        color="cluster",
        text="total_users",
        title="Number of Users in Each Cluster",
        labels={
            "cluster": "User Cluster",
            "total_users": "Number of Users"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------
    # Behaviour Comparison
    # --------------------------------------------------

    st.subheader("Cluster Behaviour Comparison")

    metric = st.selectbox(
        "Select Behaviour Metric",
        [
            "avg_usage_intensity",
            "avg_days_since_login",
            "avg_app_friction"
        ],
        format_func=lambda x: {
            "avg_usage_intensity": "Usage Intensity",
            "avg_days_since_login": "Days Since Last Login",
            "avg_app_friction": "App Friction Score"
        }[x]
    )

    metric_fig = px.bar(
        cluster_summary,
        x="cluster",
        y=metric,
        color="cluster",
        text_auto=".2f",
        title="Behaviour Comparison Across Clusters",
        labels={
            "cluster": "User Cluster",
            metric: metric.replace("_", " ").title()
        }
    )

    st.plotly_chart(
        metric_fig,
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------
    # PCA Visualization
    # --------------------------------------------------

    st.subheader("PCA Cluster Visualization")

    pca_path = os.path.join(
        "visualizations",
        "pca_cluster_visualization.png"
    )

    if os.path.exists(pca_path):

        st.image(
            pca_path,
            caption="PCA Visualization of Four User Clusters",
            width="stretch"
        )

        st.caption(
            "PCA reduces the six engineered features into two "
            "principal components for visualisation. "
            "The two components explain 33.65% of total variance."
        )

    else:

        st.warning(
            "PCA visualization not found. "
            "Run pca_visualization.py first."
        )


# ==================================================
# 7. USER INSIGHTS
# ==================================================

elif page == "User Insights":

    st.header("User Behaviour Insights")

    # --------------------------------------------------
    # Subscription Distribution
    # --------------------------------------------------

    st.subheader("Subscription Distribution Across Clusters")

    subscription_fig = px.bar(
        subscription_data,
        x="cluster",
        y="percentage",
        color="subscription_type",
        barmode="stack",
        text_auto=".1f",
        title="Subscription Type Distribution",
        labels={
            "cluster": "User Cluster",
            "percentage": "Percentage (%)",
            "subscription_type": "Subscription Type"
        }
    )

    st.plotly_chart(
        subscription_fig,
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------
    # Device Distribution
    # --------------------------------------------------

    st.subheader("Device Distribution Across Clusters")

    device_fig = px.bar(
        device_data,
        x="cluster",
        y="percentage",
        color="device_type",
        barmode="stack",
        text_auto=".1f",
        title="Device Type Distribution",
        labels={
            "cluster": "User Cluster",
            "percentage": "Percentage (%)",
            "device_type": "Device Type"
        }
    )

    st.plotly_chart(
        device_fig,
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------
    # High Friction Users
    # --------------------------------------------------

    st.subheader("High App Friction Users")

    st.write(
        "Users belonging to Cluster 3 with the highest "
        "app friction scores."
    )

    st.dataframe(
        friction_users,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------
    # Inactive Users
    # --------------------------------------------------

    st.subheader("Inactive Users")

    st.write(
        "Users in Cluster 1 who have been inactive "
        "for more than 30 days."
    )

    st.dataframe(
        inactive_users,
        width="stretch",
        hide_index=True
    )


# ==================================================
# 8. BUSINESS RECOMMENDATIONS
# ==================================================

elif page == "Business Recommendations":

    st.header("Business Recommendations")

    st.write(
        "Actionable strategies based on the four "
        "identified user segments."
    )

    # Recommendation Cards

    for _, row in recommendations.iterrows():

        st.subheader(
            f"Cluster {row['cluster']} - {row['segment_name']}"
        )

        st.markdown(
            f"**Key Finding:** {row['key_finding']}"
        )

        st.info(
            f"**Recommended Action:** {row['business_action']}"
        )

        st.divider()

    # Recommendations Table

    st.subheader("Complete Business Recommendations")

    st.dataframe(
        recommendations,
        width="stretch",
        hide_index=True
    )


# ==================================================
# 9. FOOTER
# ==================================================

st.divider()

st.caption(
    "App User Behavior Segmentation | "
    "K-Means Clustering | Python | MySQL | Streamlit"
)
