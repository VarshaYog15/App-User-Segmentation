
import os
import pandas as pd


from dotenv import load_dotenv
from sqlalchemy import create_engine, URL, text

def update_user_support_metrics():

    df = pd.read_csv("app_user_behavior_dataset.csv")

    update_data = df[
        [
            "user_id",
            "crash_events_last_30_days",
            "support_tickets_raised"
        ]
    ]

    update_query = text("""
        UPDATE users
        SET
            crash_events_last_30_days = :crashes,
            support_tickets_raised = :tickets
        WHERE user_id = :user_id
    """)

    records = [
        {
            "user_id": str(row.user_id),
            "crashes": int(row.crash_events_last_30_days),
            "tickets": int(row.support_tickets_raised)
        }
        for row in update_data.itertuples(index=False)
    ]

    with engine.begin() as connection:
        connection.execute(update_query, records)

    print("\nUser support metrics updated successfully.")
    print("Total records processed:", len(records))

# Load environment variables
load_dotenv(override=True)


# Read database configuration
db_url = URL.create(
    drivername="mysql+pymysql",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT", 3306)),
    database=os.getenv("DB_NAME")
)

# Create database connection
engine = create_engine(db_url)

def test_connection():
    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT DATABASE()"))
            database_name = result.scalar()

            print("MySQL connection successful!")
            print("Connected database:", database_name)

    except Exception as error:
        print("Database connection failed:", error)

def import_users():

    df = pd.read_csv("app_user_behavior_dataset.csv")

    columns = [
        "user_id",
        "age",
        "gender",
        "country",
        "device_type",
        "app_version",
        "subscription_type",
        "marketing_source",
        "engagement_score",
        "churn_risk_score"
    ]

    users_df = df[columns].copy()

    users_df.to_sql(
        name="users",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )

    print("\nUsers table imported successfully.")
    print("Total users imported:", len(users_df))


def import_user_clusters():

    df = pd.read_csv("outputs/clustered_app_user_data.csv")

    columns = [
        "user_id",
        "usage_intensity",
        "interaction_intensity",
        "content_engagement",
        "notifications_opened_per_week",
        "days_since_last_login",
        "app_friction_score",
        "cluster"
    ]

    cluster_df = df[columns].copy()

    cluster_df.to_sql(
        name="user_clusters",
        con=engine,
        if_exists="append",
        index=False,
        chunksize=1000
    )

    print("\nUser clusters imported successfully.")
    print("Total cluster records imported:", len(cluster_df))

def import_cluster_profiles():

    df = pd.read_csv(
        "outputs/business_cluster_profiles.csv"
    )

    df.to_sql(
        name="cluster_profiles",
        con=engine,
        if_exists="append",
        index=False
    )

    print("\nCluster profiles imported successfully.")
    print("Total profiles imported:", len(df))

def import_business_recommendations():

    df = pd.read_csv(
        "outputs/cluster_business_recommendations.csv"
    )

    df.to_sql(
        name="business_recommendations",
        con=engine,
        if_exists="append",
        index=False
    )

    print("\nBusiness recommendations imported successfully.")
    print("Total recommendations imported:", len(df))


def get_cluster_summary():

    query = text("""
        SELECT
            cluster,
            COUNT(*) AS total_users,
            ROUND(
                COUNT(*) * 100.0 /
                (SELECT COUNT(*) FROM user_clusters),
                2
            ) AS user_percentage,
            ROUND(AVG(usage_intensity), 2)
                AS avg_usage_intensity,
            ROUND(AVG(days_since_last_login), 2)
                AS avg_days_since_login,
            ROUND(AVG(app_friction_score), 2)
                AS avg_app_friction
        FROM user_clusters
        GROUP BY cluster
        ORDER BY cluster
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df

def get_cluster_profiles():

    query = text("""
        SELECT
            cluster,
            COUNT(*) AS total_users,
            ROUND(AVG(usage_intensity), 2)
                AS avg_usage_intensity,
            ROUND(AVG(interaction_intensity), 2)
                AS avg_interaction_intensity,
            ROUND(AVG(content_engagement), 2)
                AS avg_content_engagement,
            ROUND(AVG(notifications_opened_per_week), 2)
                AS avg_notifications,
            ROUND(AVG(days_since_last_login), 2)
                AS avg_days_since_login,
            ROUND(AVG(app_friction_score), 2)
                AS avg_app_friction
        FROM user_clusters
        GROUP BY cluster
        ORDER BY cluster
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df

def get_subscription_distribution():

    query = text("""
        SELECT
            uc.cluster,
            u.subscription_type,
            COUNT(*) AS total_users,
            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (
                    PARTITION BY uc.cluster
                ),
                2
            ) AS percentage
        FROM users u
        INNER JOIN user_clusters uc
            ON u.user_id = uc.user_id
        GROUP BY
            uc.cluster,
            u.subscription_type
        ORDER BY
            uc.cluster,
            u.subscription_type
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df


def get_device_distribution():

    query = text("""
        SELECT
            uc.cluster,
            u.device_type,
            COUNT(*) AS total_users,
            ROUND(
                COUNT(*) * 100.0 /
                SUM(COUNT(*)) OVER (
                    PARTITION BY uc.cluster
                ),
                2
            ) AS percentage
        FROM users u
        INNER JOIN user_clusters uc
            ON u.user_id = uc.user_id
        GROUP BY
            uc.cluster,
            u.device_type
        ORDER BY
            uc.cluster,
            u.device_type
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df

def get_high_friction_users(limit=20):
    query = text("""
        SELECT
            u.user_id,
            u.device_type,
            u.subscription_type,
            u.crash_events_last_30_days,
            u.support_tickets_raised,
            uc.app_friction_score,
            uc.cluster
        FROM users u
        INNER JOIN user_clusters uc
            ON u.user_id = uc.user_id
        WHERE uc.cluster = 3
        ORDER BY uc.app_friction_score DESC
        LIMIT :limit
    """)

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"limit": limit}
        )

    return df

def get_inactive_users(limit=20):
    query = text("""
        SELECT
            u.user_id,
            u.device_type,
            u.subscription_type,
            uc.days_since_last_login,
            uc.usage_intensity,
            uc.cluster
        FROM users u
        INNER JOIN user_clusters uc
            ON u.user_id = uc.user_id
        WHERE
            uc.cluster = 1
            AND uc.days_since_last_login > 30
        ORDER BY uc.days_since_last_login DESC
        LIMIT :limit
    """)

    with engine.connect() as connection:
        df = pd.read_sql(
            query,
            connection,
            params={"limit": limit}
        )

    return df

def get_business_recommendations():
    query = text("""
        SELECT
            cluster,
            segment_name,
            key_finding,
            business_action
        FROM business_recommendations
        ORDER BY cluster
    """)

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df

if __name__ == "__main__":

    test_connection()

    print("\n========== CLUSTER SUMMARY ==========")
    print(get_cluster_summary())

    print("\n========== DETAILED CLUSTER PROFILES ==========")
    print(get_cluster_profiles())

    print("\n========== SUBSCRIPTION DISTRIBUTION ==========")
    print(get_subscription_distribution())

    print("\n========== DEVICE DISTRIBUTION ==========")
    print(get_device_distribution())

    print("\n========== HIGH FRICTION USERS ==========")
    print(get_high_friction_users())

    print("\n========== INACTIVE USERS ==========")
    print(get_inactive_users())

    print("\n========== BUSINESS RECOMMENDATIONS ==========")
    print(get_business_recommendations().to_string(index=False))