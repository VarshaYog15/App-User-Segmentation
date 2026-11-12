# App User Segmentation System

## 1. Project Overview

The **App User Segmentation System** is a data science project that applies unsupervised machine learning to group application users based on their usage behaviour, engagement patterns, and application friction.

The project uses **K-Means clustering** to identify behavioural user segments from a dataset containing 50,000 users.

The analysis is integrated with a MySQL database and an interactive Streamlit dashboard to present cluster profiles, user insights, and business recommendations.

## 2. Business Problem

Application users have different usage patterns and levels of engagement. Some users are recently active, some have reduced recent activity, while others experience application issues such as crashes and support problems.

This project aims to answer the following questions:

- How can users be grouped based on their application behaviour?
- Which users demonstrate higher usage frequency?
- Which users have not logged in recently?
- Which users experience higher application friction?
- What business actions can improve user engagement and application experience?

## 3. Project Objectives

- Understand the application user dataset.
- Perform data cleaning and exploratory data analysis.
- Engineer meaningful behavioural features.
- Apply feature scaling using StandardScaler.
- Determine a suitable number of clusters using the Elbow Method and Silhouette Score.
- Perform K-Means clustering.
- Evaluate cluster quality and stability.
- Visualize cluster patterns using PCA.
- Generate cluster profiles and business recommendations.
- Store analytical results in MySQL.
- Develop an interactive Streamlit dashboard.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Clustering, scaling, PCA and evaluation |
| Streamlit | Interactive dashboard |
| MySQL | Database management |
| SQLAlchemy | Database connectivity |
| PyMySQL | MySQL integration |
| Joblib | Model and scaler persistence |
| Python-dotenv | Environment configuration |

## 5. Dataset Description

The project uses an application user behaviour dataset containing:

- **50,000 user records**
- **25 original columns**

The dataset includes demographic information, device details, application usage, engagement, subscription information, and user experience indicators.

### Features Used for Clustering

Six behavioural features were selected or engineered:

1. `usage_intensity`
2. `interaction_intensity`
3. `content_engagement`
4. `notifications_opened_per_week`
5. `days_since_last_login`
6. `app_friction_score`

### Additional Engineered Features

- `engagement_score`
- `churn_risk_score`

These additional indicators support behavioural analysis and interpretation. The churn risk score is not a separately trained or validated churn prediction model.

## 6. Project Workflow

### Step 1: Data Understanding

Inspected the dataset structure, dimensions, column names, data types, descriptive statistics, missing values, and duplicate records.

Script: `data_understanding.py`

### Step 2: Data Cleaning

Performed duplicate checks, missing-value analysis, and data validation.

Script: `data_cleaning.py`

### Step 3: Exploratory Data Analysis

Explored user demographics, application usage, engagement, subscription patterns, and behavioural relationships using visualizations.

Script: `eda.py`

### Step 4: Feature Engineering

Created behavioural features to represent usage intensity, interaction intensity, content engagement, and application friction.

Script: `feature_engineering.py`

### Step 5: Data Preprocessing

Validated the selected features, analysed outliers, and applied StandardScaler.

Script: `preprocessing.py`

### Step 6: Feature Comparison

Compared baseline and engineered feature sets using K-Means clustering and silhouette scores.

Script: `feature_comparison.py`

### Step 7: K-Means Clustering

Applied the Elbow Method and Silhouette Score to compare cluster counts from K=2 to K=10. Selected four clusters for the final segmentation.

Script: `clustering.py`

### Step 8: Model Evaluation

Evaluated clustering quality using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index (ARI) for cluster stability

Script: `model_evaluation.py`

### Step 9: Cluster Profiling

Analysed cluster sizes, behavioural metrics, subscription distribution, device distribution, and business recommendations.

Script: `cluster_profiling.py`

### Step 10: PCA Visualization

Reduced the six-dimensional feature space to two principal components for visualization.

Script: `pca_visualization.py`

### Step 11: Database Integration

Stored and retrieved user-level clustering results, cluster profiles, and business recommendations using MySQL.

Script: `sql.py`

### Step 12: Interactive Dashboard

Presented segmentation results and business insights through an interactive Streamlit dashboard.

Script: `dashboard.py`

## 7. Model Selection and Evaluation

### Selected Algorithm

**K-Means Clustering**

K-Means is an unsupervised learning algorithm that groups data points based on similarities in their feature values.

### Cluster Selection

The Elbow Method and Silhouette Score were evaluated for K values from 2 to 10.

The highest observed silhouette score was obtained at K=4:

**Silhouette Score: 0.1468**

Four clusters were selected for the final segmentation based on the silhouette comparison and the interpretability of the resulting behavioural profiles.

### Evaluation Results

| Evaluation Metric | Result |
|---|---:|
| Number of clusters | 4 |
| Silhouette Score | 0.1468 |
| Davies-Bouldin Index | 1.8978 |
| Calinski-Harabasz Index | 6808.8417 |
| Average Pairwise ARI Stability | 0.9912 |
| PCA Variance Explained | 33.65% |

### Interpretation

The silhouette score indicates relatively weak cluster separation. The ARI stability result indicates that the clustering assignments were highly consistent across the five random-seed experiments performed.

The PCA visualization explains 33.65% of the total variance using two components. It is useful for visual inspection but does not demonstrate complete separation between clusters.

## 8. User Segments Identified

The K-Means model identified four behavioural segments.

| Cluster | User Segment | Total Users | Percentage |
|---|---|---:|---:|
| 0 | Recently Active Users | 18,857 | 37.71% |
| 1 | Less Recently Active Users | 18,940 | 37.88% |
| 2 | High Usage Frequency Users | 6,391 | 12.78% |
| 3 | High App Friction Users | 5,812 | 11.62% |

### Cluster 0 – Recently Active Users

- Average days since last login: 10.57
- Users demonstrate comparatively recent application activity.
- Business action: Maintain engagement through relevant content and feature updates.

### Cluster 1 – Less Recently Active Users

- Average days since last login: 33.43
- Users demonstrate comparatively lower recent activity.
- Business action: Test re-engagement notifications and targeted campaigns.

### Cluster 2 – High Usage Frequency Users

- Average sessions per week: 10.27
- Average usage intensity: 271.23
- Business action: Collect feedback and improve the frequent-use experience.

### Cluster 3 – High App Friction Users

- Average app friction score: 2.23
- Average crashes: 1.49
- Average support tickets: 0.74
- Business action: Prioritise crash investigation, bug fixes, and support improvements.

## 9. Database Integration

MySQL is used to store and retrieve analytical outputs.

**Database name:** `app_user_segmentation`

The project uses the following tables:

- `users` – User records and cluster assignments.
- `user_clusters` – Cluster information.
- `cluster_profiles` – Cluster-level behavioural summaries.
- `business_recommendations` – Recommendations for different user segments.

Database credentials are configured through environment variables using a local `.env` file. Sensitive credentials should never be committed to GitHub.

## 10. Interactive Streamlit Dashboard

The dashboard provides four main sections.

### 1. Overview

- Total users
- Number of clusters
- Largest user segment
- Average app friction
- Overall segmentation summary

### 2. Cluster Analysis

- User distribution across clusters
- Behavioural metric comparison
- PCA-based cluster visualization

### 3. User Insights

- Subscription distribution
- Device distribution
- High-friction user analysis
- Inactive user analysis

### 4. Business Recommendations

- Cluster-specific findings
- Suggested business actions
- Recommendations to improve engagement and application experience

## 11. Project Structure

```text
App User Segmentation System/
│
├── app_user_behavior_dataset.csv
│
├── data_understanding.py
├── data_cleaning.py
├── eda.py
├── feature_engineering.py
├── feature_comparison.py
├── preprocessing.py
├── clustering.py
├── model_evaluation.py
├── cluster_profiling.py
├── pca_visualization.py
├── sql.py
├── dashboard.py
│
├── models/
│   ├── kmeans_model.pkl
│   └── standard_scaler.pkl
│
├── outputs/
│   ├── business_cluster_profiles.csv
│   ├── cluster_business_recommendations.csv
│   ├── cluster_feature_profiles.csv
│   └── cluster_stability_results.csv
│
├── visualizations/
│   ├── elbow_method.png
│   ├── silhouette_scores.png
│   ├── pca_cluster_visualization.png
│   └── other EDA visualizations
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Generated intermediate files may also be present in the local `outputs/` folder.

## 12. Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/VarshaYog15/App-User-Segmentation.git
```

### 2. Navigate to the Project Folder

```bash
cd App-User-Segmentation
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

### 5. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Configure MySQL

- Install and start MySQL.
- Create the `app_user_segmentation` database.
- Configure your local `.env` file using the variable names documented in `.env.example`.
- Use your own database credentials.
- Never commit the actual `.env` file.

### 7. Execute the Project Scripts

Run the scripts in the following order:

```bash
python data_understanding.py
python data_cleaning.py
python eda.py
python feature_engineering.py
python preprocessing.py
python feature_comparison.py
python clustering.py
python model_evaluation.py
python cluster_profiling.py
python pca_visualization.py
```

After generating the clustering outputs and configuring MySQL, run:

```bash
python sql.py
```

### 8. Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser.

**Note:** The execution instructions assume the project data, generated outputs, and model artifacts are available at the expected paths. Follow the script output if a required intermediate file needs to be generated first.

## 13. Business Recommendations

The segmentation results can support the following business actions:

- **Recently Active Users:** Maintain engagement through relevant content and regular feature updates.
- **Less Recently Active Users:** Consider targeted re-engagement campaigns and personalized notifications.
- **High Usage Frequency Users:** Encourage continued engagement through useful features and loyalty initiatives.
- **High App Friction Users:** Investigate crashes, support tickets, and usability issues to improve the application experience.

These are suggested actions based on observed cluster profiles, not experimentally proven interventions.

## 14. Limitations

- The silhouette score indicates that the four clusters are not strongly separated.
- PCA represents only two dimensions and explains 33.65% of the total variance.
- Cluster labels are descriptive interpretations of behavioural profiles.
- The churn risk score is not a separately validated churn prediction model.
- Business recommendations require further validation through experiments and business outcomes.

## 15. Key Learnings

- Data Understanding and Exploratory Data Analysis
- Data Cleaning and Preprocessing
- Feature Engineering and Feature Selection
- StandardScaler
- Unsupervised Machine Learning
- K-Means Clustering
- Elbow Method
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- PCA Dimensionality Reduction
- Clustering Stability Evaluation
- SQL and MySQL Integration
- Streamlit Dashboard Development
- Business Insight Generation

## 16. Conclusion

The App User Segmentation System demonstrates how unsupervised machine learning can be applied to application user behaviour data to identify behavioural segments.

By combining data analysis, feature engineering, K-Means clustering, MySQL integration, and an interactive Streamlit dashboard, the project converts user behaviour data into insights that can support data-driven business decisions.

---

**Project Domain:** Data Science | Machine Learning | Customer Analytics

**Algorithm:** K-Means Clustering

**Dashboard:** Streamlit

**Database:** MySQL
