# Customer-Segmentation-KMeans-MallCustomers

Customer segmentation on the Mall Customers dataset using K-Means clustering, PCA/t-SNE visualization, and segment-wise marketing strategy recommendations. Built as Task 2 of the CodeAlpha Data Analytics Internship.

## 1. Objective

Retail businesses often treat all customers as a single group when planning marketing campaigns, which is inefficient because high-value customers aren't specifically nurtured, and low-engagement customers aren't specifically targeted for re-engagement.

The objective of this project is to:
- Explore customer demographics and spending behavior through EDA.
- Apply **K-Means Clustering** (unsupervised learning) to segment customers based on their spending habits.
- Use **PCA** and **t-SNE** to visualize the resulting clusters in reduced dimensions.
- Translate each cluster into a customer **persona** and propose a **tailored marketing strategy** for each segment.

## 2. Project Structure

```
├── Customer_Segmentation.ipynb   # Full analysis notebook (EDA, clustering, PCA/t-SNE, results)
├── Mall_Customers.csv            # Dataset
├── Images/                       # Exported plots used in this README
│   ├── plot1_distribution_age_income_spending.png
│   ├── plot2_customercount_gender.png
│   ├── plot3_customercount_marital_education.png
│   ├── plot4_correlation.png
│   ├── plot5_pairwise.png
│   ├── plot6_annualincome_spendingscore.png
│   ├── plot7_cluster.png
│   ├── plot8_Silhouette.png
│   ├── plot9_customer_segmentation.png
│   ├── plot10_customer_segmentation_pca.png
│   ├── plot11_customer_segmentation_t-sne.png
│   ├── plot12_customer_segmentation_pca.png
│   └── plot13_final_customer_segmentation.png
└── README.md                     # Project overview (this file)
```

## 3. Dataset Overview

**Source:** Kaggle Dataset: Mall_Customers with 200 customer records.

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier for each customer |
| `Gender` | Male / Female |
| `Age` | Customer's age in years |
| `Education` | Highest education level attained |
| `Marital Status` | Married / Single / Divorced / Unknown |
| `Annual Income (k$)` | Annual income of the customer, in thousands of dollars |
| `Spending Score (1-100)` | Score assigned by the mall based on customer behavior and spending nature |

No missing values or duplicate rows were found. `Education` and `Marital Status` were cleaned and explored during EDA for demographic context but were **not** used as core clustering features, since the task's stated focus is "spending habits," best captured by `Annual Income` and `Spending Score`.

## 4. Technical Approach

1. **Data Cleaning & Preprocessing:** stripped stray whitespace from column names, checked for missing values/duplicates, inspected categorical columns for inconsistent labels, and encoded categoricals for EDA purposes.
2. **Exploratory Data Analysis (EDA):** distribution plots for Age, Income, and Spending Score; Gender/Education/Marital Status breakdowns; a correlation heatmap; a pairplot; and the key Income-vs-Spending-Score scatter plot.
3. **Feature Scaling:** standardized `Annual Income (k$)` and `Spending Score (1-100)` using `StandardScaler`, since K-Means is a distance-based algorithm.
4. **Choosing K:** used the **Elbow Method** (WCSS/inertia) and **Silhouette Score** across K = 2–10 to identify the optimal number of clusters.
5. **K-Means Clustering:** fit the final model at the optimal K on the scaled Income/Spending Score features.
6. **Dimensionality Reduction:** re-ran clustering on an extended feature set (Age + Income + Spending Score) and projected it to 2D using **PCA** and **t-SNE** to visually validate that the clusters are genuine, well-separated groupings.
7. **Cluster Profiling & Segment Labeling:** computed per-cluster averages (Age, Income, Spending Score, Gender mix) and programmatically assigned each cluster a business-readable persona label based on its income/spending profile.
8. **Marketing Strategy Mapping:** proposed a tailored marketing strategy for each identified segment.

## 5. Model Performance

**Model used:** K-Means Clustering (scikit-learn), with `k-means++` initialization and `n_init=10`.

| Metric | Value |
|---|---|
| Optimal K (Income + Spending Score) | **5** |
| Silhouette Score (Income + Spending Score) | **0.5547** |
| Inertia / WCSS (final model) | 65.57 |
| Silhouette Score (Age + Income + Spending Score, extended model) | 0.4166 |

The Elbow Method and Silhouette Score across K = 2–10 both pointed to **K = 5** as the optimal number of clusters the elbow visibly bends at 5, and the silhouette score peaks at 5 before declining for higher K. The extended 3-feature model (adding Age) yields a lower silhouette score, confirming that Income and Spending Score alone already capture most of the meaningful segmentation signal.

## 6. Images / Visualizations

| | |
|---|---|
| **Distribution of Age, Income & Spending Score** | ![Distribution](Images/plot1_distribution_age_income_spending.png) |
| **Customer Count by Gender** | ![Gender Count](Images/plot2_customercount_gender.png) |
| **Customer Count by Marital Status & Education** | ![Marital/Education Count](Images/plot3_customercount_marital_education.png) |
| **Correlation Heatmap** | ![Correlation](Images/plot4_correlation.png) |
| **Pairwise Relationships** | ![Pairwise](Images/plot5_pairwise.png) |
| **Annual Income vs Spending Score** | ![Income vs Spending](Images/plot6_annualincome_spendingscore.png) |
| **Elbow Method & Silhouette Score (Optimal K)** | ![Elbow/Silhouette](Images/plot7_cluster.png) |
| **Per-Cluster Silhouette Diagnostic** | ![Silhouette Diagnostic](Images/plot8_Silhouette.png) |
| **K-Means Clusters (Income vs Spending Score)** | ![Clusters](Images/plot9_customer_segmentation.png) |
| **PCA Projection of Clusters** | ![PCA](Images/plot10_customer_segmentation_pca.png) |
| **t-SNE Projection of Clusters** | ![t-SNE](Images/plot11_customer_segmentation_t-sne.png) |
| **Cluster Profile Comparison (Avg Income/Spending/Age)** | ![Cluster Profile](Images/plot12_customer_segmentation_pca.png) |
| **Final Labeled Customer Segments** | ![Final Segments](Images/plot13_final_customer_segmentation.png) |

## 7. Key Findings and Key Recommendations

**Key Findings:**
- Five natural customer segments emerged clearly from K-Means clustering on Annual Income and Spending Score, confirmed by both the elbow method and silhouette analysis (silhouette score ≈ 0.55).
- The clusters map cleanly onto intuitive retail personas: **Premium/VIP Shoppers**, **Cautious Affluents**, **Impulsive/Aspirational Shoppers**, **Budget-Conscious/At-Risk customers**, and **Standard/Mainstream Shoppers**.
- PCA and t-SNE visualizations both confirm the 5 clusters are genuine, well-separated groupings rather than an artifact of the chosen 2D projection.

| Segment | Profile | Recommended Marketing Strategy |
|---|---|---|
| **Premium / VIP Shoppers** | High income, high spending | VIP loyalty programs, early access to new collections, personalized concierge service, premium cross-sell offers |
| **Cautious Affluents** | High income, low spending | Investigate low engagement; targeted high-value promotions and personalized recommendations to unlock latent purchasing power |
| **Impulsive / Aspirational Shoppers** | Low income, high spending | Installment plans, loyalty points, budget-friendly bundles with value + status messaging |
| **Budget-Conscious / At-Risk** | Low income, low spending | Discounts, clearance sales, value-for-money messaging via low-cost channels (email/SMS) |
| **Standard / Mainstream Shoppers** | Average income, average spending | Broad seasonal campaigns and cross-category promotions to nudge toward higher-spending segments |

**Key Recommendation:** The mall should move away from one-size-fits-all campaigns toward segment-specific targeting. The **Cautious Affluents** segment represents the largest untapped revenue opportunity (income exists, but isn't converting to spend), while **Premium/VIP** and **Impulsive/Aspirational** segments should be prioritized for retention. The **Standard/Mainstream** segment, being the largest by count, is the best target for broad promotions aimed at gradually shifting more customers toward higher-spending segments.

## 8. How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/yashalaf/Customer-Segmentation-Using-Unsupervised-Learning.git
   cd Customer-Segmentation
   ```
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Customer_Segmentation.ipynb
   ```
4. Run all cells in order (`Kernel > Restart & Run All`). Ensure `Mall_Customers.csv` is in the same directory as the notebook.

## 9. Skills Demonstrated

- Exploratory Data Analysis (EDA) and data cleaning/preprocessing
- Unsupervised Learning K-Means Clustering
- Model selection using the Elbow Method and Silhouette Score
- Dimensionality reduction and visualization (PCA, t-SNE)
- Data visualization (Matplotlib, Seaborn)
- Customer segmentation and persona development
- Translating data-driven insights into business/marketing strategy

## Limitations & Next Steps

- The dataset is limited to 200 customers from a single mall; results should be validated on a larger, multi-location dataset.
- Additional behavioral data (purchase frequency, basket size, product categories) would allow richer segmentation beyond the income/spending-score proxy.
- Proposed marketing strategies should be A/B tested before full rollout to measure real-world impact.
