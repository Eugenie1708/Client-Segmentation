# 📊 ShopNow Customer Segmentation (KMeans Clustering)

## 🔍 Project Overview

This project performs **customer segmentation** for ShopNow, a fast-growing e-commerce company, using **KMeans clustering**.

The goal is to group customers based on behavioral similarity to support:

* Targeted marketing campaigns
* Customer retention strategies
* Premium upsell opportunities
* Reactivation campaigns

This is an **unsupervised learning** task — no target variable is used.

---
## 📂 Repository Structure

- `notebooks/segmentation.ipynb` – Main analysis notebook  
- `notebooks/segmentation.html` – Exported notebook (includes code + explanations)
- `notebooks/LearningNotes.md` – Exported learning notes  
- `reports/Customer Segmentation Report.docx` – Final written report  
- `figures/` – Saved plots (e.g., elbow curve)  
- `src/segmentation.py` – Script-based implementation  
---

## 📁 Dataset Summary

| Metric                 | Value                    |
| ---------------------- | ------------------------ |
| Customers              | 6,000                    |
| Total Variables        | 19                       |
| Segmentation Variables | 9 (RFM + Category Share) |
| Final Clusters         | 4                        |

### Segmentation Inputs

**Behavioral (RFM) Variables**

* `recency_days`
* `orders_last_12m`
* `avg_order_value`

**Category Mix Variables**

* `cat_share_electronics`
* `cat_share_apparel`
* `cat_share_home`
* `cat_share_beauty`
* `cat_share_groceries`
* `cat_share_sports`

---

## 🧠 Methodology

### 1️⃣ Data Preparation

* Selected RFM + category share variables
* Replaced ±infinity values
* Filled missing values with median
* Scaled **RFM variables only** using `StandardScaler`

  * (KMeans is distance-based; prevents unit dominance)

---

### 2️⃣ KMeans Clustering

* Evaluated k = 2 to 10 clusters
* Used **Elbow Method (WCSS / inertia)**
* Selected **k = 4**
* Ensured reproducibility with:

  * `random_state=42`
  * `n_init=10`

---

## 📈 Final Cluster Results

### Cluster Distribution

| Cluster | Customers | %     |
| ------- | --------- | ----- |
| 0       | 1,536     | 25.6% |
| 1       | 1,902     | 31.7% |
| 2       | 1,350     | 22.5% |
| 3       | 1,212     | 20.2% |

---

### Cluster Profiles (Mean Values)

| Cluster | Recency  | Orders | Avg Order ($) | Interpretation       |
| ------- | -------- | ------ | ------------- | -------------------- |
| 0       | 549 days | 1.0    | 29.9          | Inactive / Churned   |
| 1       | 93 days  | 5.6    | 71.3          | Mid-tier Regulars    |
| 2       | 79 days  | 4.0    | 205.4         | Big Spenders         |
| 3       | 57 days  | 11.6   | 45.7          | High Frequency Loyal |

---

## 🎯 Business Insights

### 🏆 High Frequency Loyal (Cluster 3)

* Most engaged repeat customers
* Strong retention candidates

**Strategy:** Loyalty rewards, referrals, early access

---

### 💎 Big Spenders (Cluster 2)

* Highest order value
* Premium-value customers

**Strategy:** VIP perks, premium bundles, personalized upsells

---

### ⚖️ Mid-tier Regulars (Cluster 1)

* Moderate engagement
* Growth potential

**Strategy:** Frequency boosters, subscriptions

---

### ⚠️ Inactive / Churned (Cluster 0)

* Very high recency
* Minimal recent engagement

**Strategy:** Win-back campaigns, targeted discounts

---

## 🔎 Key Analytical Finding

Segmentation was driven primarily by **RFM behavior**, while category mix was relatively similar across clusters.

This suggests that **engagement and spending intensity** matter more than product preference in defining customer segments for ShopNow.

---

## 🛠 Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* matplotlib
* Jupyter Notebook

---

## 📄 Deliverables

* `shopnow_segmentation.ipynb`
* Exported HTML notebook
* Word report
* Cleaned, reproducible workflow

---

## 🚀 Author

Eugenie Lai
MSBA — UIUC
Focus: Business Analytics, Customer Intelligence, Data-Driven Strategy

---