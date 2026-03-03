# 📘 Learning Notes — Questions I Encountered During This Project

This document summarizes the questions and conceptual challenges I encountered while building the ShopNow customer segmentation model.

The goal is to record not only what I did, but what I learned and clarified along the way.

---

# 1️⃣ Understanding the Framework

## ❓ What is RFM? Is it just a project abbreviation?

I learned that RFM is a real marketing framework:

* **Recency** — How recently a customer purchased
* **Frequency** — How often they purchase
* **Monetary** — How much they spend

It is widely used in marketing analytics to measure customer engagement and value.

Key realization:
Recency is often the strongest signal of future behavior.

---

# 2️⃣ Data Preparation Questions

## ❓ Why do we replace `np.inf` and `-np.inf`?

These are not missing values — they represent positive and negative infinity.

They can appear from division by zero or numerical overflow.
They must be converted to `NaN` because scaling and clustering cannot handle infinite values.

---

## ❓ Why do we use `StandardScaler`?

KMeans uses Euclidean distance.

If features are not scaled:

* Large-number variables (e.g., recency in days) dominate.
* Small-scale variables (e.g., category share 0–1) contribute less.

StandardScaler converts values into z-scores:

$$
z = \frac{x - \mu}{\sigma}
$$

This ensures each feature contributes proportionally to clustering.

Important insight:
Category shares were already between 0 and 1, so they were not scaled to preserve interpretability.

---

# 3️⃣ Model Mechanics

## ❓ What is the difference between `fit()` and `predict()`?

* `fit()` learns cluster centers.
* `predict()` assigns new observations to the learned clusters.
* `fit_predict()` does both.

Key insight:
For new incoming customers, we use `predict()` to assign them to existing clusters.

---

## ❓ What is inertia (WCSS)?

Inertia measures how tightly grouped data points are within clusters.

As k increases:

* Inertia always decreases.
* The elbow method helps determine where improvement slows down.

I selected k=4 based on:

* The bend in the curve
* Business interpretability

---

# 4️⃣ Analytical Observations

After clustering:

* Cluster 3 = High Frequency Loyal
* Cluster 2 = Big Spenders
* Cluster 1 = Mid-tier Regulars
* Cluster 0 = Inactive / Churned

Unexpected insight:
Category shares were similar across clusters.

This means segmentation was primarily driven by behavior (RFM), not product preference.

---

# 5️⃣ Technical Workflow Lessons

## ❓ Why did exporting to HTML take so long?

Even collapsed notebook outputs still exist internally.
They must be cleared to speed up export.

Lesson learned:
Clear outputs before exporting.

---

## ❓ Why use `random_state=42` and `n_init=10`?

* `random_state` ensures reproducibility.
* `n_init=10` reduces instability from random initialization.

I learned that KMeans can produce slightly different results if randomness is not controlled.

---

# 6️⃣ Personal Reflection

This project helped me:

* Understand unsupervised learning beyond theory
* Translate clustering results into marketing strategy
* Appreciate reproducibility and structured workflow
* Think more critically about preprocessing decisions

Most important realization:

> Clustering is not about maximizing mathematical performance — it is about producing interpretable and actionable business segments.

