import pandas as pd
pd.set_option('display.max_columns', None)       # don’t truncate columns
pd.set_option('display.max_rows',    None)       # don’t truncate rows
pd.set_option('display.width',       0)          # let pandas choose the width
pd.set_option('display.expand_frame_repr', False)

# Relative path to the dataset (from the project root)
DATA_PATH = "data/ShopNow Dataset.csv"

def main():
    """Load data and print quick checks (shape, columns, preview)."""
    # Load CSV into a DataFrame
    df = pd.read_csv(DATA_PATH)

    # Quick sanity checks
    print("Shape (rows, cols):", df.shape)
    print("\nColumns:\n", df.columns)
    print("\nPreview:\n", df.head())

    # ===== Step 2: select segmentation variables =====
    # Core behavioral (RFM) variables
    rfm_cols = ["recency_days", "orders_last_12m", "avg_order_value"]

    # Automatically grab all category share columns (e.g., cat_share_electronics, ...)
    cat_share_cols = [c for c in df.columns if c.startswith("cat_share_")]

    # Final segmentation feature list
    seg_cols = rfm_cols + cat_share_cols
    print("Type of seg_cols:", type(seg_cols))
    #list of segmentation columns
    print("\nSegmentation columns:", seg_cols)

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Keep only segmentation columns, and make it as a new DataFrame (X)
    X = df[seg_cols].copy()

    # ---- Basic cleaning ----
    # If there are missing values, fill them with median (simple + robust default)
    # (Later you can justify: avoids dropping customers.)
    X = X.replace([np.inf, -np.inf], np.nan) # Handle infinite values if they exist
    X = X.fillna(X.median(numeric_only=True))

    # ---- Scale ONLY RFM ----
    scaler = StandardScaler() # StandardScaler() standardizes features by removing the mean and scaling to unit variance.
    X_scaled = X.copy()
    X_scaled[rfm_cols] = scaler.fit_transform(X_scaled[rfm_cols])

    print("\nCheck means (RFM should be ~0 after scaling):")
    print(X_scaled[rfm_cols].mean())

# ===== Step 4: Run KMeans for k = 2..10 and plot elbow =====
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    wcss = []  # Within-Cluster Sum of Squares (a.k.a. inertia)
    k_values = range(2, 11) # We start from 2 because 1 cluster is trivial (all customers in one group)

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        wcss.append(model.inertia_)

    plt.figure()
    plt.plot(list(k_values), wcss, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS / Inertia (lower is tighter clusters)")
    plt.title("Elbow Curve (KMeans)")
    plt.show()

    # ===== Step 5: Train final KMeans model =====
    final_k = 4 #based on elbow plot, we choose k=4 as the optimal number of clusters
    final_model = KMeans(n_clusters=final_k, random_state=42, n_init=10) 

    # Fit model
    final_model.fit(X_scaled) # Fit the KMeans model to the scaled data (X_scaled). 
    #This step computes the cluster centers and assigns each data point to the nearest cluster center, 
    # effectively learning the clusters in the data.

    # Predict cluster labels
    cluster_labels = final_model.predict(X_scaled)
    # The predict() method assigns each data point in X_scaled to the nearest cluster center learned during fitting,
    # resulting in an array of cluster labels (integers from 0 to final_k-1) that indicate which cluster each data point belongs to.

    # Add cluster label to original dataframe
    df["cluster"] = cluster_labels

    print("\nCluster counts:")
    print(df["cluster"].value_counts())
    # ===== Step 6: Cluster profiling (means by cluster) =====
    cluster_profile = df.groupby("cluster")[seg_cols].mean()

    print("\nCluster Profile (means):")
    print(cluster_profile)

# Run main() only when this file is executed directly (not when imported)
if __name__ == "__main__":
    main()