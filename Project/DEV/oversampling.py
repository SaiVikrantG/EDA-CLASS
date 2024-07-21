import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the preprocessed dataset
df1 = pd.read_csv('preprocessed_HAM10000.csv')

# Implement K-means clustering
def kmeans(X, k, max_iters=100):
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)
        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return labels, centroids

# Implement SMOTE
def SMOTE(X, y, k_neighbors=5, oversampling_ratio=1):
    synthetic_samples = []
    for i in range(len(X)):
        neighbors = np.argsort(np.linalg.norm(X - X[i], axis=1))[1:k_neighbors+1]
        for _ in range(oversampling_ratio):
            nn = np.random.choice(neighbors)
            diff = X[nn] - X[i]
            synthetic_sample = X[i] + np.random.random() * diff
            synthetic_samples.append(synthetic_sample)
    return np.vstack([X, synthetic_samples]), np.concatenate([y, np.full(len(synthetic_samples), np.argmax(np.bincount(y)))])

# Consider only specific columns for clustering and oversampling
X = df1[['age', 'dx_type', 'localization']]
y = df1['dx']

# Apply K-means clustering
labels, centroids = kmeans(X.values, 7)

# Plot clusters
plt.scatter(X.values[:, 0], X.values[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', s=200, label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Age')
plt.ylabel('Localization')
plt.legend()
plt.show()

# Apply SMOTE to each cluster
oversampled_data = []
for i in range(7):
    cluster_data = df1[labels == i]
    cluster_X = cluster_data[['age', 'dx_type', 'localization']].values
    cluster_y = cluster_data['dx'].values
    X_resampled, y_resampled = SMOTE(cluster_X, cluster_y)
    oversampled_data.append(pd.DataFrame(np.column_stack([X_resampled, y_resampled]), columns=['age', 'dx_type', 'localization', 'dx']))

# Combine oversampled data from all clusters
oversampled_df = pd.concat(oversampled_data)

# Save oversampled dataset
oversampled_df.to_csv('oversampled_HAM10000.csv', index=False)

