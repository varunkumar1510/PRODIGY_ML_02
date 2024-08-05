import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data from Google Drive
data = pd.read_csv('/content/drive/MyDrive/Colab Datasets/Mall_Customers.csv')

# Data Preprocessing
# Convert categorical data to numerical if needed (e.g., Gender)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

# Select relevant features for clustering
features = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Find the optimal number of clusters
def find_optimal_k(scaled_features):
    wcss = []  # Within-cluster sum of squares
    silhouette_avg = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
        if k > 1:
            labels = kmeans.labels_
            silhouette_avg.append(silhouette_score(scaled_features, labels))
    
    plt.figure(figsize=(12, 6))
    
    # Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')

    # Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), silhouette_avg, marker='o')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')

    plt.show()
    
    return wcss, silhouette_avg

wcss, silhouette_avg = find_optimal_k(scaled_features)

# Assuming optimal k is determined (let's say k=5 for this example)
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
clusters = kmeans.fit_predict(scaled_features)

# Add cluster labels to the original data
data['Cluster'] = clusters

# Save the results to a new CSV file
output_file = '/content/drive/MyDrive/Colab Datasets/Clustered_Customers.csv'
data.to_csv(output_file, index=False)

print(f"Clustering completed and saved to '{output_file}'.")

# Print the metric values
print("WCSS for each number of clusters:")
for i, wcss_value in enumerate(wcss, start=1):
    print(f"Clusters: {i}, WCSS: {wcss_value}")

print("\nSilhouette Scores for each number of clusters:")
for i, score in enumerate(silhouette_avg, start=2):
    print(f"Clusters: {i}, Silhouette Score: {score}")
