# Level 2 - Task 3

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the cleaned dataset
df = pd.read_csv('Stock_Price_Cleaned.csv')
print("Cleaned dataset loaded successfully.")

# Show first five rows of the dataset
print("First five rows of the cleaned dataset:")
print(df.head())

# Numerical features for clustering
df_clusters = df[['open', 'high', 'low', 'close', 'volume']]

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_clusters)
print("Features standardized successfully.")

# Convert back to DataFrame for easier handling
scaled_df = pd.DataFrame(scaled_features, columns=df_clusters.columns)

# Determine the optimal number of clusters using the Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(scaled_df)
    wcss.append(kmeans.inertia_)
    
# Plot the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters(K)')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')

plt.savefig('elbow_method.png')
print("✅ Elbow method plot saved as 'elbow_method.png'")
plt.show()

# From the elbow plot, let's take the optimal number of clusters is 3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)

# Visualize the clusters using 2 features: 'open' and 'close'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='open', y='close', hue='Cluster', palette='Set1')
plt.title('K-Means Clustering of Stock Prices (Open vs Close)')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend(title='Cluster')

plt.savefig('kmeans_clusters.png')
print("K-Means clustering plot (Open vs Close) saved as 'kmeans_clusters.png'")
plt.show()

# Visualize the clusters using 2 features: 'high' vs 'low'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='high', y='low', hue='Cluster', palette='Set2')
plt.title("Clusters based on High vs Low Prices")
plt.xlabel("High Price")
plt.ylabel("Low Price")
plt.legend(title="Cluster")

plt.savefig('kmeans_high_low_clusters.png')
print("K-Means clustering plot (High vs Low) saved as 'kmeans_high_low_clusters.png'")
plt.show()

# Visualize the clusters using 2 features: 'volume' vs 'close'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='volume', y='close', hue='Cluster', palette='coolwarm')
plt.title("Clusters based on Volume vs Close Prices")
plt.xlabel("Volume")
plt.ylabel("Closing Price")
plt.legend(title="Cluster")

plt.savefig('kmeans_volume_close_clusters.png')
print("K-Means clustering plot (Volume vs Close) saved as 'kmeans_volume_close_clusters.png'")
plt.show()

# Visualize the clusters using 2 features: 'open' vs 'volume'
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='open', y='volume', hue='Cluster', palette='viridis')
plt.title("Clusters based on Open Price vs Volume")
plt.xlabel("Opening Price")
plt.ylabel("Volume")
plt.legend(title="Cluster")

plt.savefig('kmeans_open_volume_clusters.png')
print("K-Means clustering plot (Open vs Volume) saved as 'kmeans_open_volume_clusters.png'")
plt.show()


# Show sample of clustered data
print("Sample of clustered data:")
print(df.head())

# Save the clustered dataset
df.to_csv("Clustered_Stock_Data.csv", index=False)
print("Clustered dataset saved as 'Clustered_Stock_Data.csv'")

