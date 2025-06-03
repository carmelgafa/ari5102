from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Generate synthetic data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=0)

# Fit GMM
gmm = GaussianMixture(n_components=4, random_state=0).fit(X)
labels = gmm.predict(X)

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1],
            s=200, c='red', marker='x')
plt.title("GMM Clustering (Soft K-Means)")
plt.show()
