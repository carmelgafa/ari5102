import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# Generate synthetic dataset
X, _ = make_blobs(n_samples=40, centers=3, cluster_std=1.0, random_state=42)

# List of linkage methods to compare
methods = ['single', 'complete', 'average']

# Create linkage matrices for each method
linkage_matrices = {method: linkage(X, method=method) for method in methods}

# Plot dendrograms side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, method in zip(axes, methods):
    dendrogram(linkage_matrices[method], ax=ax, color_threshold=0)
    ax.set_title(f"{method.capitalize()} Linkage")
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Distance")

plt.tight_layout()
plt.show()
