import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

random_seed = 1000

cov = np.array([[1, -0.5], [-0.5, 1]])
mean = np.array([1, 3])
# Create the distribution
distr = multivariate_normal(mean=mean, cov=cov, seed=random_seed)
# Sample 100 points
data1 = distr.rvs(size=100)

cov = np.array([[1, 0.5], [0.5, 1]])
mean = np.array([2.5 , 1])
# Create the distribution
distr = multivariate_normal(mean=mean, cov=cov, seed=random_seed)
# Sample 100 points
data2 = distr.rvs(size=100)

cov = np.array([[1, 0], [0, 1]])
mean = np.array([2.5, -7])
# Create the distribution
distr = multivariate_normal(mean=mean, cov=cov, seed=random_seed)
# Sample 100 points
data3 = distr.rvs(size=100)

# Combine the data
data = np.concatenate((data1, data2, data3), axis=0)

plt.figure(figsize=(8, 6))
plt.scatter(data1[:, 0], data1[:, 1], alpha=0.5, c='red')
plt.scatter(data2[:, 0], data2[:, 1], alpha=0.5, c='green')
plt.scatter(data3[:, 0], data3[:, 1], alpha=0.5, c='purple')
plt.title('Sampled Data from Multivariate Normal Distributions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.grid()
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=1000, n_init='auto').fit(data)
labels = kmeans.labels_
data_c1 = data[labels == 0]
data_c2 = data[labels == 1]
data_c3 = data[labels == 2]

plt.figure(figsize=(8, 6))
plt.scatter(data_c1[:, 0], data_c1[:, 1], alpha=0.5, c='red')
plt.scatter(data_c2[:, 0], data_c2[:, 1], alpha=0.5, c='green')
plt.scatter(data_c3[:, 0], data_c3[:, 1], alpha=0.5, c='purple')
plt.title('Sampled Data from Multivariate Normal Distributions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
plt.grid()
plt.show()


kmeans = KMeans(n_clusters=2, random_state=1000, n_init='auto').fit(data)
labels = kmeans.labels_
data_c1 = data[labels == 0]
data_c2 = data[labels == 1]

plt.figure(figsize=(8, 6))
plt.scatter(data_c1[:, 0], data_c1[:, 1], alpha=0.5, c='red')
plt.scatter(data_c2[:, 0], data_c2[:, 1], alpha=0.5, c='green')
plt.title('Sampled Data from Multivariate Normal Distributions')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(['Cluster 1', 'Cluster 2'])
plt.grid()
plt.show()