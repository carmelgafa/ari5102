import numpy as np

# Two objects x_i and x_i' with p = 3 features
x_i = np.array([1.8, 3, 1])       # height (quant), grade (ordinal), red (binary)
x_ip = np.array([1.6, 2, 0])      # second object

# Define dissimilarity functions
def abs_diff(a, b): return abs(a - b)
def hamming(a, b): return int(a != b)

# Individual dissimilarities
d1 = abs_diff(x_i[0], x_ip[0])      # quantitative
d2 = abs_diff(x_i[1], x_ip[1])      # ordinal
d3 = hamming(x_i[2], x_ip[2])       # categorical

# Weights based on domain knowledge
weights = np.array([0.5, 0.2, 0.3])

# Weighted dissimilarity
D = weights[0]*d1 + weights[1]*d2 + weights[2]*d3
print(f"Weighted dissimilarity D(x_i, x_ip): {D:.2f}")
