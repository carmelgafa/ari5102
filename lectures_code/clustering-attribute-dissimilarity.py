import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import OneHotEncoder

# Example mixed-type dataset
df = pd.DataFrame({
    "height": [1.75, 1.80, 1.65],              # Quantitative
    "grade": ['A', 'C', 'B'],                  # Ordinal
    "color": ['red', 'blue', 'red']           # Categorical
})

# Map ordinal grades to contiguous integers
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
df["grade"] = df["grade"].map(grade_map)

print(df)


# One-hot encode categorical variable
ohe = OneHotEncoder(drop=None, sparse_output=False)
encoded_color = ohe.fit_transform(df[["color"]])

print(encoded_color)

# Combine into single feature matrix
X_quant_ordinal = df[["height", "grade"]].to_numpy()

print(X_quant_ordinal)

X = np.hstack([X_quant_ordinal, encoded_color])
print('----')
print(X)
print('----')

# Compute pairwise dissimilarities using Euclidean distance
dist_matrix = pairwise_distances(X, metric='euclidean')

# Display distance matrix
dist_df = pd.DataFrame(dist_matrix, index=df.index, columns=df.index)
print(dist_df.round(2))
