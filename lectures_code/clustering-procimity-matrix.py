import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define labels and matrix
labels = ['A', 'B', 'C', 'D']
D = np.array([
    [0.0, 2.0, 4.5, 3.1],
    [2.0, 0.0, 1.5, 2.2],
    [4.5, 1.5, 0.0, 3.0],
    [3.1, 2.2, 3.0, 0.0]
])

# Create dataframe for heatmap
df = pd.DataFrame(D, index=labels, columns=labels)

# Plot the heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(df, annot=True, cmap="YlGnBu", square=True, cbar_kws={"label": "Dissimilarity"})
plt.title("Proximity (Dissimilarity) Matrix")
plt.show()