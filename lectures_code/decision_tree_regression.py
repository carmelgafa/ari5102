import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# Generate synthetic data
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * rng.randn(80)  # noisy sine curve

# Fit regression tree
regr = DecisionTreeRegressor(max_depth=3)
regr.fit(X, y)

# Predict on a grid
X_test = np.linspace(0, 5, 500).reshape(-1, 1)
y_pred = regr.predict(X_test)

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(X, y, s=20, edgecolor="black", label="data")
plt.plot(X_test, y_pred, color="cornflowerblue", linewidth=2, label="prediction")
plt.title("Decision Tree Regression (max_depth=3)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
