import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
rng = np.random.RandomState(42)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * rng.randn(80)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Base model: shallow decision tree
base_tree = DecisionTreeRegressor(max_depth=3, random_state=42)

# Bagging regressor
bagging_model = BaggingRegressor(
    estimator=base_tree,
    n_estimators=10,
    bootstrap=True,
    random_state=42
)

# Fit model
bagging_model.fit(X_train, y_train)

# Predict
y_pred = bagging_model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.3f}")

# Visualize
x_plot = np.linspace(0, 5, 500).reshape(-1, 1)
y_plot = bagging_model.predict(x_plot)

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, label="Training data", alpha=0.6)
plt.plot(x_plot, y_plot, color='red', label="Bagging prediction", linewidth=2)
plt.title("Bagging Regressor with Decision Trees")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
