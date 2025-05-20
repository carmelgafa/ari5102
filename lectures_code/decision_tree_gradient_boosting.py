import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic regression data
rng = np.random.RandomState(42)
X = np.sort(5 * rng.rand(100, 1), axis=0)
y = np.sin(X).ravel() + 0.3 * rng.randn(100)  # noisy sine curve

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Gradient Boosting Regressor
gbr = GradientBoostingRegressor(
    n_estimators=100,        # number of boosting stages
    learning_rate=0.1,       # shrinkage rate
    max_depth=3,             # depth of each tree
    random_state=42
)
gbr.fit(X_train, y_train)

# Predict and evaluate
y_pred = gbr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.3f}")

# Plot predictions
X_plot = np.linspace(0, 5, 500).reshape(-1, 1)
y_plot = gbr.predict(X_plot)

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, color="blue", label="Training Data", alpha=0.6)
plt.plot(X_plot, y_plot, color="red", label="Gradient Boosting Prediction", linewidth=2)
plt.title("Gradient Boosting Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
