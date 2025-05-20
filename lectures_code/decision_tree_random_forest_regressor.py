import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic data
rng = np.random.RandomState(42)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel() + 0.2 * rng.randn(80)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
rf_reg.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_reg.predict(X_test)
print(f"Test MSE: {mean_squared_error(y_test, y_pred):.3f}")

# Plot results
x_plot = np.linspace(0, 5, 500).reshape(-1, 1)
y_plot = rf_reg.predict(x_plot)

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, label="Training Data", alpha=0.6)
plt.plot(x_plot, y_plot, color="red", label="RF Prediction", linewidth=2)
plt.title("Random Forest Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
