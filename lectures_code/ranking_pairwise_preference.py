import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Simulate ordinal labels for 10 data points
X = np.random.rand(10, 5)  # 10 samples, 5 features
# print(X)

y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])  # Ordinal classes: 0 < 1 < 2

# Step 2: Generate pairwise training data (xi - xj) and preference labels
X_pairs = []
y_pairs = []

for i in range(len(X)):
    for j in range(len(X)):
        print(i, j, y[i], y[j])

        if i == j:
            continue
        if y[i] > y[j]:
            X_pairs.append(X[i] - X[j])
            y_pairs.append(1)
        elif y[i] < y[j]:
            X_pairs.append(X[j] - X[i])
            y_pairs.append(0)  # Note: inverse direction (label = 0)

X_pairs = np.array(X_pairs)
y_pairs = np.array(y_pairs)

# Step 3: Train a logistic regression model on pairwise preferences
X_train, X_test, y_train, y_test = train_test_split(X_pairs, y_pairs, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate preference prediction accuracy
y_pred = model.predict(X_test)
print("Pairwise Accuracy:", accuracy_score(y_test, y_pred))


# STEP 6: Define a new sample (same dimensionality as X)
x_new = np.random.rand(5)  # new item, 4 features

# STEP 7: Compare x_new against each item in X
wins = 0
for xi in X:
    diff = x_new - xi
    pred = model.predict(diff.reshape(1, -1))[0]
    if pred == 1:
        wins += 1  # model says x_new > xi

# STEP 8: Compute rank (higher rank = better)
# Rank 1 means it's the best (beats all others)
rank = len(X) - wins + 1

print("x_new:", x_new)
print("Beats", wins, "items out of", len(X))
print("Predicted Rank of x_new:", rank)
