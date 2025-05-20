from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import pandas as pd

# Load California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

# Define estimator
model = LinearRegression()

# Perform RFE to select top 5 features
selector = RFE(estimator=model, n_features_to_select=5)
selector.fit(X, y)

# Organize results
ranking_df = pd.DataFrame({
    "Feature": feature_names,
    "Selected": selector.support_,
    "Ranking": selector.ranking_
}).sort_values("Ranking")

# Output
print("Top 5 selected features:", list(ranking_df[ranking_df["Selected"]]["Feature"]))
print(ranking_df)
