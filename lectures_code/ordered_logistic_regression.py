import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Example: Constructing a small diamond dataset
data = {
    "carat": [0.23, 0.21, 0.23, 0.29, 0.31, 0.24, 0.24, 0.26, 0.22, 0.23],
    "cut": ["Ideal", "Premium", "Good", "Premium", "Good", "Very Good", "Very Good", "Very Good", "Fair", "Very Good"],
    "color": ["E", "E", "E", "I", "J", "J", "J", "H", "E", "H"],
    "clarity": ["SI2", "SI1", "VS1", "VS2", "SI2", "VVS2", "VVS1", "SI1", "VS2", "VS1"],
    "depth": [61.5, 59.8, 56.9, 62.4, 63.3, 62.8, 62.3, 61.9, 65.1, 59.4],
    "table": [55.0, 61.0, 65.0, 58.0, 58.0, 57.0, 57.0, 55.0, 61.0, 61.0],
    "price": [326, 326, 327, 334, 335, 336, 336, 337, 337, 338],
    "volume": [38.2, 34.5, 38.1, 46.7, 51.9, 38.7, 38.8, 42.3, 36.4, 38.7]
}

df = pd.DataFrame(data)

# 1. Convert 'cut' to an ordered categorical variable
from pandas.api.types import CategoricalDtype

cut_order = CategoricalDtype(
    categories=['Fair', 'Good', 'Ideal', 'Very Good', 'Premium'], 
    ordered=True
)
df['cut'] = df['cut'].astype(cut_order)

# 2. Define the model
mod = OrderedModel(
    df['cut'],
    df[['volume', 'price', 'carat']],
    distr='logit'
)

# 3. Fit the model
res = mod.fit(method='bfgs')
print(res.summary())

# 4. Get threshold values
k_exog = mod.exog.shape[1]
th_params = mod.transform_threshold_params(res.params[k_exog:])
print("Thresholds:", th_params)


# 5. Predict probabilities for first 5 samples
predicted_probs = res.model.predict(res.params, exog=df[['volume', 'price', 'carat']].iloc[:5])
print("Predicted Probabilities (first 5 rows):")
print(predicted_probs)
