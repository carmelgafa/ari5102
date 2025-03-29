import statsmodels.api as smf
import pandas as pd
import os
import matplotlib.pyplot as plt 

data_path = os.path.join(os.path.dirname(__file__), 'Housing_Data.csv')

houses_df = pd.read_csv(data_path)[["price","lotsize","bedrooms","prefarea"]]

# print(houses_df.head())

# prefarea yes should be blue tyriangle no should be red circle
houses_df["color"] = houses_df["prefarea"].apply(lambda x: "blue" if x == "yes" else "red")
houses_df["marker"] = houses_df["prefarea"].apply(lambda x: "^" if x == "yes" else "o")

plt.scatter(houses_df["lotsize"], houses_df["price"], c=houses_df["color"])
plt.xlabel("Lot Size")
plt.ylabel("Price")
plt.title("Price vs Lot Size")
# plt.show()


fit_1 = smf.gm(formula = 'price ~ lotsize + prefarea + lotsize:prefarea', data = houses_df, family = smf.families.Gaussian()).fit()
print(fit_1.summary())
print('Null Deviance: ', fit_1.null_deviance)
print('Residual Deviance: ', fit_1.deviance)
print('AIC: ', fit_1.aic)