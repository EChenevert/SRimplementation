# Workflow:
# 1. Data preparation
# 2. Feature Selection
# 3. Hyperparameter Tuning
# 4. Model Metrics
# 5. Summary
from gplearn.genetic import SymbolicRegressor
from joblib import dump, load # model persistence
import pprint # Nice printing
# I will attempt to adapt the Freshwater dataset
# Load data
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from math import sin, cos, tan, log

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import quantile_transform, MinMaxScaler
from sympy import sympify

import run
# Handy function for nan filling
def backward_fill_nans(df, column):
    """Function that forward fills nans but considers the site in which values are pulled from"""
    # for row in range(len(df[column])):
    #     if df['Simple site'][row] == df['Simple site'][row-1] and df[column] == :  # If the current value was taken at the same CRMS site
    nan_ls = list(df.loc[pd.isna(df[column])].index)  # returns a list of indices where nans are present in column
    nan_ls.sort(reverse=True)
    # print(len(nan_ls))
    # relevent_nans = 0
    for idx in nan_ls:
        # print(idx)
        if idx < 986 and df['Simple site'][idx] == df['Simple site'][idx + 1]:  # If the current value was taken at the same CRMS site

            df[column][idx] = df[column][idx+1]  # replace with previous value

    return df[column]

# Grab the dataset
df = pd.read_csv(r"D:\Etienne\crmsDATATables\community_specific_datasets\Freshwater.csv", encoding="unicode escape")
# Identify the fundamental variables potentially contributing to accretion and/or surface elevation
fun_vars = [
    "Simple site", "Soil Porewater Salinity (ppt)", "Soil Porewater Specific Conductance (uS/cm)", "Soil Porewater Temperature (Â°C)",
    "Average Height Dominant (cm)", "Flood Depth (mm)", "Salinity Perturbation Ratio", "avg_percentflooded (%)"
]

# Append the sought outcome variable to the list
outcome_var_str = "Accretion Rate Shortterm"
fun_vars.append(outcome_var_str)

# Extract the desired dataframe
mdf = df[fun_vars]

# # Visualize dataframe with pairplot
# sns.pairplot(mdf)
# plt.show()

# Transformations
# dRop soil porewater salinity because of colinearity with soil porewater conductance
mdf = mdf.drop("Soil Porewater Salinity (ppt)", axis=1)
# Fill dominant tree height with median previous value using a backward fill method that takes into account the site inwhich data is pulled from
mdf["Average Height Dominant (cm)"] = backward_fill_nans(mdf, "Average Height Dominant (cm)")
# deal with outliers with the imputation method defined in run.py
mdf = run.median_outlier_imputation(mdf.drop("Simple site", axis=1))
# # log the salinity perturbation ratio due to their distributions
mdf['Salinity Perturbation Ratio'] = np.log(mdf['Salinity Perturbation Ratio'])
mdf["Soil Porewater Specific Conductance (uS/cm)"] = np.log(mdf["Soil Porewater Specific Conductance (uS/cm)"])
sns.pairplot(mdf)
plt.show()

# Avoided any other transformations right now to avoid data leakage

# Drop rows with nans
dd = mdf.dropna()

# # Scale the data with a Min Max scalar
# scaler = MinMaxScaler()
# scaled_dd = scaler.fit_transform(dd.drop(['Accretion Rate Shortterm'], axis=1))
# # Split the data into train test and split segments
# X_train, X_test, y_train, y_test = train_test_split(scaled_dd,
#                                                     dd['Accretion Rate Shortterm'], test_size=.25, train_size=.75)
# column_values = dd.drop(['Accretion Rate Shortterm'], axis=1).columns.values
# X_train = pd.DataFrame(data=X_train,
#                   # index = index_values,
#                   columns=column_values)
# sns.pairplot(X_train)
# plt.show()


# Split the data into train test and split segments
X_train, X_test, y_train, y_test = train_test_split(dd.drop(['Accretion Rate Shortterm'], axis=1),
                                                    dd['Accretion Rate Shortterm'], test_size=.25, train_size=.75)

sns.pairplot(X_train)
plt.show()

# # .............................
# # Transform the X_train data to normal distribution "make math easier"
# # Yeo transform for the selected columns
# for col in ["Average Height Dominant (cm)"]:
#     X_train[col] = yeojohnson(X_train[col], 0)
# sns.pairplot(X_train)
# plt.show()
#
# for col in ["Soil Porewater Temperature (Â°C)"]:
#     # X_train[col] = boxcox(X_train[col], 0)
#     X_train[col] = quantile_transform(np.array(X_train[col]).reshape(-1, 1), n_quantiles=10, random_state=0, copy=True)
# sns.pairplot(X_train)
# plt.show()
# # .......................................

# Recursive feature Elimination to locate variable importances
rf = RandomForestRegressor()

rfe = RFE(estimator=rf, step=1) # instantiate Recursive Feature Eliminator
rfe.fit(X_train, y_train) # fit our training data with the RFE algorithm
# We then create a dataframe of our features with associated rankings
vardf = pd.DataFrame(zip(X_train.columns,
                rfe.ranking_), columns=['Variable', 'Ranking'])
print("DataFrame of Features with associated Rank")
pprint.pprint(vardf)

# We then eliminate any features with a rank less than 1, these will be our "good features"
vardf = vardf[vardf['Ranking'] == 1]
print("\n")
print("DataFrame of Features with Rank of 1")
pprint.pprint(vardf)
good_features = list(vardf.Variable.values)
print("\n")
print("Good Features: ", good_features)
#Compute new training and testing data only using the good features from RFE
X_train, X_test, y_train, y_test = train_test_split(dd.drop("Accretion Rate Shortterm", axis=1),
                                                    dd["Accretion Rate Shortterm"], test_size=.25, random_state=42)




