from gplearn.genetic import SymbolicRegressor
from joblib import dump, load # model persistence
import pprint # Nice printing
# I will attempt to adapt the Freshwater dataset
# Load data
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from math import sin, cos, tanh, log

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

sns.pairplot(mdf)
plt.show()

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

from scipy import stats
dmdf = mdf.dropna()
# zmdf = mdf[(np.abs(stats.zscore(mdf)) < 3).all(axis=1)]

for col in dmdf.columns.values:
    print(np.shape(dmdf))
    # dmdf[col+"_z"] = dmdf[col].apply(stats.zscore)
    dmdf[col+"_z"] = stats.zscore(dmdf[col])
for col in dmdf.columns.values[7:]:
    dmdf = dmdf[np.abs(dmdf[col]) < 7]  # keep if value is less than 3 std
# drop zscore columns
dmdf = dmdf.drop([
       'Soil Porewater Specific Conductance (uS/cm)_z',
       'Soil Porewater Temperature (Â°C)_z',
       'Average Height Dominant (cm)_z', 'Flood Depth (mm)_z',
       'Salinity Perturbation Ratio_z', 'avg_percentflooded (%)_z',
       'Accretion Rate Shortterm_z'
], axis=1)
# # mdf[(np.abs(stats.zscore(mdf.select_dtypes(exclude='object'))) < 10).all(axis=1)]
# sns.pairplot(dmdf)
# plt.show()




# ...........................................
# Standardizing
y = dmdf[outcome_var_str]
X = dmdf.drop(outcome_var_str, axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into train test and split segments
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=.25, train_size=.75)

# sns.pairplot(X_train)
# plt.show()

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
#
# # Recursive feature Elimination to locate variable importances
# rf = RandomForestRegressor()
#
# rfe = RFE(estimator=rf, step=1) # instantiate Recursive Feature Eliminator
# rfe.fit(X_train, y_train) # fit our training data with the RFE algorithm
# # We then create a dataframe of our features with associated rankings
# vardf = pd.DataFrame(zip(X_train.columns,
#                 rfe.ranking_), columns=['Variable', 'Ranking'])
# print("DataFrame of Features with associated Rank")
# pprint.pprint(vardf)
#
# # We then eliminate any features with a rank less than 1, these will be our "good features"
# vardf = vardf[vardf['Ranking'] == 1]
# print("\n")
# print("DataFrame of Features with Rank of 1")
# pprint.pprint(vardf)
# good_features = list(vardf.Variable.values)
# print("\n")
# print("Good Features: ", good_features)
# #Compute new training and testing data only using the good features from RFE
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y, test_size=.25, random_state=42)

# Hyperparameter tuning
# range of point_crossover variable
pc = [x for x in np.arange(start=0.5, stop=0.95, step=0.025)]
# range of subtree mutation probability
ps = [(1-x)/3 for x in np.arange(start=0.5, stop=0.95, step=0.025)]  # a little better but still half-azz
# ps = [x/3 for x in pc]
# p hoist = p subtree mutation
ph = ps
# p_point_mutation
# pp = 1-pc-ps-ph
# pp = [1-(3*x)-(2*x) for x in ph]
pp = [1-x-(2*((1-x)/3)) for x in np.arange(start=0.5, stop=0.95, step=0.025)]  # more accurate
# parismony coefficient
parsimony_coef = [x for x in np.arange(start=0.0005, stop=0.0015, step=0.0005)]

# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'p_crossover': pc,
               'p_subtree_mutation': ps,
               # 'p_subtree_mutation': pc/3,
               'p_hoist_mutation': ph,
               'p_point_mutation': pp,
               'parsimony_coefficient': parsimony_coef}
print("\n")
pprint.pprint(random_grid)

# initialize symbolicregressor
est_gp = SymbolicRegressor(population_size=5000,
                           n_jobs=-1,
                           const_range=(-1, 1),
                           # feature_names=feature_names,
                           # init_depth=(4, 10),
                           init_method='half and half',
                           function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt'),
                           tournament_size=3,  # default value = 20
                           generations=90,
                           stopping_criteria=0.01,
                           # p_crossover=,
                           # p_subtree_mutation=0.1,
                           # p_hoist_mutation=0.01,
                           # p_point_mutation=0.1,
                           # max_samples=0.9,
                           # verbose=1,
                           # parsimony_coefficient=0.0005,
                           # random_state=0,
                           metric='mean absolute error')

sr_random = RandomizedSearchCV(estimator=est_gp, param_distributions=random_grid, n_iter=100, cv=3, verbose=3,
                               random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')  #Fit the random search model
sr_random.fit(X_train, y_train)
pprint.pprint(sr_random.best_params_)

best_params = sr_random.best_params_

# .................. Above this took a long time so do not re-run until later

# using what i learned from the random search to do a grid search with cross validation
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {
    'p_crossover': [0.4, 0.5, 0.6],
    'p_subtree_mutation': [0.1, 0.133, 0.166],
    'p_hoist_mutation': [0.14, 0.15, 0.16],
    'p_point_mutation': [0.06, 0.08, 0.1],
    'parsimony_coefficient': [0.0005]
}
# Create a based model
est_gp_grid = SymbolicRegressor(population_size=5000,
                           n_jobs=-1,
                           const_range=(-1, 1),
                           init_method='half and half',
                           function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'sqrt'),
                           tournament_size=3,  # default value = 20
                           generations=20,
                           stopping_criteria=0.01,
                           metric='mean absolute error')
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=est_gp_grid, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=3, scoring='neg_mean_absolute_error')
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_
y_pred = best_grid.predict(X_test)
dump(best_grid, "D:\Etienne\crmsDATATables\ml_dump\SRTunedGS_SSML.joblib")

# ................... Above is also computationally expensive , do not rerun if not necessary

# X = dmdf.drop("Accretion Rate Shortterm", axis=1)
# y = dmdf['Accretion Rate Shortterm']
sr_tuned = load("D:\Etienne\crmsDATATables\ml_dump\SRTunedGS_SSML.joblib")
sr_tuned.fit(X, y)

# model metrics
# Helper Function: This prints out regression metrics.
def regression_results(y_true, y_pred):

    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)


    print('explained_variance: ', round(explained_variance,4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))


y_pred = sr_tuned.predict(X_test)
regression_results(y_test, y_pred)

# sns.scatterplot(y_test, y_pred)
# plt.show()
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
ax.set_aspect('equal')  # can also be equal

ax.set_xlim(lims)
ax.set_ylim(lims)
# ax.set_title(str(feature_names))
fig.show()












# est_gp.fit(X_train, y_train)
# y_pred = est_gp.predict(X_test)
#
# converter = {
#     'sub': lambda x, y: x - y,
#     'div': lambda x, y: x / y,
#     'mul': lambda x, y: x * y,
#     'add': lambda x, y: x + y,
#     # 'neg': lambda x: -x,
#     'pow': lambda x, y: x ** y,
#     'abs': lambda x: abs(x),
#     'sin': lambda x: sin(x),
#     # 'arsin': lambda x: sin(x),
#     'cos': lambda x: cos(x),
#     'tan': lambda x: tanh(x),
#     'log': lambda x: log(x),
#     # 'inv': lambda x: 1 / x,
#     'sqrt': lambda x: x ** 0.5,
#     'pow3': lambda x: x ** 3
# }
#
# # print(est_gp._program)
# # equation = sympify((est_gp._program), locals=converter)
#
# def regression_results(y_true, y_pred):
#
#     # Regression metrics
#     explained_variance = metrics.explained_variance_score(y_true, y_pred)
#     mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
#     mse = metrics.mean_squared_error(y_true, y_pred)
#     median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
#
#
#     print('explained_variance: ', round(explained_variance,4))
#     print('MAE: ', round(mean_absolute_error, 4))
#     print('MSE: ', round(mse, 4))
#     print('RMSE: ', round(np.sqrt(mse), 4))
#
# regression_results(y_test, y_pred)
#
#
# # sns.scatterplot(y_test, y_pred)
# # plt.show()
# fig, ax = plt.subplots()
# ax.scatter(y_test, y_pred)
#
# lims = [
#     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
# ]
#
# plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
# ax.set_aspect('equal')  # can also be equal
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# # ax.set_title(str(feature_names))
# fig.show()
#



