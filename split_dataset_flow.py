# Workflow:
# 1. Data preparation
# 2. Feature Selection
# 3. Hyperparameter Tuning
# 4. Model Metrics
# 5. Summary
from joblib import dump, load # model persistence
import pprint # Nice printing
# I will attempt to adapt the Freshwater dataset
# Load data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

import run

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

df = pd.read_csv(r"D:\Etienne\crmsDATATables\basins_time_invariant\Terrebonne.csv", encoding="unicode escape")
# Identify the fundamental variables potentially contributing to accretion and/or surface elevation
fun_vars = [
    "Simple site", "Soil Porewater Salinity (ppt)", "Soil Porewater Specific Conductance (uS/cm)", 'Soil Porewater Temperature (Ã\x82Â°C)',
    "Average Height Dominant (cm)", "Flood Depth (mm)", "Salinity Perturbation Ratio", "Distance from Water",
    "avg_percentflooded (%)"
]

# Append the sought outcome variable to the list
outcome_var_str = "Average Accretion (mm)"
fun_vars.append(outcome_var_str)

# Extract the desired dataframe
mdf = df[fun_vars]

# Visualize dataframe with pairplot
sns.pairplot(mdf)
plt.show()

# Transformations
# dRop soil porewater salinity because of colinearity with soil porewater conductance
mdf = mdf.drop("Soil Porewater Salinity (ppt)", axis=1)
# Fill dominant tree height with median previous value using a backward fill method that takes into account the site inwhich data is pulled from
mdf["Average Height Dominant (cm)"] = backward_fill_nans(mdf, "Average Height Dominant (cm)")
# deal with outliers with the imputation method defined in run.py
mdf = run.median_outlier_imputation(mdf.drop("Simple site", axis=1))
# log the salinity perturbation ratio
mdf['Salinity Perturbation Ratio'] = np.log(mdf['Salinity Perturbation Ratio'])
mdf["Soil Porewater Specific Conductance (uS/cm)"] = np.log(mdf["Soil Porewater Specific Conductance (uS/cm)"])
sns.pairplot(mdf)
plt.show()


# Drop rows with nans
dd = mdf.dropna()
# dd = dd.drop("Simple site", axis=1)

# Recursive feature Elimination
rf = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(dd.drop(['Average Accretion (mm)'], axis=1),
                                                    dd['Average Accretion (mm)'], test_size=.25, train_size=.75)

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
X_train, X_test, y_train, y_test = train_test_split(dd.drop("Average Accretion (mm)", axis=1),
                                                    dd["Average Accretion (mm)"], test_size=.25, random_state=42)

# Hyperparameter tuning
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(40, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print("\n")
pprint.pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=3,
                               random_state=42, n_jobs = -1, scoring='neg_mean_absolute_error')  #Fit the random search model
rf_random.fit(X_train, y_train)
pprint.pprint(rf_random.best_params_)






# Plot number of features VS. cross-validation scores
# Create a based model
# ...............

# grid search with cross validation
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [78, 80, 82, 84],
#     'max_features': [2, 4, 6],
#     'min_samples_leaf': [1, 2],
#     'min_samples_split': [4, 5, 6],
#     'n_estimators': [1400, 1600, 1800]
# }
param_grid = {'bootstrap': [True],
 'max_depth': [36, 38, 40, 42, 44],
 # 'max_features': 'sqrt',
 'min_samples_leaf': [3, 4, 5],
 'min_samples_split': [1, 2, 3],
 'n_estimators': [390, 400, 410]}

rf_grid = RandomForestRegressor()  # Instantiate the grid search base model
grid_search = GridSearchCV(estimator=rf_grid, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=3, scoring='neg_mean_absolute_error',
                           refit=True)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_
y_pred = best_grid.predict(X_test)
dump(best_grid, "D:\Etienne\crmsDATATables\ml_dump\RandForestGS_terre_largeSSML.joblib")







# ==================================================
X = dd.drop(['Average Accretion (mm)'], axis=1)
y = dd['Average Accretion (mm)']
rf_tuned = load("D:\Etienne\crmsDATATables\ml_dump\RandForestGS_terre_largeSSML.joblib")
rf_tuned.fit(X, y)

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


y_pred = rf_tuned.predict(X_test)
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
ax.set_title('Random Forest Model Prediction of Sediment Accretion in Terrebonne Basin')
ax.set_ylabel('Observed Accretion (mm)')
ax.set_xlabel('Predicted Accretion (mm)')
# ax.set_title(str(feature_names))
fig.show()


# TRansform the training data of water temperature with quantile transformer to make normal distribution



