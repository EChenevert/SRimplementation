
from run import median_outlier_imputation  # this is my own module that
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

import pprint  # Nice printing
from joblib import dump, load # model persistence


df = pd.read_csv(r"D:\Etienne\crmsDATATables\basins_time_invariant\Terrebonne.csv", encoding="unicode escape")
# Drop columns that have more than 75 % nan values
ddf = df.dropna(thresh=df.shape[0]*0.75, how='all', axis=1)

# Append the sought outcome variable to the list
outcome_var_str = "Accretion Rate Shortterm"

# Manually define variable sthat should be important
drop_vars = [
    'Simple site', 'Unnamed: 0', 'level_1', 'Season', 'Month (mm)', 'Surface Elevation Change Rate Shortterm',
    'Accretion Measurement 1 (mm)', 'Accretion Measurement 2 (mm)',
    'Accretion Measurement 3 (mm)', 'Accretion Measurement 4 (mm)',
    'Average Accretion (mm)', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)',
    'Latitude', 'Longitude', 'Direction (Collar Number)',
    'Direction (Compass Degrees)', 'Pin Number', 'Measurement Depth (ft)', 'percent_waterlevel_complete',
    'month', 'Staff Gauge (ft)'
]

dddf = ddf.drop(drop_vars, axis=1).dropna()

# Recursive feature Elimination
rf = RandomForestRegressor()
X_trainrf, X_testrf, y_trainrf, y_testrf = train_test_split(dddf.drop(['Accretion Rate Shortterm'], axis=1),
                                                    dddf['Accretion Rate Shortterm'], test_size=.4, train_size=.6)

rfe = RFE(estimator=rf, step=1) # instantiate Recursive Feature Eliminator
rfe.fit(X_trainrf, y_trainrf) # fit our training data with the RFE algorithm
# We then create a dataframe of our features with associated rankings
vardf = pd.DataFrame(zip(X_trainrf.columns,
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



# Important variables defined by the above recurisve feature selection
imp_vars = ['Bulk Density (g/cm3)', 'Dry Volume (cm3)', 'Soil Porewater Temperature (ÃÂ°C)',
         'Soil Porewater Salinity (ppt)', 'Average Height Herb (cm)', 'avg_percentflooded (%)',
         'Flood Depth (mm)', 'Distance from Water', 'Accretion Rate Shortterm']

rf_df = dddf[imp_vars]
sns.pairplot(rf_df)
plt.show()

# Transformations
# log transforms
rf_df['Distance from Water'] = [np.log(i) if i != 0 else 0 for i in rf_df['Distance from Water'] ]
rf_df['Bulk Density (g/cm3)'] = np.log(rf_df['Bulk Density (g/cm3)'])

sns.pairplot(rf_df)
plt.show()


# # Remove outliers by zscore
# rrf_df = rf_df[(np.abs(stats.zscore(df)) < 3*df.std()).all(axis=1)]
# sns.pairplot(rf_df)
# plt.show()

# # Imputation
# mdf = median_outlier_imputation(rf_df)
#
# sns.pairplot(mdf)
# plt.show()



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
X_train, X_test, y_train, y_test = train_test_split(rf_df.drop('Accretion Rate Shortterm', axis=1),
                                                    rf_df['Accretion Rate Shortterm'], test_size=.4, train_size=.6)
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=3,
                               random_state=42, n_jobs=-1, scoring='neg_mean_absolute_error')  #Fit the random search model
rf_random.fit(X_train, y_train)
pprint.pprint(rf_random.best_params_)

# Plot number of features VS. cross-validation scores
# Create a based model
# ...............

# grid search with cross validation
param_grid = {
    'bootstrap': [False],
    'max_depth': [40, 47, 54, 61, 68, 75, 82, 89, 96, 103, 110, None],
    'max_features': ['auto', 'sqrt'],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
}

rf_grid = RandomForestRegressor()  # Instantiate the grid search base model
grid_search = GridSearchCV(estimator=rf_grid, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=3, scoring='neg_mean_absolute_error',
                           refit=True)
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
best_grid = grid_search.best_estimator_
y_pred = best_grid.predict(X_test)
dump(best_grid, "D:\Etienne\crmsDATATables\ml_dump\RandForesttime_invar_terrebonne_SSML.joblib")

X = rf_df.drop('Accretion Rate Shortterm', axis=1)
y = rf_df['Accretion Rate Shortterm']
rf_tuned = load("D:\Etienne\crmsDATATables\ml_dump\RandForesttime_invar_terrebonne_SSML.joblib")
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
# ax.set_title(str(feature_names))
fig.show()


