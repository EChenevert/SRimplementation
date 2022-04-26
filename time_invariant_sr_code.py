from gplearn.genetic import SymbolicRegressor
from joblib import dump, load # model persistence
import pprint # Nice printing
from math import sin, cos, tan, log

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv(r"D:\Etienne\crmsDATATables\basins_time_invariant\Terrebonne.csv", encoding="unicode escape")
# Drop columns that have more than 75 % nan values
ddf = df.dropna(thresh=df.shape[0]*0.75, how='all', axis=1)

# Append the sought outcome variable to the list
outcome_var_str = "Average Accretion (mm)"

# Manually define variable sthat should be important
drop_vars = [
    'Simple site', 'Unnamed: 0', 'level_1', 'Season', 'Month (mm)', 'Surface Elevation Change Rate Shortterm',
    'Accretion Measurement 1 (mm)', 'Accretion Measurement 2 (mm)',
    'Accretion Measurement 3 (mm)', 'Accretion Measurement 4 (mm)',
    'Accretion Rate Shortterm', 'Observed Pin Height (mm)', 'Verified Pin Height (mm)',
    'Latitude', 'Longitude', 'Direction (Collar Number)',
    'Direction (Compass Degrees)', 'Pin Number', 'Measurement Depth (ft)', 'percent_waterlevel_complete',
    'month', 'Staff Gauge (ft)'
]

dddf = ddf.drop(drop_vars, axis=1).dropna()

# less variables is better for the genetic algo
imp_vars = [
         'Average Height Herb (cm)', 'avg_percentflooded (%)',
         'Flood Depth (mm)', 'Distance from Water', 'Average Accretion (mm)']

sr_df = dddf[imp_vars]
sns.pairplot(sr_df)
plt.show()

# Transformations
# log transforms
sr_df['Distance from Water'] = [np.log(i) if i != 0 else 0 for i in sr_df['Distance from Water']]
sr_df['Flood Depth (mm)'] = [np.log(i) if i != 0 else 0 for i in sr_df['Flood Depth (mm)']]
sns.pairplot(sr_df)
plt.show()

# drop outliers by zscore
from scipy import stats
dmdf = sr_df.dropna()
# zmdf = mdf[(np.abs(stats.zscore(mdf)) < 3).all(axis=1)]

for col in dmdf.columns.values:
    print(np.shape(dmdf))
    # dmdf[col+"_z"] = dmdf[col].apply(stats.zscore)
    dmdf[col+"_z"] = stats.zscore(dmdf[col])
for col in dmdf.columns.values[7:]:
    dmdf = dmdf[np.abs(dmdf[col]) < 3]  # keep if value is less than 2 std

# drop zscore columns
dmdf = dmdf.drop([
    'Average Height Herb (cm)_z',
    'avg_percentflooded (%)_z', 'Flood Depth (mm)_z',
    'Distance from Water_z', 'Average Accretion (mm)_z'
], axis=1)

sns.pairplot(dmdf)
plt.show()

# ...........................................
# Standardizing
y = dmdf[outcome_var_str]
X = dmdf.drop(outcome_var_str, axis=1)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into train test and split segments
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, test_size=.4, train_size=.6)  # this ratio cuz dataset is small



# Hyperparameter tuning
# This work flow of hyperparamter tuning is based on methods from https://www.nature.com/articles/s41467-020-17263-9
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

# Make custom functions to input into symbolic regressor function set
from gplearn.functions import make_function
def pow_3(x1):
    f = x1**3
    return f
pow_3 = make_function(function=pow_3,name='pow3',arity=1)

def pow_2(x1):
    f = x1**2
    return f
pow_2 = make_function(function=pow_2,name='pow2',arity=1)


# add the new function to the function_set
function_set = ['add', 'sub', 'mul', 'div', pow_2, pow_3]

# initialize symbolicregressor
est_gp = SymbolicRegressor(population_size=5000,
                           n_jobs=-1,
                           const_range=(-1, 1),
                           # feature_names=feature_names,
                           # init_depth=(4, 10),
                           init_method='half and half',
                           function_set=function_set,
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


# simple y uncomment below Code after the above runs



# .................. Above this took a long time so do not re-run until later

# using what i learned from the random search to do a grid search with cross validation
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search
param_grid = {'p_crossover': [0.6250000000000001],
 'p_hoist_mutation': [0.07499999999999991],
 'p_point_mutation': [0.10833333333333328],
 'p_subtree_mutation': [0.06666666666666658],
 'parsimony_coefficient': [0.001]}
# Create a based model
est_gp_grid = SymbolicRegressor(population_size=5000,
                           n_jobs=-1,
                           const_range=(-1, 1),
                           init_method='half and half',
                           function_set=function_set,
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
dump(best_grid, "D:\Etienne\crmsDATATables\ml_dump\SRTunedGS_terrebonne_newfuncs_SSML.joblib")



# ----------------------
sr_tuned = load("D:\Etienne\crmsDATATables\ml_dump\SRTunedGS_terrebonne_newfuncs_SSML.joblib")
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

# ------ get equation
from sympy import *

converter = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    # 'neg': lambda x    : -x,
    # 'pow': lambda x, y : x**y,
    # 'sin': lambda x    : sin(x),
    # 'cos': lambda x    : cos(x),
    # 'inv': lambda x: 1/x,
    'sqrt': lambda x: x**0.5,
    'pow2': lambda x: x**2,
    'pow3': lambda x: x**3
}

eq = sympify((sr_tuned._program), locals=converter)
# Get best programs
# best_programs = sr_tuned._best_programs


# Ideas:
# what about an exponential function ? np.exp(x)
#

