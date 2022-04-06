import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from scipy import stats
import graphviz
from sympy import sin, cos, tan, log

from sympy import sympify
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import seaborn as sns

from gplearn.genetic import SymbolicRegressor
from scipy import stats


# Feature selection types
def correlation_coefficent_select(corr_df, outcome):

    '''Based on the pearson's correlation coefficents of certain variables with the outcome.
    outcome = string'''

    corr_filt = corr_df.loc[:, outcome][corr_df.iloc[:, 11] > np.abs(0.3)]
    return corr_filt

def plot_feat_imps(feats):
    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
    importances = importances.sort_values(by='Gini-Importance', ascending=False)
    importances = importances.reset_index()
    importances = importances.rename(columns={'index': 'Features'})
    sns.set(font_scale = 5)
    sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
    fig, ax = plt.subplots()
    fig.set_size_inches(30,15)
    sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
    plt.xlabel('Importance', fontsize=25, weight = 'bold')
    plt.ylabel('Features', fontsize=25, weight = 'bold')
    plt.title('Feature Importance', fontsize=25, weight = 'bold')

def mlregression(X, y):
    ''' Creates a simple multiple linear regression model'''
    scaler = MinMaxScaler()
    Xin = scaler.fit_transform(X)
    # Test train and split the data
    X_Train, X_Test, y_train, y_test = train_test_split(Xin, y, test_size=0.2, random_state=0)

    mlr = LinearRegression().fit(X_Train, y_train)

    y_pred = mlr.predict(X_Test)

    r, _ = pearsonr(np.squeeze(y_test), np.squeeze(y_pred))
    # r = np.corrcoef(y_test, y_pred)
    model_performance = {
        'Mean Absolute Error (cm):': metrics.mean_absolute_error(y_test, y_pred),
        # No dividing done because I use a StandardScalar which should make units "equal"
        'Mean Squared Error:': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (cm):': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        # No dividing done because I use a StandardScalar which should make units "equal"
        'R squared:': metrics.r2_score(y_test, y_pred),
        'Pearson Correlation Coefficient': r
    }
    # pd.DataFrame.from_dict(model_performance, orient='index', columns = 'MLR')



    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['MLR'])


def random_forest(X, y):
    '''This function will run a random forest machine learning algorithm on inputted
    data'''
    # X, y = self.set_up()
    scaler = MinMaxScaler()
    Xin = scaler.fit_transform(X)
    # Split the data into 20% test and 80% training
    X_train, X_test, y_train, y_test = train_test_split(Xin, y, test_size=0.2, random_state=0)
    # Create random forest regressor (fitting)
    regressor = RandomForestRegressor(n_estimators=2501, random_state=0, min_samples_leaf=10).fit(X_train, y_train)

    # Predict outcome from model
    y_pred = regressor.predict(X_test)

    feature_importance = pd.DataFrame(regressor.feature_importances_,
                                        columns=['importance']).sort_values('importance', ascending=False)

    r, _ = pearsonr(np.squeeze(y_test), np.squeeze(y_pred))
    # np.squeeze(y_test), np.squeeze(y_pred)
    model_performance = {
        'Mean Absolute Error (cm):': metrics.mean_absolute_error(y_test, y_pred),
        'Mean Squared Error:': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (cm):': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        'R squared:': metrics.r2_score(y_test, y_pred),
        'Pearson Correlation Coefficient': r
    }

    # plt.figure()
    # plt.scatter(y_test, y_pred)
    # plt.show()

    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['RF'])


def feature_importance_select(predictor_df, outcome, thres=0.06):
    ''' Is a recursive feature elimination method. Based on a random forest model
    Help from: https://chrisalbon.com/code/machine_learning/trees_and_forests/feature_selection_using_random_forest/
    '''
    # Hold the predictor names
    feat_labels = list(predictor_df.columns.values)
    # Replace all nans with the mean of the column

    # Scale the predictor values to be within 1 and 0
    scaler = MinMaxScaler()
    X = scaler.fit_transform(predictor_df)

    # Split the data (80_train/20_test)
    X_train, X_test, y_train, y_test = train_test_split(X, outcome, test_size=0.2, random_state=0)

    # Create a random forest classifier
    regressor = RandomForestRegressor(n_estimators=10000, random_state=0, n_jobs=-1)

    # Train the classifier
    regressor.fit(X_train, y_train)

    # Print the name and gini importance of each feature
    feature_importances = []
    for feature in zip(feat_labels, regressor.feature_importances_):
        feature_importances.append(feature)

    # return feature_importances

    # Create a selector object that will use the random forest classifier to identify
    # features that have an importance of more than 0.15
    sfm = SelectFromModel(regressor, threshold=thres)  # 0.05

    # Train the selector
    sfm.fit(X_train, y_train)

    important_names = []
    # Print the names of the most important features
    for feature_list_index in sfm.get_support(indices=True):
        important_names.append(feat_labels[feature_list_index])

    # plot_feat_imps(feature)

    return feature_importances, important_names


def symbolic_regression(X, y, feature_names, output="srU_1.png"):
    """ Creates a symbolic regression and returns the performance of the model AS WELL as the best fit equation"""
    # X, y = self.set_up()
    # scaler = MinMaxScaler()
    # Xin = scaler.fit_transform(X)

    # Xin = rm_outliers_zscore(Xin)  # This removes outliers greater than 3 standard deviations from the mean

    X_Train, X_Test, y_train, y_test = train_test_split(Xin, y, test_size=0.2, random_state=0)

    # Create the SR, the inputs are from the tutoriol on gplearn documentation website
    est_gp = SymbolicRegressor(population_size=5000,
                               n_jobs=-1,
                               const_range=(-5, 5),
                               # feature_names=feature_names,
                               # init_depth=(4, 10),
                               init_method='half and half',
                               function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'tan', 'log', 'abs', 'sqrt',
                                             'inv', 'neg'),
                               tournament_size=20,  # default value
                               generations=20,
                               stopping_criteria=0.01,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               # max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.001,
                               random_state=0,
                               metric='rmse')
    est_gp.fit(X_Train, y_train)
    y_pred = est_gp.predict(X_Test)

    converter = {
        'sub': lambda x, y: x - y,
        'div': lambda x, y: x / y,
        'mul': lambda x, y: x * y,
        'add': lambda x, y: x + y,
        'neg': lambda x: -x,
        'pow': lambda x, y: x ** y,
        'abs': lambda x: abs(x),
        'sin': lambda x: sin(x),
        'cos': lambda x: cos(x),
        'tan': lambda x: tan(x),
        'log': lambda x: log(x),
        'inv': lambda x: 1 / x,
        'sqrt': lambda x: x ** 0.5,
        'pow3': lambda x: x ** 3
    }

    # print(est_gp._program)
    equation = sympify((est_gp._program), locals=converter)
    r, _ = pearsonr(np.squeeze(y_test), np.squeeze(y_pred))
    model_performance = {
        'Mean Absolute Error (cm):': metrics.mean_absolute_error(y_test, y_pred),
        # No dividing done because I use a StandardScalar which should make units "equal"
        'Mean Squared Error:': metrics.mean_squared_error(y_test, y_pred),
        'Root Mean Squared Error (cm):': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
        # No dividing done because I use a StandardScalar which should make units "equal"
        'R squared:': metrics.r2_score(y_test, y_pred),
        'Pearson Correlation Coefficient': r
    }

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(str(feature_names))
    plt.show()
    plt.savefig(output)

    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['SR']), equation, est_gp


def pca(X, categorical_var):
    ''''''
    Xpca = X
    Xpca = Xpca.drop([categorical_var])

    p = PCA(n_components=np.shape(Xpca)[1])
    Xpca = Xpca.fillna(value=Xpca.mean())
    p.fit(Xpca)
    print('explained variance ratio: %s'
          % str(p.explained_variance_ratio_))
    plt.plot(np.r_[[0], np.cumsum(p.explained_variance_ratio_)])
    plt.xlim(0, 5)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')

    # part2

    y = X[categorical_var]
    cat_labs = X[categorical_var].unique()
    y = y.map({'High': 0, 'Nuetral': 1, 'Low': 2})

    X_r = PCA.transform(Xpca)
    plt.scatter(X_r[y == 0, 0], X_r[y == 0, 1], alpha=.8, color='blue',
                label='High')
    plt.scatter(X_r[y == 1, 0], X_r[y == 1, 1], alpha=.8, color='green',
                label='Nuetral')
    plt.scatter(X_r[y == 2, 0], X_r[y == 2, 1], alpha=.8, color='orange',
                label='Low')
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.xlabel('First component')
    plt.ylabel('Second component')
    plt.title('PCA of IRIS dataset')

def median_outlier_imputation(df):
    """Replace outliers with the median of the outliers in a dataset. Returns dataframe"""
    col_name = df.columns.values
    for i in range(len(col_name)):
        # sns.boxplot(df[col_name[i]])
        # plt.title('Box plot before outlie imputation: ' + col_name[i])
        # plt.show()
        for x in df[col_name[i]]:
            q1 = df[col_name[i]].quantile(0.25)
            q3 = df[col_name[i]].quantile(0.75)
            iqr = q3 - q1
            lower_tail = q1 - 1.5*iqr
            upper_tail = q3 + 1.5*iqr
            if x > upper_tail or i < lower_tail:
                df[col_name[i]] = df[col_name[i]].replace(i, np.median(df[col_name[i]]))
    return df

def rm_outliers_zscore(df):
    for col in df.columns.values:
        df[col] = df[col][(np.abs(stats.zscore(df[col])) < 3*df[col].std()).all(axis=1)]

    return df












if __name__ == '__main__':

    # Just using th eti and jank csv (bysite)
    ej_df = pd.read_csv(r"D:\Etienne\crmsDATATables\Small_bysite_byyear_season_FYA2.csv")
    my_imps = ['Soil Porewater Temperature (°C)',
               'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)', 'Staff Gauge (ft)', 'Flood Depth (mm)',
               'Accretion Rate']
    print(np.shape(ej_df[my_imps]))
    new = ej_df[my_imps]
    # new = rm_outliers_zscore(ej_df[my_imps])
    print(np.shape(new))
    new = new.dropna(subset='Staff Gauge (ft)')
    print(np.shape(new))

    # V = ej_df[[
    #     'Wet Soil pH (pH units)', 'Dry Soil pH (pH units)',
    #     'Soil Specific Conductance (uS/cm)', 'Soil Salinity (ppt)',
    #     'Soil Moisture Content (%)', 'Bulk Density (g/cm3)',
    #     'Organic Matter (%)', 'Wet Volume (cm3)', 'Dry Volume (cm3)',
    #     'Belowground Live Biomass (g/m2)',
    #     'Belowground Dead Biomass (g/m2)', 'Organic Density (g/cm3)',
    #     'Total Carbon (g/kg)', 'Carbon Density (mg/cm3)',
    #     'Total Nitrogen (g/kg)', 'Total Phosphorus (mg/kg)', 'Sand (%)',
    #     'Silt (%)', 'Clay (%)', 'Particle Size Mean (phi)',
    #     'Particle Size Median (phi)', 'Staff Gauge (ft)', 'Soil Porewater Temperature (°C)',
    #     'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)',  'Mud (%)', 'Flood Depth (mm)',
    #     'avg_percentflooded (%)', 'percent_waterlevel_complete', 'Distance from Water'
    # ]]
    # # V = V.dropna(subset='Staff Gauge (ft)')

    # scaler = MinMaxScaler()
    # new = scaler.fit_transform(new)
    print(new.std())
    new = rm_outliers_zscore(new)

    print(np.shape(new))
    V = median_outlier_imputation(new)

    y = new['Accretion Rate'].fillna(new['Accretion Rate'].mean())  # extract output var

    # V_new = V.fillna(V.mean())
    V = new.drop('Accretion Rate', axis=1)
    V_new = V.fillna(V.mean())
    # Feature importances from RF
    importances, imp_names = feature_importance_select(V_new, y)
    # My feature importances
    my_imps = ['Soil Porewater Temperature (°C)',
            'Soil Porewater Salinity (ppt)', 'Average Height Dominant (cm)', 'Staff Gauge (ft)', 'Flood Depth (mm)']
    # # Unsupervised selection Run symbolic regression
    # sr_u, eq_u = symbolic_regression(V_new[imp_names], y, imp_names)
    # # Run random forest model
    # rf_u = random_forest(V_new[imp_names], y)
    # # Run mlr
    # mlr_u = mlregression(V_new[imp_names], y)

    # with open('U_sr.txt', 'w') as f:
    #     f.write(eq_u)
    #     f.write('\n')
    #     f.write(sr_u)

    use = V[my_imps].dropna()
    # Supervise: Run symbolic regression
    sr_s, eq_s, srreg = symbolic_regression(use, y, my_imps, "srS_1.png")
    # Run random forest model
    rf_s = random_forest(use, y)
    # run mlr
    mlr_s = mlregression(use, y)

    # Plot tree
    dot_data = srreg._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph




    # V = X[(np.abs(stats.zscore(X)) < 10).all(axis=1)]  # drop any points greater than 3 standard deviations from mean

    # Data Transformations

#     V = ej_df[[
#         'Distance from Water', 'Average Flood Depth (mm)', 'Organic Density (g/cm3)',
#         'Average Flood Depth Jank (mm)', 'Average Height Dominant (cm)', 'Soil Porewater Temperature (°C)',
#         'Soil Porewater Salinity (ppt)', 'Staff Gauge (ft)', 'Soil Moisture Content (%)', 'Bulk Density (g/cm3)',
#        'Organic Matter (%)', 'Average Height Dominant (cm)'
#     ]]

#     imputationdf = median_outlier_imputation(V)
#     imputationdf = imputationdf.fillna(imputationdf.mean())
#     # Test feature importances
#     importances, imp_names = feature_importance_select(imputationdf, y)
#     # imp_names = ['Organic Matter (%)', 'Distance from Water',
#     #               'Soil Porewater Temperature (°C)', 'Average Flood Depth (mm)']
#     # plot ideal
# # Longitude being highlighted as an important feature likely means we need to split the data between western coast and
# # eastern LA coast
# # 'Oberservation Length (ft) also being highlighted as an important feature may be highlighting a sampling bias in the data
#     # Make dataset of selected important features
#     Ximp = imputationdf[imp_names]
#     # Run symbolic regression
#     sr, eq = symbolic_regression(Ximp, y)
#     # Run random forest model
#     # Ximp_rf = V[imp_names]
#     rf = random_forest(Ximp, y)





    # # ej_df = pd.read_csv(r"D:\Etienne\crmsDATATables\Average_byyear_bysite.csv")  # average_bysite_byyear of ALL crms
    # # ej_df = pd.read_csv(r"D:\Etienne\crmsDATATables\Average_byyear_bysite_EAST.csv")  # Average_bysite_byyear of only EASTERN crms sites
    # ej_df = pd.read_csv(r"D:\Etienne\crmsDATATables\Average_bymonth_bysite_EAST.csv")
    # ej_df = ej_df.dropna(subset=['Average Accretion (mm)'])
    # y = ej_df['Average Accretion (mm)']
    # X = ej_df.drop(['Simple site',
    #                 'Particle Size Standard Deviation (phi)',
    #                 'Measurement Depth (ft)',
    #                 'Longitude',
    #                 'Turbidity (FNU)', 'Chlorophyll a (ug/L)',
    #                 'Total Nitrogen (mg/L)', 'Total Kjeldahl Nitrogen (mg/L)',
    #                 'Nitrate as N (mg/L)', 'Nitrite as N (mg/L)',
    #                 'Nitrate+Nitrite as N (unfiltered; mg/L)',
    #                 'Nitrate+Nitrite as N (filtered; mg/L)',
    #                 'Ammonium as N (unfiltered; mg/L)', 'Ammonium as N (filtered; mg/L)',
    #                 'Total Phosphorus (mg/L)', 'Orthophosphate as P (unfiltered; mg/L)',
    #                 'Orthophosphate as P (filtered; mg/L)', 'Silica (unfiltered; mg/L)',
    #                 'Silica (filtered; mg/L)', 'Total Suspended Solids (mg/L)',
    #                 'Volatile Suspended Solids (mg/L)', 'Secchi (ft)',
    #                 'Fecal Coliform (MPN/100ml)', 'pH (pH units)', 'Velocity (ft/sec)',
    #                 'Radiometric Dating Method and Units',
    #                 'Isotope Concentration', 'Latitude',
    #                 'Accretion Measurement 1 (mm)',
    #                 'Accretion Measurement 2 (mm)', 'Accretion Measurement 3 (mm)',
    #                 'Accretion Measurement 4 (mm)', 'Direction (Collar Number)',
    #                 'Direction (Compass Degrees)', 'Pin Number',
    #                 'Observed Pin Height (mm)', 'Verified Pin Height (mm)',
    #                 'Average Accretion (mm)'], axis=1)
    # X_rf = X.fillna(X.median())
    # # Test feature importances
    #
    # # importances, imp_names = feature_importance_select(X_rf, y)
    #
    # # plot ideal
    # # Longitude being highlighted as an important feature likely means we need to split the data between western coast and
    # # eastern LA coast
    # # 'Oberservation Length (ft) also being highlighted as an important feature may be highlighting a sampling bias in the data
    # # Make dataset of selected important features
    # # X_dropped = X.dropna()
    # # Ximp = X_dropped[imp_names]
    #
    # # Ximp = X_rf[imp_names]
    #
    # # Run symbolic regression
    # # sr, eq = symbolic_regression(Ximp, y)
    # # Run random forest model
    #
    # # rf = random_forest(Ximp, y)
    #
    # # Run a multiple linear regression
    # # mlr = mlregression(Ximp, y)
    #

    #
    # big = ej_df
    #
    # s = ej_df[['Month (mm)', 'Bulk Density (g/cm3)', 'Wet Volume (cm3)', 'Simple site']]
    # gb = s.groupby(['Simple site']).mean()
    # # siteidx = s.index.get_level_values(1)
    # # monidx = s.index.get_level_values(0)
    # # condition = [big['Month (mm)'] == monidx[i] and big['Simple site'] == siteidx[i] for i in range(len(siteidx))]
    # big.set_index(['Simple site'], inplace=True)
    #
    # # test = big.merge(gb, how="left", left_on="Modular Index", right_on="Index")
    # test = big.join(gb, how='left', lsuffix='_left', rsuffix='_right')
    #




