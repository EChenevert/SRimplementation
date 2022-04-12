import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from scipy import stats
from scipy.stats import pearsonr

import graphviz
from sympy import sin, cos, tan, log
from sympy import sympify

from gplearn.genetic import SymbolicRegressor, SymbolicTransformer

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import geostatspy.geostats as geostat

def declustering_geospatial():
    """

    :return:
    """




def feature_importance_select(predictor_df, outcome_str, thres=0.06):
    ''' Is a recursive feature elimination method. Based on a random forest model
    Help from: https://chrisalbon.com/code/machine_learning/trees_and_forests/feature_selection_using_random_forest/
    '''
    outcome = predictor_df[outcome_str]
    predictor_df = predictor_df.drop(outcome_str, axis=1)
    # drop categorical variables for regression
    if "Basins" in predictor_df.columns.values:
        predictor_df = predictor_df.drop("Basins", axis=1)
    if "Simple site" in predictor_df.columns.values:
        predictor_df = predictor_df.drop("Simple site", axis=1)
    if "Community" in predictor_df.columns.values:
        predictor_df = predictor_df.drop("Community", axis=1)
    # Hold the predictor names
    feat_labels = list(predictor_df.columns.values)
    # Drop columns with all nan values
    predictor_df = predictor_df.dropna(axis=1, how='all')
    # Replace all nans with the mean of the column
    predictor_df = predictor_df.fillna(predictor_df.median())
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


def identify_vars_rf(df, outcome_str, user_input=False):
    """
    :param outcome_str: a string that corresponds to a column in the inputted df
    :param df = a dataframe
    :param outcome = a column name in the df dataframe
    :param user_input = If FALSE (default), then
                        function computes the most important variables from the recursive feature selection method
                        Else, user should input a list of string names
    """
    if not user_input:
        _, important_features = feature_importance_select(df, outcome_str, thres=0.06)
    # if user_input:
    #     user_input = user_input
    # if user_input:
    #     important_features = input("Input a variable names separated by one space: ").split()
    return important_features


def median_outlier_imputation(df):
    """Replace outliers with the median of the outliers in a dataset. Returns dataframe"""
    col_name = df.columns.values
    for i in range(len(col_name)):
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
    # for col in df.columns.values:
    df = df[(np.abs(stats.zscore(df)) < 3*df.std()).all(axis=1)]
    return df


def rm_outliers_interquartile(df):
    """

    :param df:
    :return:
    """
    # get the range
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    # remove and make new dataset
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df




def deal_w_outliers(df, user_input=True):
    """
    Function that interacts with the user to ask how to deal with outliers. Remember that if user_input != True,
    then outliers will be dealt with imputation method (replacing outliers with the median of outliers
    :param df: a dataframe
    :param user_input: default = FALSE, if not, should by a string as specified in prompt
    :return: a dataframe cleared of outliers
    """
    if user_input:
        user_input = input('Enter option to deal with outliers: imputation, zscore, none, or both')
        if user_input == 'imputation':
            df = median_outlier_imputation(df)
        elif user_input == 'zscore':
            df = rm_outliers_zscore(df)
        elif user_input == 'none':
            pass
        elif user_input == 'both':
            df = median_outlier_imputation(df)
            df = rm_outliers_zscore(df)
        else:
            print('improper input. Type the words: imputation, zscore, none, or both')
    else:
        df = rm_outliers_interquartile(df)
        df = median_outlier_imputation(df)
    return df

def drop_nans(df, user_input=True):
    """

    :param df: inputted dataframe
    :param user_input: interacts with the user to ask how to drop nans in current dataset
    :return: a dataframe with nans dropped or dropped and filled with median
    """

    if user_input:
        user_input = input("Deal with left over nans in the dataset, enter: dropna or dropna subset")
        if user_input == 'dropna':
            df = df.dropna()
        elif user_input == 'dropna subset':
            subset = input("Input the subset to drop nans by:")
            while subset not in df.columns.values:
                print("This is an invalied subset")
                subset = input("Input the subset to drop nans by:")
            df = df.dropna(subset=subset)
            df.fillna(df.median())
    if not user_input:
        df = df.dropna()

    return df


def random_forest(X, outcome_str):
    '''This function will run a random forest machine learning algorithm on inputted
    data'''
    y = X[outcome_str]
    X = X.drop(outcome_str, axis=1)
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

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_ylabel('Predicted')
    ax.set_xlabel('Observed')
    ax.set_title('Random Forest Model')
    fig.show()

    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['RF'])


def symbolic_transformer(X, y_str, feature_names, output="srU_1.png"):
    """

    :param X:
    :param y_str:
    :param feature_names:
    :param output:
    :return:
    """
    y = X[y_str]
    X = X.drop(y_str, axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create the SR, the inputs are from the tutoriol on gplearn documentation website
    function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
    gp = SymbolicTransformer(generations=20, population_size=2000,
                             hall_of_fame=100, n_components=10,
                             function_set=function_set,
                             parsimony_coefficient=0.0005,
                             max_samples=0.9, verbose=1,
                             random_state=0, n_jobs=3)
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
    ax.set_aspect('equal')  # can also be equal
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(str(feature_names))
    fig.show()


    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['SR']), equation, est_gp


def symbolic_regression(X, y_str, feature_names, output="srU_1.png"):
    """ Creates a symbolic regression and returns the performance of the model AS WELL as the best fit equation"""
    y = X[y_str]
    X = X.drop(y_str, axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create the SR, the inputs are from the tutoriol on gplearn documentation website
    est_gp = SymbolicRegressor(population_size=5000,
                               n_jobs=-1,
                               const_range=(-5, 5),
                               # feature_names=feature_names,
                               # init_depth=(4, 10),
                               init_method='full',
                               function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos', 'tan', 'log', 'abs', 'sqrt',
                                             'inv', 'neg'),
                               tournament_size=1,  # default value = 20
                               generations=150,
                               stopping_criteria=0.01,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               # max_samples=0.66,
                               verbose=1,
                               parsimony_coefficient=0.01,
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
    ax.set_aspect('equal')  # can also be equal
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(str(feature_names))
    fig.show()


    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['SR']), equation, est_gp


# def symbolic_transformer()

def declustering():
    """

    :return: WEights to attach to
    """

def run(df_file, outcome_str,  output_graph_str, user_vars=False, user_ls=None):
    """
    :param d_file = file path name to csv file datset
    :param outcome = string that corresponds to a column in dataframe from csv dataset
    """

    df = pd.read_csv(df_file)
    if not user_vars:
        id_vars = identify_vars_rf(df, outcome_str, False)
    if user_vars:
        id_vars = user_ls
    id_vars.append(outcome_str)  # add outcome var to the variables to extract from main dataframe
    ex_df = df[id_vars]  # extract from main dataframe
    drop_df = drop_nans(ex_df, False)
    no_df = deal_w_outliers(drop_df, False)

    rf_metrics = random_forest(no_df, outcome_str)
    sr_metrics, sr_equation, est_gp = symbolic_regression(drop_df, outcome_str, output_graph_str)
    return rf_metrics, sr_metrics, sr_equation, est_gp, id_vars, drop_df





if __name__ == '__main__':

    file = r"D:\Etienne\crmsDATATables\Average_bysite.csv"
    var_list = ["Soil Porewater Temperature (°C)",
                "Average Height Dominant (cm)", "Organic Density (g/cm3)", "Staff Gauge (ft)",
                "Soil Porewater Salinity (ppt)"]
    df = pd.read_csv(file)
    df = df[var_list]
    rf_metrics, sr_metrics, sr_equation, est_gp, important_vars, sr_df = run(file, "Average Accretion (mm)", "sr_plot1.png",
                                                                             True, var_list)

    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    # graph

    for col in sr_df.columns.values:
        plt.figure()
        sns.histplot(sr_df[col])
        plt.xlabel(col)
        plt.show()
    # Save model outputs to a file

