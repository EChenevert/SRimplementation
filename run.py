import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

from gplearn.genetic import SymbolicRegressor


def feature_importance_select(predictor_df, outcome_str, thres=0.06):
    ''' Is a recursive feature elimination method. Based on a random forest model
    Help from: https://chrisalbon.com/code/machine_learning/trees_and_forests/feature_selection_using_random_forest/
    '''
    outcome = predictor_df[outcome_str]
    predictor_df = predictor_df.drop(outcome_str, axis=1)
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


def identify_vars(df, outcome_str, user_input=False):
    """
    :param outcome_str: a string that corresponds to a column in the inputted df
    :param df = a dataframe
    :param outcome = a column name in the df dataframe
    :param user_input = If FALSE (default), then
                        function computes the most important variables from the recursive feature selection method
                        Else, user should input a list of string names
    """
    if not user_input:
        user_input = feature_importance_select(df, outcome_str, thres=0.06)
    return user_input

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
    for col in df.columns.values:
        df[col] = df[col][(np.abs(stats.zscore(df[col])) < 3*df[col].std()).all(axis=1)]
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

    fig, ax = plt.figure()
    ax.scatter(y_test, y_pred)
    ax.set_ylabel('Predicted')
    ax.set_xlabel('Observed')
    ax.set_title('Random Forest Model')
    fig.show()

    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['RF'])



def symbolic_regression(X, y, feature_names, output="srU_1.png"):
    """ Creates a symbolic regression and returns the performance of the model AS WELL as the best fit equation"""
    y = X[y]
    X = X.drop(y, axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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
    ax.set_aspect('auto')  # can also be equal
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title(str(feature_names))
    fig.show()


    return pd.DataFrame.from_dict(model_performance, orient='index', columns=['SR']), equation, est_gp


def run(d_file, outcome_str,  output_graph_str):
    """
    :param d_file = file path name to csv file datset
    :param outcome = string that corresponds to a column in dataframe from csv dataset
    """

    df = pd.read_csv(d_file)
    id_vars = identify_vars(df, outcome_str)
    id_vars.append(outcome_str)  # add outcome var to the variables to extract from main dataframe
    ex_df = df[id_vars]  # extract from main dataframe
    no_df = deal_w_outliers(ex_df)
    drop_df = drop_nans(no_df)
    rf_metrics = random_forest(drop_df, outcome_str)
    sr_metrics, sr_equation, est_gp = symbolic_regression(drop_df, outcome_str, output_graph_str)
    return rf_metrics, sr_metrics, sr_equation, est_gp





if __name__ == '__main__':

    file = ''
    rf_metrics, sr_metrics, sr_equation, est_gp = run(file, "Average Accretion", "sr_plot1.png")


