import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


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


def identify_vars(df, outcome, user_input=False):
    """
    :param df = a dataframe
    :param outcome = a column name in the df dataframe
    :param user_input = If FALSE (default), then
                        function computes the most important variables from the recursive feature selection method
                        Else, user should input a list of string names
    """
    if not user_input:
        user_input = feature_importance_select(df, outcome, thres=0.06)
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

def deal_w_outliers(df, user_input=False):
    """
    :param user_input: default = FALSE, if not, should by a string as specified in prompt
    :return: a dataframe cleared of outliers
    """
    if not False:
        user_input = input('Enter: imputation, zscore, none, or both')
    if user_input == 'imputation':
       df = median_outlier_imputation(df)
    elif user_input == 'zscore':
        df =



def run(d_file, outcome):
    """
    :param d_file = file path name to csv file datset
    :param outcome = string that corresponds to a column in dataframe from csv dataset
    """

    df = pd.read_csv(d_file)
    id_vars = identify_vars(df, outcome)
    id_vars.append(outcome)  # add outcome var to the variables to extract from main dataframe
    ex_df = df[id_vars]  # extract from main dataframe






if __name__ == '__main__':

    file = ''
    run(file)


