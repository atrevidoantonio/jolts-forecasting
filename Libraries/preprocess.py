import logging
log = logging.getLogger(__name__)

import pandas as pd
import numpy as np

def create_all_time_features(df, feature_cols, date_col = 'Date', target_col = 'Value', is_larger_than_one_year = False, drop_first = True):
    """
    Description: Create all time features from the date column in the input pandas dataframe.
    If there is more than a year's of data, create month features base on date column.

    Params:
        - df: pandas dataframe, input pandas dataframe
        - feature_cols: string list, features from model configuration file.
        - modeling_class: string, ML model name.
        - date_col: string, date column name.
        - target_col: string, target column name.
        - is_larger_than_one_year: boolean, if true, create month features based on the date column.
        - drop_first: boolean, if true then the first dummy variable for categorical fatures is dropped.
    Return:
        - df: pandas dataframe, input pandas dataframe for model training or forecast.
    """    

    feature_categories = {}
    feature_categories['month'] = {"zero_indexed": False, "categories": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
    final_feature_cols = []

    if not is_larger_than_one_year and 'month' in feature_cols:
        feature_cols.remove('month')

    for feature in feature_cols:
        if feature == 'day_of_month':
            df.loc[:,'day'] = df[date_col].dt.day
            final_feature_cols.append('day')
        elif feature in feature_categories.keys():
            dummies, new_feature_cols = _get_dummy_date_feature(df[date_col], feature_categories[feature]["categories"], feature, 
                                                                zero_indexed = feature_categories[feature]["zero_indexed"],
                                                                drop_first = drop_first)
            df =  pd.concat([df, dummies], axis = 1).reindex(dummies.index)
            final_feature_cols.extend(new_feature_cols)
        else:
            log.debug("Skipping unkown feature '{}'.".format(feature))
            continue

    final_feature_cols.append(date_col)
    if target_col in df.columns:
        final_feature_cols.append(target_col)
    df = df[final_feature_cols]
    return df

def _get_dummy_date_feature(date_series, feature_categories, feature_name, zero_indexed = True, drop_first = True):
    """
    Description: Transforms a data series into dummy variables for a datetime attribute such as 'month'.
    Params:
        - date_series: A Pandas Series of type DateTime
        - feature_categories: A list of the categorical values for the date feature.
        - feature_name: Name of the date feature. Assumes that the the feature_name exists as an attribute
                        of date_series.dt.
        - drop_first: Determines if the first level, sorted alphabetically, of a categorical variable should be dropped. Defaults True.
    Return: DataFrame and Column Names of Dummy Variables. 
    """

    offset = 0 if zero_indexed else 1
    feature_dict = dict((key + offset, value) for (key, value) in enumerate(feature_categories))
    date_feature = getattr(date_series.dt, feature_name.replace('_', ''))
    categorical_date_feature = pd.Categorical(date_feature.map(feature_dict))
    dummies = pd.get_dummies(categorical_date_feature, drop_first = drop_first)
    return dummies, dummies.columns

def check_larger_than_one_year(df, date_col, interval):
    """
    Description: function to return a Boolean object for the span of the date column larger than 12 month or not
    Params:
        - df: a pandas dataframe
        - date_col: the name of the date column
    Return: a Boolean object
    """
    df.loc[:,date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
    min_date = min(df[date_col])
    max_date = max(df[date_col]) + pd.Timedelta('1 days')
    diff_month = (max_date.year - min_date.year)* 12 + max_date.month - min_date.month
    if diff_month < 11 and interval == "Monthly":
        return False
    else:
        return True