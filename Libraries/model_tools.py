import logging
log = logging.getLogger(__name__)
import warnings
import sys
import re
import os
import pickle
import math
import random

from pandas import Series
import pandas as pd
import numpy as np
from numpy import linalg
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import ARResults
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

import libraries.metrics as met
import libraries.models as md
import libraries.preprocess as pp

# Initialize global variables
DATE_COL_NAME = "Date"

class model_input():
    """ This class is used to generate models given some "modeling_class" (which is a model instance) and dataset that has a date column. It has functions to perform CV and CV w/grid search depending on whether you specify a grid.
        You must pass the dataset to the constructor and it must have a date column.
    """

    def __init__(self, dataset, modeling_class=None, grid=None, scoring_metric=['MSE'], n_jobs=1, n_splits=0, window_size=None, horizon=0.3, configs=None, return_cvtrain_score = True, Interval = None, drop_first = False, features = None):
        """ Class Constructor

            Args
            :param dataset: Dataframe that must have date and target as columns.
            :param modeling_class: Instance of model object.
            :param grid: Grid of parameters to optimize over.
            :param scoring_metric: Scoring function to use in grid_search. See met.scorers().get_scorer_list() to see full available list.
            :param njobs: Number of parallel jobs to run in grid search.
            :param n_splits: Number of folds. 1 is the equivalent to hold-out.
            :param window_size: Number of units in training set. Defaults to max. Units will be taken to be the sampling frequency 
            :param horizon (int or float): If int, it's the number of units to forecast in test set at each fold. Similar unit limitation as in window_size.
                                           If under 1 (decimal), taken as percent (pct) of training data. If window_size is specified and horizon is a pct then will be taken as a pct of the window_size.
            :param configs (dict): dictionary of configuration settings. if provided, for any attribute matching a key, the value of the attribute will be changed to the value of key. For example if the dict contains {'njobs':1} then the value of njobs shall be updated.
            :param return_train_score(boolean): Boolean to control returning training score or not.
            :param Interval (str): if 'Daily' then data comes in on daily volume; if "Monthly", then data comes in on monthly volume.
            :param drop_first (boolean): if True then remove the first level of the categorical level.
            :return: Instance of model_factory class.


            Fields (not in constructor):
            dates: a dataframe with the dates in the dataset having the same index as the dataframe
            last_date_in_train: it's the last date in the training set
            tscv: is an instance of met.time_series_split object in validation module. This will be used to split the indices into training/test during CV
            scorer: the actual scoring object passed into sklearn interfaces for CV/GridSearchCV
            X: X passed into .fit() 
            Y: Y passed into .predict()

            *** TO-DO:
            - General exception handling on bad inputs.
        """

        # Init parameterized and configurable attributes
        self.modeling_class = modeling_class
        self.grid=grid
        self.scoring_metric=scoring_metric
        self.n_jobs=n_jobs
        self.n_splits=n_splits
        self.window_size=window_size
        self.horizon=horizon        
        self.return_cvtrain_score = return_cvtrain_score
        self.Interval = Interval
        self.drop_first = drop_first
        self.features = features

        # Override attributes above according to the configuration
        if configs is not None:
            self.unpack(configs)   
        
        # Init Static Attributes that are for internal use and not intended to be configurable     
        self.error_score= np.nan
        self.date_col=DATE_COL_NAME
        self.target_col="Value"
        self.best_params = None

        # Init dependant attributes
        self.is_larger_than_one_year = pp.check_larger_than_one_year(dataset, self.date_col, self.Interval)
        self.dataset = dataset
        self.dataset.loc[:, DATE_COL_NAME] = pd.to_datetime(self.dataset[DATE_COL_NAME], format='%Y%m%d')
        self.dataset=pp.create_all_time_features(df = self.dataset[[self.date_col,self.target_col]], feature_cols = self.features, date_col = self.date_col, target_col = self.target_col, is_larger_than_one_year = self.is_larger_than_one_year, drop_first = self.drop_first)
        self.update_scorer()
        self.dates=pd.DataFrame({self.date_col:dataset[self.date_col]}, index=dataset.index)
        self.last_date_in_train = max(self.dates.values)[0]
        self.tscv = met.time_series_split(n_splits=self.n_splits, window_size=self.window_size, horizon=self.horizon, date_col=self.date_col, interval = self.Interval)
        self.tscv.split(self.dates)
        self.X = self.dataset.drop([self.target_col], axis=1, inplace=False)
        self.Y = self.dataset[self.target_col]

        # Update grid search pool of seasonal_orde
        if not self.is_larger_than_one_year and "seasonal_order" in self.grid:
            self.grid["seasonal_order"] = self._remove_yearly_seasonal_order(self.grid["seasonal_order"])

        if "seasonal_order" in self.grid is not None:
            self.grid["seasonal_order"] = self._update_seasonal_order(self.grid["seasonal_order"])

    def grid_search(self, modeling_class=None, grid=None, dataset=None):
            """ Grid search function. Called in generate_model but can work stand-alone.  Returns grid-search-cv-object.
            Args:
            :param modeling_class: You can specify a new modeling_class to assign to this object.
            :param grid: You can specify a new grid to assign to this object.
            :param dataset: You can specify a new dataset to assign to this object.
            :return: Returns the grid search output from GridSearchCV
            """
        # Adjust attributes if optional parameters passed in.
            if modeling_class is not None:
                self.modeling_class = modeling_class
            if grid is not None:
                self.grid = grid
            self.debug_log()

        # Set object and run grid search.
            tuned_model = GridSearchCV(self.modeling_class, scoring=self.metric, param_grid=self.grid, cv=self.tscv.get_splits(), n_jobs=self.n_jobs, error_score=self.error_score, return_train_score=self.return_cvtrain_score,refit=self.refit)
            tuned_model.fit(self.X, self.Y)

        # Update attributes given grid search results.
            self.tuned_model = tuned_model
            self.model = tuned_model.best_estimator_
            self.best_score = {self.refit: -tuned_model.best_score_}
            self.best_params = tuned_model.best_params_
            self.cv_results = tuned_model.cv_results_
        
            return(self.tuned_model)

    def _is_grid_point(self):
        """ Tests if the grid is a single point.

        Args:
        :param grid: Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.

        """
        return all([len(value) == 1 for value in self.grid.values()])

    def generate_model(self, modeling_class=None, grid=None, dataset=None):
        """ This function generates a model given the saved dataset. If a grid is passed in the Constructor
            then it will use the grid_search method. Otherwise will fit modeling class on data and return fit model
            using default model hyperparameters.

            Args:
            :param modeling_class: You can specify a new modeling_class to assign to this object.
            :param grid: You can specify a new grid to assign to this object.
            :param dataset: You can specify a new dataset to assign to this object.

            :return: Returns the fit model.
        """
        # Adjust attributes if optional parameters passed in.
        if modeling_class is not None:
            self.modeling_class = modeling_class
        if dataset is not None:
            self.update_data(dataset)
        if grid is not None:
            self.grid = grid
        # If there is a grid then do grid search, else, just do CV to get best_score then fit and return the model.
        if self.grid is not None and not self._is_grid_point():
            self.grid_search()
        else:
            if self.grid is not None:
                # Overwrites the parameters of the modeling class with the single point of values from the grid.
                # An alternative solution could be to use the fit_params parameter.
                self.modeling_class = _reconfigure_class(self.modeling_class, {key: val[0] for (key, val) in self.grid.items()})
            best_score = cross_validate(self.modeling_class,
                                        self.X, y=self.Y, groups=None,
                                        scoring=self.scorer, cv=self.tscv.get_splits(),
                                        n_jobs=1, verbose=0,
                                        fit_params=None,
                                        return_train_score=self.return_cvtrain_score)
            self.cv_results = best_score
            self.model=self.modeling_class
            self.model.fit(self.X, self.Y)
            
        log.info("Model has been fit in model factory.")

        return(self.model)

    def get_best_model(self):
        """ Function to return model """
        return(self.model)

    def unpack(self, configs):
        self = _reconfigure_class(self, configs)

    def update_scorer(self):
        """ A function to get metric values from the validation object.
        """
        self.scorer={}
        for score in self.scoring_metric:
            self.scorer[score] = met.scorers().get_make_scorer(score)
        refit = self.scoring_metric[0]
        if self.scorer[refit].valid_optimizer:
            self.refit = refit
        else:
            raise ValueError('The first validation metric configured, "{}", is not a valid metric for optimization.'.format(refit))

    
    def summarize(self):
        """Summary function to report the stats of the models.
           Each scorer configured will have three columns
           1. CV_test_<scorename>: The average score of the Cross Validation folds measured against the test subset.
           2. CV_train_<scorename>:The average score of the Cross Validation folds measured against the train subset.
           3. Full_train_<scorename>: The score measured against the full training dataset.
        """
        best_params_str = ""
        if self.best_params is None:
            # For example, best_params will be none when modeling autoarima because autoarima does its own internal 
            # grid search of the hyperparamter space.
            best_params_str = "Unknown"
        else:
            for k, v in self.best_params.items():
                best_params_str = best_params_str + str(k) + "=" + str(v) + "; "
        summary = pd.Series()
        summary['Params'] = best_params_str
        summary['Error_Message'] = "None"
        split_types = []
        if self.n_splits >= 1: 
            split_types.append('test')
            if self.return_cvtrain_score:
                split_types.append("train")

        if "rank_test_" + self.refit in self.cv_results.keys():
            ranks = self.cv_results["rank_test_"+self.refit]==1
        else:
            ranks = [True]

        for metric in self.scoring_metric:
            # If there is Error_Message then the metrics should be all NULL
            if summary['Error_Message'] != "None":
                full_metric_res = None

            metric_func = self.scorer[metric]
            full_metric_res = metric_func(self.model, X = self.X, y_true = self.Y)
            # Greater_is_better = False then the metric result will have a flipped sign thanks to the make scorer function
            if not metric_func.greater_is_better:
                full_metric_res = -full_metric_res
            for split_type in split_types:
                if self.grid is not None and not self._is_grid_point():
                    cv_metric_res = self.cv_results["mean_" + split_type + '_' +metric][ranks][0]
                else:
                    cv_metric_res = self.cv_results[split_type + '_' +metric][ranks][0]

                # Greater_is_better = False then the metric result will have a flipped sign due to the make scorer function
                if not metric_func.greater_is_better:
                    cv_metric_res = -cv_metric_res

                if metric == 'mean_squared_error' and cv_metric_res>10000000:
                    log.info("Exploding mean squared error: {}.".format(cv_metric_res))
                if metric == 'weighted_absolute_percent_error' and cv_metric_res>1000:
                    log.info("Exploding weighted absolute percent error: {}.".format(cv_metric_res))
                if metric == 'RMSE' and cv_metric_res>10000:
                    log.info("Exploding RMSE: {}.".format(cv_metric_res))
                if metric == 'CVRMSE' and cv_metric_res>1000:
                    log.info("Exploding CVRMSE: {}.".format(cv_metric_res))

                if np.isinf(cv_metric_res):
                    cv_metric_res = None
                summary['CV_' + split_type + '_' + metric] = cv_metric_res

            if np.isinf(full_metric_res):
                full_metric_res = None
            summary['Full_train_' + metric] = full_metric_res
        return summary

    def zero_summary(self, error_message = "Failure Unknown"):
        """A function to return an empty summary table if any unknow failure encountered in modeling and forecasting and log the error message.
            :param string error_message: the error message from modeling or forecasting.
            :return series d: a series contains nothing for each attribute except for error message.
        """
        d = pd.Series()
        d['Params'] = "None"
        d['Error_Message'] = error_message
        split_types = ['test']
        if self.return_cvtrain_score:
          split_types.append('train')
        for metric in self.scoring_metric:
            d['Full_train_' + metric] = None
            for split_type in split_types:
                d['CV_' + split_type + '_' + metric] = None
        return d

    def debug_log(self):
        """A function to validate the input dataset with correct headers.
        """
        log.debug("Head of dataframe X: ")
        log.debug(self.X.head().to_string())
        log.debug("Head of dataframe Y: ")
        log.debug(pd.DataFrame(self.Y.head()).to_string())
        log.debug("Model: {}, Grid: {}".format(self.modeling_class, self.grid))

        nan_cols_list = self.X.columns[self.X.isnull().any()].tolist()
        if nan_cols_list:
            nan_cols_string = " ".join(str(x) for x in nan_cols_list)
            log.warning("Columns in X contain NaNs: {}".format(nan_cols_string))

        nan_in_Y = pd.Series(self.Y).isnull().any()
        if nan_in_Y:
            log.warning("NaNs in Y")

    def _update_seasonal_order(self, seasonal_order):
        """
        Function to update the seasonal order in flat list or nested lists.
        Params:
            - seasonal_order: weekly or monthly seasonal order.
        Return:
            - seasonal_order: updated seasonal_order.
        """
        if seasonal_order is None:
            return seasonal_order

    def _remove_yearly_seasonal_order(self, seasonal_order):
        """A function to return filtered seasonal order for time series models, like ETS, Auto.Arima.
           :param seasonal_order: a list of lists that contains seasonal orders. Example: [[7], [7, 30], [7, 30, 365]].
           :return filtered seasonal order without yearly seasonality. Example: [[7], [7, 30]]
        """
        daily_yearly_seasonal_order = 365
        monthly_yearly_seasonal_order = 12
        filtered_order = []
        for order in seasonal_order:
            if isinstance(order, list):
                order = [item for item in order if item not in (daily_yearly_seasonal_order, monthly_yearly_seasonal_order)]
            else:
                # TODO: This state should not occure given the new assumption that the seaonal order is always a two level list of lists.
                #       If there are orders that are not lists they should be removed with a warning.
                if order == daily_yearly_seasonal_order or order == monthly_yearly_seasonal_order:
                    continue
            if len(order) == 0:
               filtered_order.append([1])
            elif order not in filtered_order:
                filtered_order.append(order)     
                
        return filtered_order 

    def model_summary_col_names(scoring_metrics):
        """A function to return scores from each cross validation splits.
        :param list scoring_metrics: a list of strings contains all the name of the score metrics.
        :param boolean return_cvtrain_score: if true, then report scores of each cross validate splits.
        :return list colnames: a list contains all the column names.
        """
        colnames = ['Model', 'Params', 'Error_Message']
        split_types = ['test', 'train'] 

        for metric in scoring_metrics:
            for split_type in split_types:
                colnames.append('CV_' + split_type + '_' + metric)
            colnames.append('Full_train_' + metric)

        return colnames

def _reconfigure_class(object, config):
    """ Given a dictionary of configuration elements (n_splits, etc.),
        this unpack function will assign the values of the keys with the same name to this object.
        Because some of the configs may change the data settings, we have to update_data() to make it consistent.

        Logic is: if model_factory has this key as an attribute, set attribute to the value of this key.
    """
    for k, v in config.items():
        if hasattr(object, k):
            setattr(object, k, v)

    return object