import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import make_scorer

import time
import logging
log = logging.getLogger(__name__)


"""
Module to validate time-series models and report results.

List of classes:

* time_series_split: object called by validated_TS to split data into folds.
* metrics: holds the different scoring functions to be called by metrics class.

"""


class time_series_split():
    """ This class can be used to create rolling cross-validation splits (use n_splits = 1 for hold-out validation).
        n_splits = 0 means no cross validation will be done and the full training data will be used as the test data.
        Unlike the TimeSeriesSplit class from sklearn, which requires sorted data,
        and therefore doesn't recognize to split data with non-unique dates, this class splits
        directly based on dates. It also allows you to specify a window size, and a horizon size.
        
    """
    def __init__(self, n_splits = 1, window_size = None, horizon = 0.3, date_col = 'date', interval= None):
        """ Class constructor

            Args:
            n_splits (int): number of folds
            window_size (int): number of units in training set. Defaults to max. Units will be taken to be the sampling frequency
            horizon (int or float): if int, it's the number of units to forecast in test set at each fold. Similar unit limitation as in window_size.
                                    If under 1 (decimal), taken as pct of training data. If window_size is specified and horizon is a pct then will be taken as a pct of the window_size.
            date_col (string): the name of the column that will hold the date. Defaults to 'date'.
            interval (string): data on monthly level.

            Returns:
            Instance of class.

            Fields (not in Args):
            split_indices: will hold the indices of the dataset that were split on. See output of split function for more info.
            split_dates: the actual dates that were split on (the start/end of each fold). Note that the end date is not included.
            resolution: this is the sampling frequency of the data in units of days (1 day, 7 days, etc.).

        """
        self.window_size = window_size
        self.n_splits = n_splits
        self.horizon = horizon
        self.date_col = date_col
        self.split_indices = None
        self.split_dates = None
        self.interval = interval
        self.resolution = None

    def split(self, X):
        """This function indentifies and returns the training and test indices for each fold.

            Args:
            X (pandas Dataframe): this dataframe must have a column with dates. The dates must be a datetime object.
                                  Note that the indices of the training and test data must match that of this dataframe.

            Returns:
            [
            (split1_train_idxs, split1_test_idxs),
            (split2_train_idxs, split2_test_idxs),
            (split3_train_idxs, split3_test_idxs),
            ...
            ]: list of tuples holding training and test indices (as Int64Index objects in integer indices) corresponding to the appropriate folds.
        """
        # Store dates
        self.dates = X[self.date_col]

        if self.n_splits == 0:
            return(self._train_only(X))

        # Get minimum date
        min_date = min(self.dates)
        # Initialize values and create empty list to store results
        test_end = max(self.dates)

        # Set forecast horizon
        horizon = int(self.horizon*self.resolution)

        split_indices = []
        split_dates = []
        # Loop for each fold
        for fold in range(0, self.n_splits):
            test_start = test_end - pd.Timedelta(months=horizon)   # test_start is included (unlike last_date_in_fold)
            train_end = test_start    # not actually included in training data
            if self.window_size is None:
                train_start = min_date # first month in the dataset is default
            else:
                train_start = train_end - pd.Timedelta(months=int(self.window_size*self.resolution))   # start of training data
            train_index = self.date[(self.dates >= train_start) & (self.dates < train_end)].index
            test_index = self.date[(self.dates >= test_start) & (self.dates < test_end)].index
            split_indices.append((train_index, test_index))
            # Just for bookeeping later on
            split_dates.append({"train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end})
            # Update value before loop restarts
            test_end = test_start
        self.split_dates = split_dates
        self.split_indices = split_indices
        return(self.split_indices)


    def get_splits(self):
        """This is a getter function for the indices that split on.
        Returns:
            [
            (split1_train_idxs, split1_test_idxs),
            (split2_train_idxs, split2_test_idxs),
            (split3_train_idxs, split3_test_idxs),
            ...
            ]: list of tuples holding training and test indices (as Int64Index objects in integer indices) corresponding to the appropriate folds.
        """
        return(self.split_indices)

    def _train_only(self, X):
        """ Function is used when n_splits = 0, which signals to return all indices for training and test.
            The use is to exclusively measure training error when there isn't sufficient data for validation.
        """
        train_start = min(self.dates)
        train_end = max(self.dates)
        test_start = train_start
        test_end = train_end
        split_indices = []
        split_dates = []
        train_index = self.dates.index
        test_index = self.dates.index
        split_indices.append((train_index, test_index))
        split_dates.append({"train_start": train_start, "train_end": train_end, "test_start": test_start, "test_end": test_end})
        self.split_dates = split_dates
        self.split_indices = split_indices
        return(split_indices)

class metrics():
    """ This class will be used to store metrics for evaluation.

        scoring_metric = getattr(metrics, <name of metric; e.g. 'mean_squared_error'>)
        scoring_metric(y, y_hat)

        This way you can evaluate y/y_hat using the same method call (scoring_metric) but can pass in
        name of metric as a parameter.

        All metrics must be defined with the parameters observed and predictions and one of the
        decorators @greater_is_better, @less_is_better, or @not_optimizer.
    """
    def greater_is_better(func):
        func.valid_optimizer = True
        func.greater_is_better = True
        return func

    def less_is_better(func):
        func.valid_optimizer = True
        func.greater_is_better = False
        return func

    def not_optimizer(func):
        func.valid_optimizer = False
        func.greater_is_better = True
        return func

    def get_metric_list():
        """ Returns list of metrics stored in this class. Should be continuously updated as metrics are added.
        """
        metrics = ['WAPE', 'MSE', 'RMSE', 'CVRMSE', 'R_squared', 'sum_of_validation_training', 'sum_of_validation_residuals', 'sum_of_validation_forecast']
        return(metrics)

    def get_metric(self, metric):
        scoring_metric = getattr(metrics, metric)
        return(scoring_metric)

    # The function sklearn.metrics.make_scorer wraps scoring functions for use in GridSearchCV. 
    # It takes a metric function, such as accuracy_score, mean_squared_error, adjusted_rand_index or average_precision 
    # and returns a callable that evaluates an estimator’s output.
    def get_make_metric(self, metric):
        scoring_metric = self.get_metric(metric)
        if not (hasattr(scoring_metric, 'valid_optimizer') or hasattr(scoring_metric, 'greater_is_better')):
            log.error('The function definition for metric "{}" is missing a required decorator.'.format(metric))
            raise RuntimeError('The function definition for metric "{}" is missing a required decorator.'.format(metric))

        # make_metric does not preserve function attributes
        valid_optimizer = scoring_metric.valid_optimizer
        greater_is_better = scoring_metric.greater_is_better

        scoring_metric = make_scorer(scoring_metric, greater_is_better = scoring_metric.greater_is_better)
        scoring_metric.valid_optimizer = valid_optimizer
        scoring_metric.greater_is_better = greater_is_better

        return(scoring_metric)

    @less_is_better
    def WAPE(observed, predictions):
        """
        Weighted Absolute Percentage Error: abs(Forecast - Observed)/Observed.Range: (0, 100% )
        """
        observed, predictions = np.array(observed), np.array(predictions)
        return(np.sum(np.abs(observed-predictions))/np.sum(observed))

    @less_is_better
    def MSE(observed, predictions):
        """
        Mean Squared Error. Range: (0, inf)
        """
        observed, predictions = np.array(observed), np.array(predictions)
        return(np.mean(np.square(observed-predictions)))

    @less_is_better
    def RMSE(observed, predictions):
        """"
        Root Mean Squared Error. Range: (0, inf)
        """
        observed, predictions = np.array(observed), np.array(predictions)
        return(np.sqrt(np.mean(np.square(observed-predictions))))

    @less_is_better
    def CVRMSE(observed, predictions):
        """
        Coefficient of Variation of Root Mean Squared Error, i.e. RMSE normalized by the mean of the data.
        """
        observed, predictions = np.array(observed), np.array(predictions)
        cv_rmse = np.sqrt(np.mean(np.square(observed-predictions)))/np.mean(observed)
        return(cv_rmse)

    @greater_is_better
    def R_squared(observed, predictions):
        """
        R squared: proportion of variance explained by the model, ranges from 0 to 1. When the model is a bad fit to the data, 
        it can lead to negative values, implying that the fit is actually worse than just fitting a horizontal line rather than the model.
        
        """
        observed, predictions = np.array(observed), np.array(predictions)
        if np.all(np.abs(observed-predictions) > 1e-6):
            ssr = np.sum(np.square(observed-predictions))
            sst = np.sum(np.square(observed-np.mean(observed)))
            Rsquared = 1 - ssr/sst
        else:
            Rsquared = 1
        return(Rsquared)

    @not_optimizer
    def sum_of_validation_training(observed, predictions):
        observed, predictions = np.array(observed), np.array(predictions)
        return(np.sum(observed))

    @not_optimizer
    def sum_of_validation_residuals(observed, predictions):
        observed, predictions = np.array(observed), np.array(predictions)
        return(np.sum(observed-predictions))

    @not_optimizer
    def sum_of_validation_forecast(observed, predictions):
        observed, predictions = np.array(observed), np.array(predictions)
        return(np.sum(predictions))
