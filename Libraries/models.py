import logging
log = logging.getLogger(__name__)
import copy
import os
import inspect
import sys

from statsmodels.api import GLM
from statsmodels.api import GLS
from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

mingw_path = 'C:\\Program Files\\mingw-w64\\x86_64-5.3.0-posix-seh-rt_v4-rev0\\mingw64\\bin'
os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

sys.path.append(os.path.realpath('..'))
import rpy2
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects import numpy2ri
numpy2ri.activate()
from rpy2.robjects.packages import STAP
from rpy2.robjects.packages import importr
forecast = importr("forecast")
stats = importr("stats")
dlm = importr("dlm")

# STAP translates the signatures of the R functions in r_wrapper.R script 
# And stores it as an object with the variable name "r_wrapper".

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "R_wrapper.R"), 'r') as f:
    model_string = f.read()
r_wrapper = STAP(model_string, "r_wrapper")

def get_classes():
    clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    return clsmembers

# Initialize global variables
DATE_COL_NAME = 'Date'


def pandas_df_to_r_df(pDF):
    """
    This function converts pandas dataframe to r data frame, which is used for the input in the xreg in the auto.arima model. 
    For more details, refer to https://rpy2.readthedocs.io/en/version_2.7.x/rlike.html
    """
    orderedDict = rlc.OrdDict()

    for colName in pDF:
        colValues = pDF[colName].values
        orderedDict[colName] = robjects.IntVector(colValues)

    rDF = robjects.DataFrame(orderedDict)
    rDF.rownames = robjects.StrVector(pDF.index)
    return rDF


class glm(BaseEstimator, RegressorMixin):
    """ This class is a wrapper around the statsmodels GLM class.
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    def __init__(self, family=None, offset=None, exposure=None, freq_weights=None, missing=None, level=None):
        """
        Describe: Called when initializing the classifier
        :param family: The parent class for one-parameter exponential families.
        :param offset: An offset to be included in the model.
        :param exposure: Log(exposure) will be added to the linear prediction in the model.
                         Exposure is only valid if the log link is used.
        :param freq_weights: 1d array of frequency weights. The default is None.
        :param missing: Available options are 'none', 'drop', and 'raise'.
        :return: None
        """
        self.family = family
        self.offset = offset
        self.exposure = exposure
        self.freq_weights = freq_weights
        self.missing = missing
        self.level=level

    def fit(self, X, y):
        """
        Describe: fit classifer with response and explanatory variables
        :param X: explanatory variables
        :param y: response
        :return: GLMResults inherits from statsmodels.LikelihoodModelResults
        """
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        model = GLM(y, X, family=self.family, offset=self.offset, exposure=self.exposure,
                    freq_weights=self.freq_weights, missing=self.missing)

        self.fit_ = model.fit()
        return self

    def predict(self, X, y=None):
        """
        Describe: Called for prediction based on the fitted GLM object
        :param X: The explanatory variables to be used for prediction
        :param y: response variable which is None for this model
        :return: the predicted value for the explanatory variables
        """
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        yhat = self.fit_.predict(X)
        yhat = [max(0, x) for x in yhat]
        
        return (yhat)

class TSLM(BaseEstimator, RegressorMixin):
    """ This class is a wrapper around the tslm from R.
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    NULL = robjects.NULL
    #pandas2ri.activate()
    def __init__(self, seasonal_order = [12], pi_upper=None, level= 5, first_month = None):
        """
        Description: Called when initializing the regressor.
        Params:
            - pi_upper: upper level of the prediction interval
            - level: confidence level for prediction interval
            - first_month: first month of the time series
        Return: Prediction from TSLM in R
        """
        self.seasonal_order = seasonal_order
        self.level =level
        self.pi_upper = pi_upper
        self.first_month = first_month

    def fit(self, X, y=NULL):
        """
        Description: fit classifer with response and explanatory variables for TSLM model
        :param X: X (series, dataframe): series with a date column.
        :param y: array, the training data.
        :return: to self-object
        """ 
        self.first_month = X[DATE_COL_NAME][0]
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        model_input = stats.ts(robjects.FloatVector(y.values), frequency = self.seasonal_order[0])

        if X is not None and not X.empty:
            X = pandas_df_to_r_df(X.astype('int64'))
            self.model_fit_ = r_wrapper.fit_tslm_xreg(y = model_input, X = X, FUN = forecast.tslm)
        else:
            self.model_fit_ = r_wrapper.fit_tslm(y = model_input, FUN = forecast.tslm)
        return self

    def predict(self, X, y=None):
        """
        Description: Out-of-sample forecasts with forecast_steps
        :param X: X (series, dataframe): series with a date column.
        :param y: response variable which is None for this model
        :return: numpy array, prediction from TSLM in R.
        """
        forecast_steps = len(X[DATE_COL_NAME])
        # Create X_dummy, i.e. the monthly categorical variables as xreg
        X_dummy = X.drop([DATE_COL_NAME], axis = 1, inplace = False)

        if X_dummy is not None and not X_dummy.empty:
            X_dummy = pandas_df_to_r_df(X_dummy.astype('int64'))      
            ret_value = r_wrapper.predict_tslm_xreg(model = self.model_fit_, X = X_dummy, FUN = forecast.forecast, h = forecast_steps, level = self.level)
        else:
            ret_value = r_wrapper.predict(model = self.model_fit_, FUN = forecast.forecast, h = forecast_steps, level = self.level)
        
        # yhat is fitted.value when doing in-sample fitting and producing performance metrics and
        # yhat is forecasting.value when predicting forecasts.
        if self.first_day in list(X[DATE_COL_NAME]):
            yhat = ret_value[0]
        else:
            yhat = ret_value[1]

        self.pi_upper = ret_value[2]

        if yhat is rpy2.rinterface.NULL:
            yhat = None

        return (yhat)
class rf(BaseEstimator, RegressorMixin):
    """ This class is a wrapper around the sklearn random forest class.
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    def __init__(self, 	max_features='sqrt', min_samples_leaf =4, n_estimators = 10, max_depth = None, max_leaf_nodes = None, level=None):
        """
        Describe: Called when initializing the regressor.
        :param max_features: The number of features to consider when looking for the best split.
        :param n_estimators: The number of trees in the forest.
        :param max_depth: The maximum depth of the tree.
        :param max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion.
                               Best nodes are defined as relative reduction in impurity.
                               If None then unlimited number of leaf nodes.
        :return: None
        """
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.level=level

    def fit(self, X, y):
        """
        Describe: fit classifer with response and explanatory variables
        :param X: explanatory variables
        :param y: response
        :return: a random forest model object from sklearn
        """
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        model = RandomForestRegressor(max_features =self.max_features, min_samples_leaf = self.min_samples_leaf,
                                      n_estimators = self.n_estimators, max_depth = self.max_depth, 
                                      max_leaf_nodes = self.max_leaf_nodes)
        self.fit_ = model.fit(X,y)
        return self

    def predict(self, X, y=None):
        """
        Describe: Called for prediction based on the fitted random forest object
        :param X: The explanatory variables to be used for prediction
        :param y: response variable which is None for this model
        :return: the predicted value for the explanatory variables
        """
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        yhat = self.fit_.predict(X)
        yhat = [max(0, x) for x in yhat]
        return (yhat)

class svr(BaseEstimator, RegressorMixin):
    """
    Epsilon Support Vector Regression from scikit learn.
    """
    def __init__(self, gamma = 'scale', C = 1.0, epsilon = 0.1,  kernel = 'rbf',  degree =2, level =None):
        """
        Describe: Called when initializing the classifier
        :param C(float): Penalty parameter C of the error term. Default to be 1.0.
        :param epsilon(float): Epsilon in the epsilon-SVR model. Default to be 0.1.
        :param gamma(float): Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Default to be 'auto'.
        :param kernel(string): Specifies the kernel type to be used in the algorithm. Default to be 'rbf'.
        :return: None 
        """
        self.gamma = gamma
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.degree = degree
        self.level=level

    def fit(self, X, y, **kwargs):
        """
        Describe: fit classifer with response and explanatory variables
        :param X: explanatory variables
        :param y: response
        :return: a Epsilon-Support Vector Regression model object
        """
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        model = SVR(gamma = self.gamma, C = self.C, epsilon = self.epsilon,  kernel = self.kernel, degree = self.degree)
        self.fit_ = model.fit(X, y)
        return self

    def predict(self, X, y=None):
        """
        Describe: Called for prediction based on the fitted Epsilon-Support Vector Regression object
        :param X: The explanatory variables to be used for prediction
        :param y: response variable which is None for this model
        :return: the predicted value for the explanatory variables
        """
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        yhat = self.fit_.predict(X)
        yhat = [max(0, x) for x in yhat]
        return (yhat)

class ETS(BaseEstimator, RegressorMixin):
    """ Exponential Smoothing State Space Model.
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    def __init__(self, opt_crit="mse", ic= "aic", seasonal_order = [12], method = "ets", pi_upper=None, level= 5, first_month = None):
        """
        Description: Called when initializing the regressor.
        Params:
            - model: Usually a three-character string identifying method Hyndman et al. (2008).
            For more details, refer to https://www.rdocumentation.org/packages/forecast/versions/8.4/topics/ets
            - opt_crit: Optimization criterion.
            - ic: Information criterion to be used in model selection.
            - pi_upper: upper level of the prediction interval
            - level: confidence level for prediction interval
            - first_day: first date of the history data, used to differentiate where X is history data or future data and the 
            corresponding yhat is the fitted_value or forecasting_value
        Return: None
        """
        
        self.seasonal_order = seasonal_order
        self.method = method
        self.opt_crit = opt_crit
        self.ic = ic
        self.level=level
        self.pi_upper = pi_upper
        self.first_month = first_month

    def fit(self, X, y= None):
        """
        Describe: fit model with response and explanatory variables
        :param X: X (series, dataframe): series with a date column.
        :param y: array, the training data.
        :return: self object.
        """
        self.first_day = X[DATE_COL_NAME][0]
        model_input = stats.ts(robjects.FloatVector(y.values), frequency = self.seasonal_order[0])
        model_input = stats.ts(model_input, frequency = 12)
        self.model_fit_ = r_wrapper.fit(y = model_input, FUN = forecast.stlm, method = self.method)
        
        return self

    def predict(self, X, y = None):
        """
        Description: Exponential Smoothing State Space Model forecasts with forecast_steps
        :param X: X (series, dataframe): series with a date column.
        :param y: response variable which is None for this model
        :return: numpy array, prediction from Exponential Smoothing State Space Model in R.
        """
        forecast_steps = len(X[DATE_COL_NAME])
        ret_value = r_wrapper.predict(model = self.model_fit_, FUN = forecast.forecast, h = forecast_steps, level = self.level)
        
        # yhat is fitted.value when doing in-sample fitting and producing performance metrics and
        # yhat is forecasting.value when predicting forecasts.
        if self.first_month in list(X[DATE_COL_NAME]):
            yhat = ret_value[0]
        else:
            yhat = ret_value[1]

        self.pi_upper = ret_value[2]

        if yhat is rpy2.rinterface.NULL:
            yhat = None

        return (yhat)

class AutoArima(BaseEstimator, RegressorMixin):
    """ This class is a wrapper around the auto.arima from R.
        Returns best ARIMA model according to either AIC, AICc or BIC value.
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    def __init__(self, max_p = 5, max_q = 5, max_P = 2, max_Q = 2, max_d = 2, max_D = 1, seasonal = True, ic= "aic",
                 seasonal_order = None, pi_upper=None, level= 5, first_month= None):
        """
        Description: Called when initializing the regressor.
        Params:
            - max_p(integer): Maximum value of p.
            - max_q(integer): Maximum value of q.
            - max_P(integer): Maximum value of P.
            - max_Q(integer): Maximum value of Q.
            - max_d(integer): Maximum number of non-seasonal differences.
            - max_D(integer): Maximum number of seasonal differences.
            - seasonal(boolean): If FALSE, restricts search to non-seasonal models.
            - ic: Information criterion to be used in model selection.
            - seasonal_order(integer): the seasonal frequncy of the input object will be passed into grid search.
            - pi_upper: upper level of the prediction interval
            - level: confidence level for prediction interval
            - first_month: first month of the time series, used to differentiate where X is past or future data and the 
            corresponding yhat is the fitted_value or forecasting_value
        Return: None
        """
        self.max_p = max_p
        self.max_q = max_q 
        self.max_P = max_P
        self.max_Q = max_Q 
        self.max_d = max_d
        self.max_D = max_D
        self.seasonal = seasonal
        self.ic = ic
        self.seasonal_order = seasonal_order
        self.level = level
        self.pi_upper = pi_upper
        self.first_month = first_month

    def fit(self, X, y= None):
        """
        Describe: fit classifer with response and explanatory variables
        :param X: X (series, dataframe): series with a date column.
        :param y: array, the training data.
        :return: self object.
        """
        self.first_month = X[DATE_COL_NAME][0]
        X = X.drop([DATE_COL_NAME], axis=1, inplace=False)

        model_input = stats.ts(robjects.FloatVector(y.values), frequency = self.seasonal_order[0])

        if X is not None and not X.empty:
            X = pandas_df_to_r_df(X.astype('int64'))
            self.model_fit_ = r_wrapper.fit_autoarima(y=model_input,
                                                FUN = forecast.auto_arima,
                                                xreg = X,
                                                max_p = self.max_p, 
                                                max_q = self.max_q,
                                                max_P = self.max_P,
                                                max_Q = self.max_Q,
                                                max_d = self.max_d,
                                                max_D = self.max_D,
                                                seasonal = self.seasonal,
                                                ic = self.ic)
        else:
            self.model_fit_ = r_wrapper.fit(y=model_input,
                                                FUN = forecast.auto_arima,
                                                max_p = self.max_p, 
                                                max_q = self.max_q,
                                                max_P = self.max_P,
                                                max_Q = self.max_Q,
                                                max_d = self.max_d,
                                                max_D = self.max_D,
                                                seasonal = self.seasonal,
                                                ic = self.ic)
        return self

    def predict(self, X, y = None):
        """
        Description: ARIMA forecasts with forecast_steps
        :param X: X (series, dataframe): series with a date column.
        :param y: response variable which is None for this model
        :return: numpy array, prediction from ARIMA in R.
        """
        forecast_steps = len(X[DATE_COL_NAME])
        X_dummy = X.drop([DATE_COL_NAME], axis=1, inplace=False)
        if X_dummy is not None and not X_dummy.empty:
            #X = pandas2ri.py2ri(X.astype('int64'))
            X_dummy = pandas_df_to_r_df(X_dummy.astype('int64'))      
            ret_value = r_wrapper.predict_autoarima(model = self.model_fit_,
                                         FUN = forecast.forecast,
                                         xreg = X_dummy , level = self.level)
        else:
            ret_value = r_wrapper.predict(model = self.model_fit_,
                                         FUN = forecast.forecast,
                                         h = forecast_steps, level= self.level)
        
        # yhat is fitted.value when doing in-sample fitting and producing performance metrics and
        # yhat is forecasting.value when predicting forecasts.
        if self.first_month in list(X[DATE_COL_NAME]):
            yhat = ret_value[0]
        else:
            yhat = ret_value[1]

        self.pi_upper = ret_value[2]

        if yhat is rpy2.rinterface.NULL:
            yhat = None

        return (yhat)

class StlArima(BaseEstimator, RegressorMixin):
    """ This class is a wrapper around the auto.arima in R.
        Returns best ARIMA model according to either AIC, AICc or BIC value.
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    def __init__(self, seasonal_order=[12], method = "arima", pi_upper =None, level= 5):
        """
        Description: Called when initializing the regressor.
        Params:
            - pi_upper: upper level of the prediction interval
            - level: confidence level for prediction interval
            - first_month: first date of the history data, used to differentiate where X is history data or future data and the 
            corresponding yhat is the fitted_value or forecasting_value
        Return: None
        """
        self.seasonal_order = seasonal_order
        self.method = method
        self.level=level
        self.pi_upper = pi_upper

    def fit(self, X, y= None):
        """
        Describe: fit classifer with response and explanatory variables
        :param X: X (series, dataframe): series with a date column.
        :param y: array, the training data.
        :return: self object.
        """
        self.first_month = X[DATE_COL_NAME][0]
        model_input = stats.ts(robjects.FloatVector(y.values), frequency = self.seasonal_order[0])

        if len(y.values) <= self.seasonal_order[0]* 52 *2:
            if (self.seasonal_order[0] ==1) or (self.seasonal_order[0] ==12):
                self.model_fit_ = r_wrapper.fit(y= model_input,
                                                    FUN = forecast.auto_arima,
                                                    seasonal = True
                                                    )
            else:
                self.model_fit_ = r_wrapper.fit(y= model_input,
                                                            FUN = forecast.stlm,
                                                            method = self.method
                                                            )
        else:
            
            if self.seasonal_order[0] > 1:
                seasonal_period = robjects.IntVector([self.seasonal_order[0], self.seasonal_order[0]*52 ] )
                model_input = forecast.msts(model_input, seasonal_period)
                self.model_fit_ = r_wrapper.fit(y = model_input, 
                                                    FUN = forecast.stlm,
                                                    method = self.method
                                                    )
            else:
                model_input = stats.ts(model_input, frequency = 52)
                self.model_fit_ = r_wrapper.fit(y= model_input, 
                                                    FUN = forecast.stlm,
                                                    method = self.method
                                                    )

        return self

    def predict(self, X, y = None):
        """
        Description: ARIMA forecasts with forecast_steps
        :param X: X (series, dataframe): series with a date column.
        :param y: response variable which is None for this model
        :return: numpy array, prediction from ARIMA in R.
        """
        forecast_steps = len(X[DATE_COL_NAME])
        ret_value = r_wrapper.predict(model = self.model_fit_, FUN = forecast.forecast, h = forecast_steps, level = self.level)

        # yhat is fitted.value when doing in-sample fitting and producing performance metrics and
        # yhat is forecasting.value when predicting forecasts.
        if self.first_month in list(X[DATE_COL_NAME]):
            yhat = ret_value[0]
        else:
            yhat = ret_value[1]

        self.pi_upper = ret_value[2]

        if yhat is rpy2.rinterface.NULL:
            yhat = None
        return (yhat)


    def summarize(self):
        """
            Since Auto.ARIMA does its own grid search of the parameter space a method for extracting
            the best parameters and scores is not required.
        """
        raise NotImplementedError
class DLM(BaseEstimator, RegressorMixin):
    """ This Dynamic Linear Model (DLM) class is a wrapper around dlm from R.
        For more details, see https://cran.r-project.org/web/packages/dlm/dlm.pdf
        It inherits from sklearn BaseEstimator so that it can work with sklearn Pipelines.
    """
    NULL = robjects.NULL
    #pandas2ri.activate()
    def __init__(self, seasonal_order = [7, 365], pi_upper= None, level= None, first_month = None):
        """
        Description: Called when initializing the regressor.
        Params:
            - pi_upper: upper level of the prediction interval
            - level: confidence level for prediction interval
            - first_month: first date of the history data, used to differentiate where X is history data or future data and the 
            corresponding yhat is the fitted_value or forecasting_value
        Return: None
        """
        self.seasonal_order = seasonal_order
        self.level =level
        self.pi_upper = pi_upper
        self.first_month = first_month

    def fit(self, X, y=NULL):
        """
        Description: fit classifer with response and explanatory variables
        :param X: X (series, dataframe): series with a date column.
        :param y: array, the training data.
        :return: ARIMAResults inherits from statsmodels.tsa.arima_model
        """        
        self.first_month = X[DATE_COL_NAME][0]
        model_input = stats.ts(robjects.FloatVector(y.values), frequency = self.seasonal_order[0])

        self.model_fit_ = r_wrapper.fit_dlm(y= model_input, 
                                                EST= dlm.dlmMLE,
                                                FUN = dlm.dlmFilter)
        return self

    def predict(self, X, y=None):
        """
        Description: out-of-sample forecasts with forecast_steps
        :param x: x (series, dataframe): series with a date column.
        :param y: response variable which is none for this model
        :return: numpy array, prediction from DLM in r.
        """
        forecast_steps = len(X[DATE_COL_NAME])
        yhat = r_wrapper.predict_dlm(model = self.model_fit_, FUN = dlm.dlmForecast, nAhead= forecast_steps)

        # yhat is fitted.value when doing in-sample fitting and producing performance metrics and
        # yhat is forecasting.value when predicting forecasts.
        if self.first_month in list(X[DATE_COL_NAME]):
            yhat = self.model_fit_.rx2('f')
        else:
            yhat = yhat

        if yhat is rpy2.rinterface.NULL:
            yhat = None

        return (yhat)