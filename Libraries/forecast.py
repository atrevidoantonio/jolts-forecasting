from Libraries.model_tools import model_input
import pandas as pd
import numpy as np
from itertools import chain

import libraries.model_tools as mt
import libraries.preprocess as pp

import logging
log = logging.getLogger(__name__)

# For calculating standard deviation
from scipy.stats import norm 

# Initialize global variables
DATE_COL_NAME = "Date"

def forecast_sequence(training_input, full_configs, monthly_table = None, *args):
    """ Function for creating an end-to-end forecast for a given dataset and model.

        Args:
        :param training_input (pandas dataframe):   training data
        :param full_configs (dict):     dictionary with model configurations and grid search configurations.
        :param monthly_table (pandas dataframe): pandas dataframe contains date column 'DateID' and number of days column 'DaysPerMonth'.

        Return:
        forecast (pandas dataframe): a dataframe holding unified forecast.
        summary (pandas dataframe): a dataframe holding the model summary information.
    """
    log.info("Unloading Configurations.")
    # Initialize static variables
    exp_fcst_threshold = 10

    #Initialize configurable variables
    target_col          = full_configs['target_col_name']
    scoring             = full_configs['scoring_metric']
    forecast_end_date   = pd.to_datetime(full_configs["forecast_end_date"], format = '%Y%m%d')
    forecast_start_date = pd.to_datetime(full_configs["forecast_start_date"], format = '%Y%m%d')
    modeling_display_name = full_configs["modeling_display_name"]
    features = full_configs["features"]
    interval_type = full_configs["Interval"]
    # Initialize dependent variables
    max_training = training_input[target_col].max()
    # Set defaults
    model    = None
    forecast = None

    log.info("Creating Modeling objects.")
    ####             Instanciate Objects            ####
    ####################################################
    model_input = mt.model_input(dataset=training_input, configs=full_configs)

    log.info("Creating forecasting objects.")
    fcst_seq  = forecast_sequence(forecast_start_date = forecast_start_date, forecast_end_date=forecast_end_date,configs=full_configs, monthly_table = monthly_table, is_larger_than_one_year = model_input.is_larger_than_one_year)
    

    # Filter Data
    sum_total = training_input[target_col].mean()
    if sum_total == 0:
        log.warning("Data has no observations!")
        return None, model_input.zero_summary("Data had no observations!")

    ####          Modeling and Forecasting          ####
    log.info("Modeling with type {}.".format(modeling_display_name))
    # Modeling try-catch
    try:
        model    = model_input.generate_model()
        summary  = model_input.summarize()
    except Exception as e:
        log.exception("Could not produce model for model type '{}'. See Error: {}".format(modeling_display_name, e))
        return None, model_input.zero_summary(str(e))

    log.info("Forecasting with model {}.".format(modeling_display_name))
    # Forecasting try-catch
    try:
        forecast = fcst_seq.produce_forecast(model)
    except Exception as e:
        log.error("Could not produce forecast for model type '{}'. See Error: {}".format(modeling_display_name, e))
        return None, model_input.zero_summary(str(e))

    # Detect and filter out exploding forecast
    forecast, summary = fcst_seq.detRemExpFcst(summary, max_training, forecast, exp_fcst_threshold)
    
    log.info("Forecast Complete.")
    return forecast, summary

class forecast_procedure():
    def __init__(self, forecast_start_date=None, forecast_end_date=None, date_col_name=None, configs=None, monthly_table=None, is_larger_than_one_year=False, features=None, interval=None, volume_data_type=None, drop_first=True):
        """
        Args:
        :param forecast_start_date: (pandas timestamp) forecast start date. 
        :param forecast_end_date: (pandas timestamp) forecast end date.
        :param date_col_name: (string) name of date column in output table.
        :param cofigs: (dict) model configurations.
        :param monthly_table: (pandas dataframe) a dataframe containing months
        :param is_larger_than_one_year: (boolean) used to specify whether create month features in the forecast input dataset.
        :param features: (list) list of strings for features to be create for ML models.
        :param Interval: (string) a monthly dataset identifier.
        :param volume_data_type: (string) if 'int' the forecast value will be rounded into integer, otherwise will be keeping as float.
        """
        # Initialize configurable paramters
        self.forecast_start_date = forecast_start_date
        self.forecast_end_date = forecast_end_date
        self.date_col_name=date_col_name
        self.monthly_table = monthly_table
        self.is_larger_than_one_year = is_larger_than_one_year
        self.features = features
        self.Interval = interval
        self.VolumeDataType = volume_data_type
        self.drop_first = drop_first
       
        #Override attributes above according to the configuration
        if configs is not None:
            self.unpack(configs)

        # Initialize static attributes that are for internal use and not intended to be configurable
        self.date_col="Date"
        self.target_col_name = "Value"
        self.value_normalized = "Value_Normalized"
        self.value_pi_mean = "PI_Mean"
        self.value_std_dev = "PI_Std_Dev"
        self.days_per_month = "DaysPerMonth"
        self.normalize_name = "NormalizeValueByDay"
        self.dates=None
        self.max_forecast = None

        # Initialize dependant attributes
        self.modeling_class = str(configs["modeling_class"]).split("(")[0]
        self.forecast_input = self.generate_monthly_forecast_input()
        self.monthly_table[self.days_per_month] = self.monthly_table[self.days_per_month].astype(int)
        self.normalize = configs[self.normalize_name]

    def generate_pi_std_dev(self, target, pi_upper, level):
        """ 
        Function to generate forecast standard deviation used for calculating prediction interval (pi).
        "pi" is short for prediction interval.
        """
        z_score = norm.ppf( 1 - (1 - level * 0.01) /2 )
        pi_std_dev =list( (np.array(pi_upper) - np.array(target) ) /z_score )
        return (pi_std_dev)


    def produce_forecast(self, model):
        """ Basic function to produce a forecast and forecast std_dev. 
            The std_dev is only calculated for time series models.
            :param model: a model object from models.
            Returns:
            Dataframe with schema of |Date|forecast value| -> with column names as passed to this function.
        """
        ### TO-DO ### CHECK THAT SIZE OF DATES AND X MATCH
        targets  = model.predict(self.forecast_input)
           
        if model.level is not None: 
            pi_std_dev = self.generate_pi_std_dev( targets, model.pi_upper, model.level)
        else:
            pi_std_dev = None

        dates = list(self.dates.apply(lambda x: x.strftime('%Y%m%d')))
        forecast_table = self.populate_table(dates, targets, pi_std_dev, model)
        # print(forecast_table)
        return(forecast_table)


    def populate_table(self, dates, targets, pi_std_dev, model):
        """
        Description: A function to line up forecasting values, std_dev, and forecasting dates
                    and return a Pandas dataframe object populated with forecasting
                    values and dates.
        Params:
            - dates: a forecating dates column contains starting and end of 
            forecasting dates
            - targets: a list contains all the forecast point estimate.
            - pi_std_dev: a list contains the forecast standard deviations of time series models.
        Return: 
            - forecast_table: A pandas dataframe contains forecast point estimate, forecast error, and dates.
        """
        self.max_forecast = max(targets)
        forecast_table = pd.DataFrame()
       
        # fill in closed dates
        forecast_table[self.date_col_name]   = dates
        forecast_table[self.date_col_name] = forecast_table[self.date_col_name].astype(int)
        forecast_table[self.target_col_name] = targets
        forecast_table = pd.merge(forecast_table, self.monthly_table, on = self.date_col_name, how = 'left')
        forecast_table[self.value_normalized] = forecast_table[self.target_col_name]
        forecast_table[self.value_pi_mean] = forecast_table[self.target_col_name]
        forecast_table[self.value_std_dev] =  None
        if model.level is not None:
            forecast_table[self.value_std_dev]= pi_std_dev
        if self.normalize:
                # Denormalize Value column here, then include in the forecast table columns below
                forecast_table[self.target_col_name] = forecast_table[self.target_col_name]*forecast_table[self.days_per_month]
                forecast_table[self.value_pi_mean] = forecast_table[self.target_col_name]
                if model.level is not None:
                    forecast_table[self.value_std_dev]= pi_std_dev*forecast_table[self.days_per_month]
                forecast_table = forecast_table[[self.date_col_name,self.target_col_name, self.value_normalized,self.value_pi_mean,self.value_std_dev]]
        else:
            forecast_table[self.target_col_name] = forecast_table[self.target_col_name]
            forecast_table = forecast_table[[self.date_col_name,self.target_col_name, self.value_normalized,self.value_pi_mean,self.value_std_dev]]
            
        if self.VolumeDataType == 'int':
            forecast_table[self.target_col_name] = [round(x,0) for x in forecast_table[self.target_col_name]]
            # Add this for the value normalized column
            forecast_table[self.value_normalized] = [round(x, 0) for x in forecast_table[self.value_normalized]]

        # replace all negative values with zeros
        forecast_table[self.target_col_name] =[max(0, x) for x in forecast_table[self.target_col_name] ]
        forecast_table[self.value_normalized] =[max(0, x) for x in forecast_table[self.value_normalized] ]

        return forecast_table

    def generate_zero_forecast(self):
        """
        Description: Appends a column of zero and a identifier column to a dateframe of dates.
        Params: None
        Return:
            - forecast_table: a Pandas dataframe contains a zero cloumn and forecasting dates.
        """
        dates   = list(self.dates.apply(lambda x: x.strftime('%Y%m%d')))
        targets = [0]*len(dates)
        forecast_table = self.populate_table(dates, targets)
        return forecast_table

    def unpack(self, configs):
        """ Given a dictionary of configuration elements
            this unpack function will assign the values of the keys with the same name to this object.
            Because some of the configs may change the data settings, we have to update_data() to make it consistent.

            Logic is: if forecast_factory has this key as an attribute, set attribute to the value of this key.
        """
        for k, v in configs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def detRemExpFcst(self, summary, max_training, forecast, threshold):
        """
        Description: Function to detect and log exploding forecast in SummaryModel.
                     Replace exploding forecast to None if exploding.
        Params:
            - summary: a summary table to save all the summary information from modeling.
            - max_training: the maximum of training (normalized).
            - threshold: the threshold to be used in comparing the maximum training(normalized) and maximum
                forecast(normalized) for detecting and removing exploding forecast.
        Return:
            - summary: a summary table for saving all the summary information from modeling.
            - forecast: a single forecast for a pair of MUID and model.
        """

        if (self.max_forecast is not None and self.max_forecast > threshold * max_training):
            log.info("Exploding forecast and will not be attached to final results.")
            # If there is an Error_Message then the metrics should be all NULL
            summary.iloc[~summary.index.isin(['Params'])]=None
            summary['Error_Message'] = 'Exploding forecast with maximum ' + str(threshold) + ' times larger than that of training data.'
            forecast = None
        return forecast, summary

    def detRemNaNPI(self, summary, forecast):
        """
        Description: Function to detect and remove forecasts with NaN prediction intervals. 
        Params:
            - summary: a summary table to save all the summary information from modeling.
            - pi_std_dev: the standard deviation of the prediction interval
        Return:
            - summary: a summary table for saving all the summary information from modeling.
            - forecast: a single forecast for a pair of MUID and model.
        """
        if (  all(x is not None for x in forecast['PI_Std_Dev']) ):
            if (np.all(np.isnan( forecast['PI_Std_Dev'])) ):
                log.info("Forecasts have NaN prediction intervals and will not be attached to final results.")

                # If Error_Message is not None, then the metrics should be all NULL
                summary.iloc[~summary.index.isin(['Params'])]=None
                summary['Error_Message'] = 'Forecast prediction interval is NaN.'
                forecast = None

        return forecast, summary

    def generate_monthly_forecast_input(self):
        """Function to be used to generate monthly forecast input dataframe by forecast starting date and end date.
            Return:
                - forecast_input: (pandas dataframe) input dataset for monthly forecasting pipeline.
        """
        forecast_end_date = pd.to_datetime(self.forecast_end_date)
        forecast_start_date = pd.to_datetime(self.forecast_start_date)
        forecast_horizon = (forecast_end_date-forecast_start_date).days + 1
        forecast_input = pd.DataFrame(pd.date_range(forecast_start_date, forecast_end_date, freq=pd.offsets.MonthBegin()), columns = [self.date_col])
        forecast_input= pp.create_all_time_features(df = forecast_input, feature_cols = self.features, date_col = self.date_col, target_col = self.target_col_name, is_larger_than_one_year = self.is_larger_than_one_year, drop_first = self.drop_first)
        log.debug("Forecast Input Head: {}".format(forecast_input.head().to_string()))
        return forecast_input