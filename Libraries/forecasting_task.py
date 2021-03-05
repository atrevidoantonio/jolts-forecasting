
import argparse
import collections
import os
import string
import sys
import time
import json
import pickle
import pandas as pd
import numpy as np
import logging
import csv

import statsmodels.api as sm

sys.path.append('.')

import libraries.profiling_tools as profiling_tools
import libraries.model_tools as mt
import libraries.models as md
import libraries.forecasting_tools as ft
import libraries.preprocess as pp

log = logging.getLogger()
# Initialize global variables
VOLUME_DATA_TYPE = "VolumeDataType"
LOG_STORAGE_CONTAINER = "logs"
INTERVAL_TYPE = "Interval"
NORMALIZE_VALUE_BY_DAY = "NormalizeValueByDay"
DATE_COL_NAME = "Date"
MODELING_CLASS = "modeling_class"
GRID = "grid"
LEN_WORKWEEK = "len_workweek"
SEASONAL_ORDER = "seasonal_order"
FORECAST_HEADER = ["DateID", "Value","Value_Normalized", "PI_Mean", "PI_Std_Dev", "Model"]

def false():
    return False

def main(data_path, config_path, data_config_path, dim_monthly_file_path= None, 
         output_dir=None):
    """Function to load input data and data/modeling configurations, execute modeling and forecasting pipeline and dump results to local file system.
    Args
    :param data_path (string) file path of training data for modeling.
    :param config_path (string) file path of bundled JSON modeling configurations.
    :param data_config_path (string) file path of bundled JSON data configurations.
    :param dim_monthly_file_path (string) file path for DimMonthlyReporting table.
    :param output_dir (string) output directory in local file system.
      """

    forecast_results = pd.DataFrame(columns = FORECAST_HEADER)
    summary_results = pd.DataFrame()
    
    log.info("Forecast Task Start")
    profiler = profiling_tools.Profiler()
    profiler.start_profiling("forecasting_task")
    start_time = time.time()

    profiler.start_profiling("reading_data_from_path")
    data_file = os.path.realpath(data_path)
    with open(data_file) as f:
        all_data = pd.read_csv(f, na_values = "None")
    profiler.end_profiling("reading_data_from_path")

    profiler.start_profiling("reading_configs_from_path")
    model_config_file = os.path.realpath(config_path)
    with open(model_config_file, 'r') as f:
        model_config_bundle = json.load(f)
    profiler.end_profiling("reading_configs_from_path")
    
    data_config_file = os.path.realpath(data_config_path)
    with open(data_config_file) as f:
        data_config = pd.read_csv(f, na_values = "None")

    dim_monthly_file = os.path.realpath(dim_monthly_file_path)
    with open(dim_monthly_file) as f:
        dim_monthly_data = pd.read_csv(f, na_values = "None")

    data_id = all_data[ID_COL_NAME].iloc[0]
    
    log.debug("Number of handlers = {}".format(len(log.handlers)))
    
    for _, model_config in model_config_bundle.items():
        modeling_display_name = model_config["modeling_display_name"]
        log.info("Running forecast pipeline for model {}...".format(data_id, modeling_display_name))
        model_config[INTERVAL_TYPE] = data_config[INTERVAL_TYPE].values[0]
        model_config[NORMALIZE_VALUE_BY_DAY] = data_config[NORMALIZE_VALUE_BY_DAY].values[0]
        model_config["forecast_start_date"] = (pd.to_datetime(str(data_config["Train_DateID_End_Desired"].values[0]), format = '%Y%m%d') + pd.DateOffset(1)).strftime('%Y%m%d')
        model_config["forecast_end_date"] = str(data_config["Forecast_DateID_End"].values[0])
        model_config[VOLUME_DATA_TYPE] = data_config[VOLUME_DATA_TYPE].values[0]
        model_config[MODELING_CLASS] = eval(model_config[MODELING_CLASS])()
        # Evaluate specific grid search values as a python expression
        for param_name in model_config['eval_grid']:
            param_vals = model_config['grid'][param_name]
            model_config['grid'][param_name] = [eval(val)() for val in param_vals]

        output_path = "output/{}/{}/{}".format(output_dir, data_id, modeling_display_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        modeling_logs_path =  "logs/{}/performance_logs".format(output_dir)

        if not os.path.exists(modeling_logs_path):
            os.makedirs(modeling_logs_path)

        forecast_file = "{}/Muid_{}_{}_forecast_{}.csv".format(
            output_path,
            data_id,
            modeling_display_name,
            time.strftime('%Y-%m-%d.%H.%M.%S'))

        summary_file = "{}/Muid_{}_{}_summary_{}.csv".format(
            output_path,
            data_id,
            modeling_display_name,
            time.strftime('%Y-%m-%d.%H.%M.%S'))

        log_file = "{}/Muid_{}_{}_log_{}.json".format(
            modeling_logs_path,
            data_id,
            modeling_display_name,
            time.strftime('%Y-%m-%d.%H.%M.%S'))

        ###  Run pipeline. ft.forecast_pipeline.
        log.info("Starting Pipeline")
        profiler.start_profiling("{}_pipeline".format(modeling_display_name))
        profiler.start_cpu_profiling("{}_cpu_pipeline".format(modeling_display_name))
        
        forecast, summary = ft.forecast_pipeline(all_data.drop(ID_COL_NAME, 1), full_configs=model_config, dim_monthly_table = dim_monthly_data)
        profiler.end_cpu_profiling("{}_cpu_pipeline".format(modeling_display_name))
        profiler.end_profiling("{}_pipeline".format(modeling_display_name))


        profiler.start_profiling("{}_output_data".format(modeling_display_name))

        if forecast is not None:
            forecast['Model'] = modeling_display_name
            forecast = forecast.reindex(FORECAST_HEADER, axis = 'columns')

        summary['Model'] = modeling_display_name
        summary = summary.reindex(mt.model_summary_col_names(model_config['scoring_metric']), )
        log.info("Outputting results to csv.")
        if forecast is not None:
            forecast.to_csv(forecast_file, sep=',', encoding='utf-8',index=False, quoting = csv.QUOTE_NONNUMERIC)
            pd.DataFrame(summary).transpose().to_csv(summary_file, sep = ',', encoding = 'utf-8', index = False, quoting = csv.QUOTE_NONNUMERIC)

    profiler.end_profiling("forecasting_task")
    profiler.add_context({ID_COL_NAME: data_id})
    profiles = profiler.get_profiles()
    
    print(profiles)
    end_time = time.time()
    log.info('Task end: {}, Elapsed time: {}'.format(end_time, end_time - start_time))
    handler.setFormatter(old_formatter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required = True)
    parser.add_argument('--config', required = True)
    parser.add_argument('--data_config_path', required = True)
    parser.add_argument('--dim_monthly_file_path', required = True)
    parser.add_argument('--output_dir', required = True)
  
    args = parser.parse_args()

    # Required so that the forecasting task will produce logs in batch.

    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s {}'.format(""))
    handler.setFormatter(formatter)
    log.addHandler(handler)
    log.setLevel(logging.INFO)
  
    main(args.data_path,
         args.config,
         args.data_config_path,
         args.dim_monthly_file_path,
         args.output_dir)