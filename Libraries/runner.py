import os
import pickle
import csv
import sys
import logging
from random import shuffle
import datetime
import itertools
import json
import time

import numpy as np
import pandas as pd
sys.path.append('../..')

import configurations.configuration_tools as cfg
import scripts.forecasting_task as task

log = logging.getLogger(__name__)


def create_data_files(training_data, summary_training, config_data,  id_col_name="date",
                      interval_col_name="Interval"):
    """ Creates list of input, and data config files for each unique training dataset.

        Args:
        :type master_ids: `numpy.ndarray`
        :param training_data: The training DataFrame to process.
        :type training_data: `pandas.DataFrame`        
        :param summary_training: A DataFrame containing summary information of the training data.
        :type summary_training: `pandas.DataFrame`
        :param config_data: The config DataFrame to write to file.
        :type config_data: `pandas.DataFrame`
        :param str interval_col_name: The column name for the interval column in data_config_file_paths. Default is "Interval".
        :rtype: tuple
        :return: (list of input file paths, list of data config file paths)
    """
    input_file_paths = {}
    data_config_file_paths = {}
    intervals = config_data[interval_col_name].unique()
    for interval in intervals:
        input_file_paths[interval] = []
        data_config_file_paths[interval] = []

    log.info("Setting up input file paths. Adding training data files and data config files for each task...")

        ### Adding Training data file for each task
    interval = config_data[config_data[id_col_name]] == [interval_col_name].values[0]
    data_file = "data/training_data_{}.csv"
    input_file_paths[interval].append(os.path.realpath(data_file))
    training_data[training_data[id_col_name] == date].to_csv(data_file)
        ### Adding data config file for each task
    data_config_file = "data/data_config_{}.csv".format(date)
    data_config_file_paths[interval].append(os.path.realpath(data_config_file))
    config_data_copy = config_data.copy()
    config_data_copy.to_csv(data_config_file)

    return input_file_paths, data_config_file_paths


def create_model_files(config_dir, intervals, methods, task_config_dir="modeling_configs"):
    """ Creates a list of config files for each interval type.

        Args:
        :param str config_dir: The directory source the model config files.
        :param intervals: Iterable containing names of intervals.
        :type intervals: `dict_keys`
        :param list methods: The names of modeling methods to use.
        :param str task_config_dir: Name of the output directory for config files.
        :rtype: list
        :return: config file paths
    """
    config_file_paths = {}
    log.info("Creating Model files...")
    for interval in intervals:
        cfg_mng = cfg.configuration_manager(config_path=config_dir)
        cfg_mng.set_inclusions(methods[interval])
        config_file_paths[interval] = [os.path.realpath(file_path)
                                       for file_path in cfg_mng.save_configs_list([task_config_dir])]
    return config_file_paths


def create_task_configs(data_config_file_paths, input_file_paths, model_config_file_paths, intervals):
    """ Combines data inputs, and modeling configs by interval. Bundles model configs for each
        interval into one file.
        Together the three files (data, data config, and model config) define a forecasting task.

        Args:
        :param dict data_config_file_paths: Contains a list of data config file paths for each interval.
        :param dict input_file_paths: Contains a list of data input file paths for each interval.
        :param dict model_config_file_paths: Contains a list of model config file paths for each interval.
        :param intervals: Iterable containing names of intervals.
        :type intervals: `dict_keys`
        :rtype: list of tuples
        :return: [((data_path, data_config_path), model_config_file)],
    """

    filepaths = []
    model_bundle_paths = []
    log.info("Creating Task configs...")
    for interval in intervals:
        zipped_input_config_file_paths = zip(input_file_paths[interval], data_config_file_paths[interval])
        bundled_model_config = {}

        for model_path in model_config_file_paths[interval]:
            with open(model_path, "r") as f:
                current_json = json.load(f)
                bundled_model_config[os.path.basename(f.name)] = current_json

        bundled_file_path = os.path.realpath("modeling_configs/bundled_{}_configs.json".format(interval))
        model_bundle_paths.append(bundled_file_path)
        with open(bundled_file_path, "w+") as bundled_file:
            bundled_file.write(json.dumps(bundled_model_config))

        for pair in zipped_input_config_file_paths:
            filepaths.append(((pair), bundled_file_path))
    return filepaths, model_bundle_paths


def run_tasks(config, task_files, dim_monthly_file_path, log_container_sas_token,
                    output_container_sas_token, output_file_dir, to_blob, log_to_blob, summary_blob_name,
                    forecast_blob_name):
    """
    Args
    :param config (object) configurations with JSON format.
    :param task_files (list) list contains training data, data configuration and modeling configuration in format of ((training data, data configuration), modeling configuration).
    :param dim_monthly_file_path (string) local system file path for dim monthly reporting file.
    :param output_file_dir (string) output file directory for either local file system or part of the blob file name.
    Return:
        None
    """
    log.info("Running locally....")
    run_locally = True
    total = len(task_files)
    for i, ((data_path, data_config_path), model_config_file_path) in enumerate(task_files):
        task.main(data_path,
                  model_config_file_path,
                  data_config_path,
                  dim_monthly_file_path=dim_monthly_file_path,
                  output_dir=output_file_dir,
                  run_locally=run_locally)

def data_prep(config, input_id):
    """
    Description: a function to wrap the data preparation process, including input trianing data, input configuration data, model configuration data, etc.
    Params:
        - config: (dictionary) contains data and model configurations.
    Return:
        - master_ids: numpy array, contains all the muids that will be trained and predicted.
        - training: pandas dataframe, contains all the daily and monthly training input data.
        - summary_training: pandas dataframe, contains all the daily and monthly data from the summary training table.
        - full_master_config_data: pandas dataframe, contains all the daily and monthly input data configurations for all muids.
        - dim_monthly_file_path: real file path in system, a file path target to the dim_monthly_file.
        - message_random: sting, a string to be shown in console that indicating the sample is randomized or not.
    """
    
    if config.input.INPUT_TABLE_NAME is not None:
        log.info("Using table {}".format(config.input.INPUT_TABLE_NAME))
        training_data = input_connect.read(input_name=config.input.INPUT_TABLE_NAME)
        training_data = training_data.loc[
            training_data[config.input.INPUT_FILTER_COLUMN] >= config.input.INPUT_FILTER_VALUE,
            [config.input.ID_COL_NAME, config.input.DATE_COL_NAME, config.input.TARGET_COL_NAME,
             config.input.NORMALIZATION_VALUE]
        ]
        training_data[config.input.TARGET_COL_NAME] = training_data[config.input.TARGET_COL_NAME].astype('float64')

        # Used in arbitration. Will be depricated soon after arbitration is ported to SQL R Server.
        training_data[config.input.TRAINING_MAX_VALUE] = training_data[config.input.NORMALIZATION_VALUE] * \
                                                         training_data[config.input.TARGET_COL_NAME]
        training_max_value = pd.DataFrame(
            training_data.groupby([config.input.ID_COL_NAME])[config.input.TRAINING_MAX_VALUE].max() )
        training_max_value.index.name ='None'
        training_max_value[config.input.ID_COL_NAME] = training_max_value.index

        summary_training = input_connect.read(input_name=config.input.SUMMARY_TRAINING_TABLE_NAME)
        summary_training = summary_training.loc[:, [config.input.ID_COL_NAME,
                                                    config.input.TRAINING_DATE_END,
                                                    config.input.SUMMARY_TRAINING_WORKWEEK,
                                                    config.input.TRAINING_COUNT_FINAL,
                                                    config.input.EXCLUSION_REASON,
                                                    config.input.TRAIN_DATE_END_DESIRED,
                                                    config.input.FORECAST_DATE_END,
                                                    config.input.VOLUME_DATA_TYPE,
                                                    config.input.INTERVAL_TYPE,
                                                    config.input.NORMALIZE_VALUE_BY_DAY]]
        summary_training[config.input.SUMMARY_TRAINING_WORKWEEK] = [workweek_to_dict(week) for week in summary_training[
            config.input.SUMMARY_TRAINING_WORKWEEK]]
        # Can be depricated when arbitration ported to SQL R Server.
        summary_training = summary_training.merge(training_max_value, left_on=config.input.ID_COL_NAME,
                                                  right_on=config.input.ID_COL_NAME, how="left")

    else:
        training_data = pd.DataFrame(
            columns=[config.input.ID_COL_NAME, config.input.DATE_COL_NAME, config.input.TARGET_COL_NAME])
        summary_training = pd.DataFrame(
            columns=[config.input.ID_COL_NAME, config.input.SUMMARY_TRAINING_WORKWEEK, config.input.TRAINING_MAX_VALUE,
                     config.input.EXCLUSION_REASON])

    training_data.rename(columns={config.input.DATE_COL_NAME: config.output.DATE_COL_NAME,
                                  config.input.TARGET_COL_NAME: config.output.TARGET_COL_NAME}, inplace=True)
    master_ids = np.sort(training_data[config.input.ID_COL_NAME].unique())
    summary_training = summary_training[summary_training[config.input.EXCLUSION_REASON].isnull()]

    full_master_config_data = summary_training.loc[
                              :,
                              [config.input.ID_COL_NAME, config.input.TRAINING_DATE_END, config.input.FORECAST_DATE_END,
                               config.input.VOLUME_DATA_TYPE, config.input.INTERVAL_TYPE,
                               config.input.NORMALIZE_VALUE_BY_DAY,config.input.SUMMARY_TRAINING_WORKWEEK]
                              ]
    # FIXME: This renamed operation is confusing because both values exist externally.
    #        This should be fixed such that this config variable is only refered to as the
    #        FORECAST_START_DATE and defining it to be equal to TRAINING_DATE_END + 1.
    full_master_config_data.rename(columns={config.input.TRAINING_DATE_END: config.input.TRAIN_DATE_END_DESIRED},
                                   inplace=True)
    summary_training = summary_training.loc[:,
                       [config.input.ID_COL_NAME, config.input.TRAINING_COUNT_FINAL, config.input.TRAINING_MAX_VALUE,
                        config.input.VOLUME_DATA_TYPE, config.input.SUMMARY_TRAINING_WORKWEEK]]

    # If SUPPORTED_INTERVALS is not subset remove the offending interval types from  master ids.
    if set(config.forecasting.SUPPORTED_INTERVALS) < set(full_master_config_data[config.input.INTERVAL_TYPE].unique()):
        log.info(
            "Found unsupported interval in input. MUIDs will be removed from master_ids if in unsupported intervals.")
        unsupported_intervals = set(full_master_config_data[config.input.INTERVAL_TYPE].unique()) - set(
            config.forecasting.SUPPORTED_INTERVALS)
        unsupported_muids = full_master_config_data[config.input.ID_COL_NAME].loc[
            full_master_config_data[config.input.INTERVAL_TYPE].isin(unsupported_intervals)].unique()
        master_ids = np.setdiff1d(master_ids, unsupported_muids)

    # Read dim monthly reporting data
    dim_monthly_reporting = input_connect.read(input_name=config.input.DIM_MONTHLY_REPORTING_TABLE_NAME)
    dim_monthly_reporting = dim_monthly_reporting.loc[:, [config.input.DATE_COL_NAME, config.input.DAYS_PER_MONTH]]
    dim_monthly_reporting[config.input.DATE_COL_NAME] = dim_monthly_reporting[config.input.DATE_COL_NAME].astype(int)

    dim_monthly_file = "data/dim_monthly_reporting.csv"
    dim_monthly_reporting.to_csv(dim_monthly_file, index=False)
    dim_monthly_file_path = os.path.realpath(dim_monthly_file)

    return master_ids, training_data, summary_training, full_master_config_data, dim_monthly_file_path, message_random




