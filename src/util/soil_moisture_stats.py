from typing import Optional, Union
import numpy as np

"""
  DESCRIPTION
            The Soil moisture estimation metrics
            
            The metrics are based on the following paper:
            Entekhabi, D., R. H. Reichle, R. D. Koster, and W. T. Crow, 2010: 
            Performance Metrics for Soil Moisture Retrievals and Application Requirements. 
            J. Hydrometeor., 11, 832–840, https://doi.org/10.1175/2010JHM1223.1.
             
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-09-29
   Updated  2022-06-22: clean up and add comments
 
   Copyright 2022 University of Southern California
"""


def cal_rmse(true_sm: np.ndarray, est_sm: np.ndarray, axis: Optional[Union[int, tuple]] = None):
    """
    Calculate root mean square error (RMSE) of soil moisture estimation. This except values of nan


    :param true_sm: True soil moisture values
    :param est_sm: Estimated soil moisture values
    :param axis: axis of summation
    :return: RMSE
    """
    mse = np.sqrt(np.nanmean((true_sm - est_sm) ** 2, axis=axis))
    return mse


def cal_unbiased_rmse(true_sm: np.ndarray, est_sm: np.ndarray, axis: Optional[Union[int, tuple]] = None):
    """
    Calculate unbiased root mean square error (RMSE) of soil moisture estimation. This except values of nan

    :param true_sm: True soil moisture values
    :param est_sm: Estimated soil moisture values
    :param axis: axis of summation
    :return: ubRMSE
    """
    true_sm_mean = np.nanmean(true_sm, axis=axis)
    est_sm_mean = np.nanmean(est_sm, axis=axis)
    unbiased_rmse = np.sqrt(np.nanmean(((est_sm - est_sm_mean) - (true_sm - true_sm_mean)) ** 2))
    return unbiased_rmse


def cal_bias(rmse=None, ub_rmse=None, true_sm=None, est_sm=None, axis: Optional[Union[int, tuple]] = None):
    """
    Calculate bias of soil moisture estimation, either from RMSE and ubRMSE or from the data.

    :param rmse: RMSE
    :param ub_rmse: ubRMSE
    :param true_sm: True soil moisture values
    :param est_sm: Estimated soil moisture values
    :param axis: axis of summation, only if rmse and ubRMSE are None
    :return: Bias
    """

    if true_sm is not None and est_sm is not None:
        rmse = cal_rmse(true_sm, est_sm, axis)
        ub_rmse = cal_unbiased_rmse(true_sm, est_sm, axis)
    if rmse is not None and ub_rmse is not None:
        b = np.sqrt(rmse ** 2 - ub_rmse ** 2)
    else:
        raise RuntimeError('You have to input either rmse and ub_rmse or true_sm and est_sm.')
    return b


def cal_corr_coeff(true_sm, est_sm):
    """
    Calculate correlation coefficient r of soil moisture estimation.

    :param true_sm: True soil moisture values
    :param est_sm: Estimated soil moisture values
    :return:
    """
    r = np.nanmean((est_sm - np.nanmean(est_sm)) * (true_sm - np.nanmean(true_sm))) / (np.nanstd(est_sm) * np.nanstd(true_sm))
    return r

