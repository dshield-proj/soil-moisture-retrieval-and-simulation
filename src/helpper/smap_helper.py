import datetime
import fnmatch
import os
import numpy as np
"""
  DESCRIPTION
            Processing smap data
                         
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-06-22
   Updated  
 
   Copyright 2022 University of Southern California
"""

smap_l4_default_folder = os.environ.get('SMAP_L4_PATH')
NO_SM_VAL = -9999.0  # soil moisture fill value, this when there is no value from SMAP


def get_smap_file_name(data_time: datetime.datetime, smap_data_folder=None):
    """
    Get file path of closest SMAP data time.
    The SMAP data is the SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture.

    :param data_time: soil moisture time
    :param smap_data_folder: SMAP data folder path, if None will get the folder from $SMAP_L4_PATH
    :return: file path
    """
    if smap_data_folder is None:
        smap_data_folder = smap_l4_default_folder

    smap_data_hour = int(np.round(data_time.hour / 3.0) * 3)
    smap_fn_pattern = f'*{data_time.year:04d}{data_time.month:02d}{data_time.day:02d}T{smap_data_hour:02d}0000*.h5'
    list_smap_files = list()
    for root, dirs, files in os.walk(smap_data_folder):
        for name in files:
            if fnmatch.fnmatch(name, smap_fn_pattern):
                list_smap_files.append(os.path.join(root, name))
    if not list_smap_files:
        raise FileNotFoundError(f'Cannot find {smap_fn_pattern}\n download the file from https://nsidc.org/data/SPL4SMAU/versions/6')
    if len(list_smap_files) > 1:
        raise ValueError(f'Expected one SMAP file, got {len(list_smap_files)}')
    return list_smap_files[0]
