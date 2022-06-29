from helpper.instruments_data import list_inc_angles2op_mode
import astropy.time
import datetime
import fnmatch
import json
import os
import pandas as pd

"""
  DESCRIPTION
            Input data processing for the gen_science_data.py file.
            This is for processing input data from experiments
             
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-06-22
   Updated  
 
   Copyright 2022 University of Southern California
"""


def get_list_biomes(experiment_folder: str):
    """

    Get list of biomes from the common.json file

    :param experiment_folder: experiment folder path
    :return:
    """

    common_json = os.path.join(experiment_folder, 'common', 'common.json')
    with open(common_json) as f:
        common_data = json.load(f)
    biomes = [item['id'] for item in common_data['biome']]
    return biomes


def pointing_options2inc_angle(pointing_option, pointing_options_dic):
    """

    get science data simulation incidence angles from pointing_option using dic from get_pointing_options_dic()

    :param pointing_option: pointing option number
    :param pointing_options_dic: from get_pointing_options_dic()
    :return: science data incidence angle
    """
    inc_angles_bin_id_list = [option_['incAngleBinId'] for option_ in pointing_options_dic['pointingOption'] if option_['id'] == f'P{int(pointing_option)}']
    if len(inc_angles_bin_id_list) != 1:
        raise RuntimeError(f'No angle found or multiple angles found, len inc angles list {inc_angles_bin_id_list}')

    if inc_angles_bin_id_list[0] == 'AX':
        raise RuntimeError('Un supported incidence angle')

    inc_angle = [float(bin_id['bin'][0:2]) for bin_id in pointing_options_dic['incAngleBin'] if inc_angles_bin_id_list[0] == bin_id['id']]
    if len(inc_angle) != 1:
        raise RuntimeError(f'Error')
    return inc_angle[0]


def extract_epoch_id(plan_csv_file) -> datetime.datetime:
    """
    Extract epoch id from plan csv file

    :param plan_csv_file: plan csv file path
    :return: epoch start time
    """
    epoch_jd = pd.read_csv(plan_csv_file, skiprows=[0, 1], nrows=1, header=None).astype(str)  # 3rd row contains the epoch
    epoch_jd = float(epoch_jd[0][0].split()[4])
    epoch = astropy.time.Time(epoch_jd, format='jd', scale='utc').datetime  # get the epoch as a datetime object from the Julian Date
    return epoch


def extracting_operation_modes(pointing_options: list, pointing_potions_dic: dict, gp_id: int):
    """
    Extract instruments operation modes from pointing_options list

    :param pointing_options: list of satellites pointing options
    :param pointing_potions_dic: from get_pointing_options_dic()
    :param gp_id: GP id, only for printing errors
    :return:
    """

    inc_angles_ = []
    for pointing_op_ in pointing_options:
        inc_angles_.append(int(pointing_options2inc_angle(pointing_op_, pointing_potions_dic)))
    if len(inc_angles_) > 2:
        print(f'GP: {gp_id}: Only two incidence angles are supported, got {inc_angles_}, only keeping 2')
        inc_angles_ = inc_angles_[:2]
    operation_mode = list_inc_angles2op_mode(inc_angles_)
    if operation_mode < 0:
        print(f'GP: {gp_id} : P-BAND, Not supported operation mode: {inc_angles_}')

    return operation_mode


def unpack_gb_list(gp_string):
    gp_list = gp_string[1:-1]  # leave out the first and last characters which would be '[' and ']'
    gp_list = gp_list.split(' ')
    gp_list = list(map(int, gp_list))
    return gp_list


def get_plans_cvs_files(plans_folder):
    smap_fn_pattern = f's*prettyPlan.csv'
    files_path_list = list()
    for root, dirs, files in os.walk(plans_folder):
        for name in files:
            if fnmatch.fnmatch(name, smap_fn_pattern):
                files_path_list.append(os.path.join(root, name))

    return files_path_list


def extract_metadata_from_grid_csv(gp_num, gp_df=None) -> dict:
    """
    Get GP info from gp_df or csv file in "input_parameters/grid.csv".
    Note if GP is out of bound, you'll get error

    :param gp_num: GP number
    :param gp_df: Dataframe of grid.csv, if None, the code will read input_parameters/grid.csv file
    :return: {gp, lat, lon, row, col, igbp_class, biome_id}
    """
    if gp_df is None:
        gird_csv_file_path = os.path.join('input_parameters', 'grid.csv')
        gp_df = pd.read_csv(gird_csv_file_path)

    sel_data = gp_df[gp_df['GP index'] == gp_num].reset_index()
    return {'gp': int(sel_data['GP index']),
            'lat': float(sel_data['lat[deg]']),
            'lon': float(sel_data['lon[deg]']),
            'row': int(sel_data['row']),
            'col': int(sel_data['col']),
            'igbp_class': int(sel_data['IGBP Class']),
            'biome_id': sel_data['Biome ID'][0]}


def get_pointing_options_dic(experiment_folder: str) -> dict:
    """

    Get pointing options from common.json file
    :param experiment_folder: experiment folder path
    :return: {'incAngleBin', 'pointingOption'}
    """

    common_json = os.path.join(experiment_folder, 'common', 'common.json')
    with open(common_json) as f:
        common_data = json.load(f)

    inc_angle_dic = common_data['incAngleBin']
    pointing_opt_dic = common_data['pointingOption']
    return {'incAngleBin': inc_angle_dic, 'pointingOption': pointing_opt_dic}
