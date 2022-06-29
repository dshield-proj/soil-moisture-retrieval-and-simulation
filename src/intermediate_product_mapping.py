from util.run_util import check_folder, log_print
from util.save_data import NumpyArrayEncoder
import argparse
import json
import numpy as np
import os
import pandas as pd
import re
"""
  DESCRIPTION
            Generate empty intermediate product of soil moisture performance metrics, 
                         
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-03-03
   Updated  2022-06-22: clean up and add comments
 
   Copyright 2022 University of Southern California  
"""


def gen_optimized_input_file(out_json_path, in_xls_path):
    if in_xls_path is None:
        num_instruments = 4
        df = create_df_operation_modes(num_instruments)
    else:
        if not os.path.isfile(in_xls_path):
            raise FileNotFoundError(f'{in_xls_path} file not found')
        df: pd.DataFrame = pd.read_excel(in_xls_path)
        list_instruments = [int(re.findall('\d+', inst_num_)[0]) for inst_num_ in df.columns]
        num_instruments = len(list_instruments)
    inc_angle_list = [35, 45, 55]
    num_inc_angles = len(inc_angle_list)
    l_band_inc_angle_and_obs = np.zeros((df.shape[0], num_inc_angles), dtype=int)
    p_band_inc_angle_and_obs = np.zeros((df.shape[0], num_inc_angles), dtype=int)
    for irow, row_data in df.iterrows():
        l_band_inc_angle_and_obs[irow, :], p_band_inc_angle_and_obs[irow, :] = operation_mode2num_observ_inc_angle(row_data, num_inc_angles, num_instruments)
    combined_: np.ndarray = np.concatenate((l_band_inc_angle_and_obs, p_band_inc_angle_and_obs), axis=1)
    combined_ = np.unique(combined_, axis=0)
    dict_out = {'l_band': combined_[:, :num_inc_angles],
                'p_band': combined_[:, num_inc_angles:],
                'inc_angle': inc_angle_list}

    with open(out_json_path, 'w') as f:
        json.dump(dict_out, f, cls=NumpyArrayEncoder)
    return out_json_path


def create_df_operation_modes(num_instruments):
    """
    Create list of operation modes, for 4 instruments: 1 & 2 are p_band, 3 & 4 are l_band

    :param num_instruments: number of instruments, only designed for 4 instruments
    :return: Dataframe with operation modes for all instruments
    """
    max_op_mode = 10
    op_modes_list_ = []
    for intst1 in range(max_op_mode):
        for intst2 in range(intst1, max_op_mode):
            for intst3 in range(max_op_mode):
                for intst4 in range(intst3, max_op_mode):
                    op_modes_list_.append([intst1, intst2, intst3, intst4])
    op_modes_list_ = np.array(op_modes_list_, dtype=int)
    col_ = [f'inst{i_}' for i_ in range(1, num_instruments + 1)]
    df = pd.DataFrame(op_modes_list_, columns=col_)
    return df


def gen_input_file_smap(smap_out_json_path):
    inc_angle_list = [40.0]

    dict_out = {'smap': [[1]],
                'inc_angle': inc_angle_list}
    with open(smap_out_json_path, 'w') as f:
        json.dump(dict_out, f, cls=NumpyArrayEncoder)
    return smap_out_json_path


def operation_mode2num_observ_inc_angle(row_data, num_inc_angles, num_instruments):
    num_observ_l_band_row = np.zeros(num_inc_angles)
    num_observ_p_band_row = np.zeros(num_inc_angles)
    op_mode_ = (row_data.to_numpy(dtype=int)[0:num_instruments]).tolist()
    for i_instr in range(0, int(num_instruments/2)):
        if op_mode_[i_instr] == 1:
            num_observ_l_band_row[0] += 1
        elif op_mode_[i_instr] == 2:
            num_observ_l_band_row[1] += 1
        elif op_mode_[i_instr] == 3:
            num_observ_l_band_row[2] += 1
        if op_mode_[i_instr] == 4:
            num_observ_l_band_row[0] += 2
        elif op_mode_[i_instr] == 5:
            num_observ_l_band_row[1] += 2
        elif op_mode_[i_instr] == 6:
            num_observ_l_band_row[2] += 2
        elif op_mode_[i_instr] == 7:
            num_observ_l_band_row[0] += 1
            num_observ_l_band_row[1] += 1
        elif op_mode_[i_instr] == 8:
            num_observ_l_band_row[0] += 1
            num_observ_l_band_row[2] += 1
        elif op_mode_[i_instr] == 9:
            num_observ_l_band_row[1] += 1
            num_observ_l_band_row[2] += 1
    for i_instr in range(int(num_instruments/2), num_instruments):
        if op_mode_[i_instr] == 1:
            num_observ_p_band_row[0] += 1
        elif op_mode_[i_instr] == 2:
            num_observ_p_band_row[1] += 1
        elif op_mode_[i_instr] == 3:
            num_observ_p_band_row[2] += 1
        if op_mode_[i_instr] == 4:
            num_observ_p_band_row[0] += 2
        elif op_mode_[i_instr] == 5:
            num_observ_p_band_row[1] += 2
        elif op_mode_[i_instr] == 6:
            num_observ_p_band_row[2] += 2
        elif op_mode_[i_instr] == 7:
            num_observ_p_band_row[0] += 1
            num_observ_p_band_row[1] += 1
        elif op_mode_[i_instr] == 8:
            num_observ_p_band_row[0] += 1
            num_observ_p_band_row[2] += 1
        elif op_mode_[i_instr] == 9:
            num_observ_p_band_row[1] += 1
            num_observ_p_band_row[2] += 1
    return num_observ_l_band_row, num_observ_p_band_row


def fill_op_mode_from_intermediate_product(in_xls_path: str, intermediate_prod_json_path: str, out_xls_path: str):
    if not os.path.isfile(in_xls_path):
        raise FileNotFoundError(f'{in_xls_path} file not found')
    df: pd.DataFrame = pd.read_excel(in_xls_path)
    list_instruments = [int(re.findall('\d+', inst_num_)[0]) for inst_num_ in df.columns]
    num_instruments = len(list_instruments)
    inc_angle_list = [35, 45, 55]
    num_inc_angles = len(inc_angle_list)
    out_json_data = {}
    with open(intermediate_prod_json_path, 'r') as f:
        inter_json_data = json.load(f)
    list_scenarios_names = [key for key in inter_json_data.keys() if key not in ['input_param']]
    for scenario_ in list_scenarios_names:
        df[f'{scenario_}_rmse'] = [np.nan] * df.shape[0]
        df[f'{scenario_}_ubrmse'] = [np.nan] * df.shape[0]
        df[f'{scenario_}_std'] = [np.nan] * df.shape[0]
        df[f'{scenario_}_bias'] = [np.nan] * df.shape[0]
        out_json_data[scenario_] = list()
    df[f'total_time_sec'] = [np.nan] * df.shape[0]
    sensor_names_list = ['l_band', 'p_band']
    for irow, row_data in df.iterrows():
        num_observ_l_band_row, num_observ_p_band_row = operation_mode2num_observ_inc_angle(row_data, num_inc_angles, num_instruments)
        row_param = {'l_band': num_observ_l_band_row,
                     'p_band': num_observ_p_band_row}
        is_sel_row = np.zeros((len(inter_json_data[list_scenarios_names[0]]), len(sensor_names_list)), dtype=bool)
        for ikey, key in enumerate(sensor_names_list):
            is_sel_row[:, ikey] = np.prod(np.array(inter_json_data['input_param'][key]) == row_param[key], axis=1)
        idx_ = np.where(np.prod(is_sel_row, axis=1))[0]
        if idx_.size <= 0:
            log_print(f'row {irow}: no performance data for {row_data}')
        elif idx_.size == 1:
            for scenario_ in list_scenarios_names:
                for key1, key2 in zip(['rmse', 'ubrmse', 'std', 'bias'], ['sm_rmse', 'sm_ubrmse', 'est_sm_std', 'sm_bias']):
                    df.at[irow, f'{scenario_}_{key1}'] = inter_json_data[scenario_][idx_[0]][key2]
            df.at[irow, 'total_time_sec'] = sum([np.array(inter_json_data[scenario_][idx_[0]]['retrieval_duration_sec']).sum() for scenario_ in list_scenarios_names])
        else:
            raise RuntimeError(f'Expected single index value, got {idx_}')

    df.to_excel(out_xls_path, index=False)
    return out_xls_path


def create_parser():
    global pars
    # Create the parser
    pars = argparse.ArgumentParser(description='Intermediate product mapping')
    # Add the arguments
    pars.add_argument('-o', '--out_folder', nargs='?', metavar='output_main_path', default=None, type=str, help='path of the config file')
    pars.add_argument('--op_mode_xls_path', nargs='?', default=None, type=str, help='path to xls file that contain instruments with operation mode')
    pars.add_argument('--inter_json_file', nargs='?', default=None, type=str, help='intermediate product json file name [input/output]')
    pars.add_argument('--metric_xls_file', nargs='?', default=None, type=str, help='soi moisture performance with operation mode file name [output]')
    pars.add_argument('--gen_empty_inter_file', action='store_true', default=False,
                      help='generate an empty intermediate json file from operation mode file')
    pars.add_argument('--gen_empty_inter_file_smap', action='store_true', default=False,
                      help='generate an empty intermediate json file for smap radar')

    return pars


if __name__ == '__main__':
    pars = create_parser()
    args = pars.parse_args()
    if args.out_folder is None:
        args.out_folder = os.path.dirname(args.op_mode_xls_path)
    check_folder(args.out_folder)
    if args.inter_json_file is None:
        args.inter_json_file = 'instruments_inc_angles_and_observations.json'
    out_json_path = os.path.join(args.out_folder, args.inter_json_file)
    if args.gen_empty_inter_file:
        gen_optimized_input_file(out_json_path, args.op_mode_xls_path)
    elif args.gen_empty_inter_file_smap:
        out_smap_json_path = os.path.join(args.out_folder, 'smap_observations_input.json')
        gen_input_file_smap(out_smap_json_path)
    else:
        if args.metric_xls_file is None:
            args.metric_xls_file = os.path.basename(args.op_mode_xls_path).replace('.xlsx', '_with_rmse.xlsx')
        metric_xls_path = os.path.join(args.out_folder, args.metric_xls_file)
        fill_op_mode_from_intermediate_product(args.op_mode_xls_path, out_json_path, metric_xls_path)
