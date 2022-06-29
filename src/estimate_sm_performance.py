from core.VegMultiRcsSmRetrieval import VegRcsSmRetrieval
from helpper.instruments_data import get_sensor_using_inc_angle
from helpper.scenarios_data import get_scenario
from tqdm import tqdm
from typing import Optional
from util.plotting import plt_sm_error_history, plt_hist_all_scenario_metrics, plt_hist_single_scenario_metrics
from util.run_util import log_print, convert_timedelta2str, check_folder
from util.save_data import NumpyArrayEncoder
from util.soil_moisture_stats import cal_rmse, cal_unbiased_rmse, cal_bias
import argparse
import copy
import datetime as dt
import json
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import random
import string

"""
  DESCRIPTION
            Generate soil moisture retrieval performance metric.  
            
                         
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-02-03
   Updated  2022-06-22: Clean up and add comments
 
   Copyright 2022 University of Southern California
"""

default_optimizer_param = {'iter': 15,
                           'n_t': 5,
                           'n_s': 5,
                           'n_mul': 5,
                           'temper': 10.0,
                           'r_t': 0.85,
                           'c_j': 2.5,
                           'step_lim': 1e-2,
                           'f_stop': 1e-2,
                           'overflow_cost': 100,
                           'overflow_option': 2,
                           'upper_limit': 0.8,
                           'lower_limit': 0.0001,
                           'max_f_eval': None,
                           'silent_mode': True,
                           'soil_clay_prce': None,
                           'num_lyr': 1,
                           'ground_lyr_dist': None,
                           'cost_fun': 'l2'}


def run_multi_instruments_retrieval(sensors_list: list[dict], scenario_name: str, veg_param_path: str, num_trials: int, md_rtrv_opt: Optional[dict],
                                    out_folder_path: str, dry_run: bool = False, print_prog=False, tf_save_error_history_fig=True,
                                    history_fig_rand_tag: Optional[str] = None):
    parallel_ = True
    if md_rtrv_opt is None:
        md_rtrv_opt = copy.copy(default_optimizer_param)

    scenario = get_scenario(scenario_name, veg_param_path)
    est_sm = np.ones((len(scenario['sm_list']), num_trials)) * np.nan
    n_f_eval = np.zeros((len(scenario['sm_list']), num_trials))
    rtrvl_duration = np.zeros((len(scenario['sm_list']), num_trials))
    cost_fun = np.zeros((len(scenario['sm_list']), num_trials))
    sm_error_history_list = [[None for _ in range(num_trials)] for _ in range(len(scenario['sm_list']))]
    if scenario_name in ['las_cruces']:  # remove cross-pol for surfaces without vegetation
        for i_sensor, sensors in enumerate(sensors_list):
            for i_pol, pol in enumerate(sensors['polarization']):
                if pol in ['vh', 'hv']:
                    sensors_list[i_sensor]['polarization'].pop(i_pol)

    if not sensors_list or dry_run:
        sm_rmse, sm_ubrmse, sm_bias, est_sm_std = np.nan, np.nan, np.nan, np.nan
    else:
        if parallel_:
            with mp.Pool() as pool:
                args = ((md_rtrv_opt, sensors_list, scenario, sm, i_sm, i_trial, print_prog)
                        for i_sm, sm in enumerate(scenario['sm_list']) for i_trial in range(num_trials))

                for i_, (est_sm_, cost_fun_, n_f_eval_, rtrvl_duration_, sm_error_hist_, i_sm, i_trial) in enumerate(
                        pool.starmap(_single_scenario_sim, args)):
                    est_sm[i_sm, i_trial] = est_sm_
                    n_f_eval[i_sm, i_trial] = n_f_eval_
                    cost_fun[i_sm, i_trial] = cost_fun_
                    rtrvl_duration[i_sm, i_trial] = rtrvl_duration_
                    sm_error_history_list[i_sm][i_trial] = sm_error_hist_

        else:
            for i_sm, sm in enumerate(scenario['sm_list']):
                for i_trial in range(num_trials):
                    est_sm_, cost_fun_, n_f_eval_, rtrvl_duration_, sm_error_hist_, i_sm, i_trial = _single_scenario_sim(
                        md_rtrv_opt, sensors_list, scenario, sm, i_sm, i_trial, print_prog)

                    est_sm[i_sm, i_trial] = est_sm_
                    n_f_eval[i_sm, i_trial] = n_f_eval_
                    cost_fun[i_sm, i_trial] = cost_fun_
                    rtrvl_duration[i_sm, i_trial] = rtrvl_duration_
                    sm_error_history_list[i_sm][i_trial] = sm_error_hist_

        true_sm = np.tile(np.array(scenario['sm_list'])[..., np.newaxis], (1, num_trials))
        est_sm_std = np.mean(np.std(est_sm, axis=1), axis=0)
        sm_rmse = cal_rmse(true_sm, est_sm, axis=(0, 1))
        sm_ubrmse = cal_unbiased_rmse(true_sm, est_sm, axis=(0, 1))
        sm_bias = cal_bias(sm_rmse, sm_ubrmse)
        if tf_save_error_history_fig:
            sm_list = scenario['sm_list']
            img_save_tag = f'{scenario_name}_num_sensors_{len(sensors_list)}'
            plt_sm_error_history(sm_error_history_list, sm_list, num_trials, img_save_tag, history_fig_rand_tag, out_folder_path,
                                 tf_save_error_history_fig)

    out_dict = {'sensors_list': sensors_list,
                'scenario_name': scenario_name,
                'num_trials': num_trials,
                'retrieval_duration_sec': rtrvl_duration,
                'true_sm': scenario['sm_list'],
                'est_sm': est_sm,  # dim 0: soil moisture, dim 1: Monte Carlo trials
                'sm_rmse': sm_rmse,
                'sm_ubrmse': sm_ubrmse,
                'sm_bias': sm_bias,
                'est_sm_std': est_sm_std,
                'n_f_eval': n_f_eval,
                'cost_fun': cost_fun,
                'scenario_info': scenario,
                'sm_error_history': sm_error_history_list}
    return out_dict


def _single_scenario_sim(md_rtrv_opt: dict, sensors_list: list[dict], scenario: dict, sm: float, i_sm: int, i_trial: int, print_prog: bool = False):
    veg = copy.copy(scenario['vegetation'])
    veg['soilcontent'] = sm
    sim_veg_obj = VegRcsSmRetrieval(veg, sensors_list, 1, md_rtrv_opt['iter'], md_rtrv_opt['n_t'], md_rtrv_opt['n_s'],
                                    md_rtrv_opt['n_mul'], md_rtrv_opt['temper'], md_rtrv_opt['r_t'], md_rtrv_opt['c_j'],
                                    md_rtrv_opt['step_lim'], md_rtrv_opt['f_stop'], md_rtrv_opt['overflow_cost'], md_rtrv_opt['upper_limit'],
                                    md_rtrv_opt['lower_limit'], md_rtrv_opt['overflow_option'], md_rtrv_opt['max_f_eval'], False,
                                    md_rtrv_opt['silent_mode'], md_rtrv_opt['cost_fun'])
    t = dt.datetime.now()
    sim_veg_obj.run_multi_sim_ann()
    dur_ = dt.datetime.now() - t
    duration_sec = dur_.total_seconds()
    est_sm_ = sim_veg_obj.x_opt[0]
    n_f_eval_ = sim_veg_obj.n_feval
    cost_fun_ = sim_veg_obj.f_opt
    rtrvl_duration_ = duration_sec
    sm_error_history = sim_veg_obj.get_sm_error_history()
    if print_prog:
        log_print(f'finished trial {i_trial + 1} for sm idx {i_sm + 1} in {convert_timedelta2str(dur_)}')

    return est_sm_, cost_fun_, n_f_eval_, rtrvl_duration_, sm_error_history, i_sm, i_trial


def run_multi_instruments_retrieval_from_json(in_json_path: str, list_scenarios_names: list[str], veg_param_path: str, num_trials: int,
                                              out_folder_path: str, xls_out_file_name: str, debug_mode: bool = False, dry_run: bool = False,
                                              tf_save_error_history_fig: bool = False):
    md_rtrv_opt = copy.copy(default_optimizer_param)
    if debug_mode:
        md_rtrv_opt['max_f_eval'] = 10

    if not os.path.isfile(in_json_path):
        raise FileNotFoundError(f'{in_json_path} file not found')
    with open(in_json_path, 'r') as f:
        sensor_freq_inc_angle: dict = json.load(f)
    for key in sensor_freq_inc_angle.keys():
        sensor_freq_inc_angle[key] = np.array(sensor_freq_inc_angle[key])
    sensor_names_list = [key for key in sensor_freq_inc_angle.keys() if len(sensor_freq_inc_angle[key].shape) > 1]
    num_rows = sensor_freq_inc_angle[sensor_names_list[0]].shape[0]

    out_xls_file_path = os.path.join(out_folder_path, xls_out_file_name)
    json_file_name = xls_out_file_name.replace('.xlsx', '.json')
    json_file_path = os.path.join(out_folder_path, json_file_name)

    pbar = tqdm(total=len(list_scenarios_names) * sensor_freq_inc_angle[sensor_names_list[0]].shape[0], desc='Simulating SM retrieval ', unit='sim')
    out_json_data = {'input_param': {}}
    df = pd.DataFrame({})
    for key in sensor_names_list:
        out_json_data['input_param'][f'{key}'] = sensor_freq_inc_angle[key]
        for i_ang, angle_ in enumerate(sensor_freq_inc_angle['inc_angle']):
            df[f'sensor_{key}_ang_{angle_}'] = sensor_freq_inc_angle[key][:, i_ang]
            out_json_data['input_param'][f'sensor_{key}_ang_{angle_}'] = sensor_freq_inc_angle[key][:, i_ang]
    for scenario_ in list_scenarios_names:
        df[f'{scenario_}_rmse'] = [np.nan] * num_rows
        df[f'{scenario_}_ubrmse'] = [np.nan] * num_rows
        df[f'{scenario_}_std'] = [np.nan] * num_rows
        df[f'{scenario_}_bias'] = [np.nan] * num_rows
        out_json_data[scenario_] = list()
    for irow in range(num_rows):
        sensors_list = []
        for sensor_name in sensor_names_list:
            for i_ang, angle_ in enumerate(sensor_freq_inc_angle['inc_angle']):
                if sensor_freq_inc_angle[sensor_name][irow, i_ang] > 0:
                    sensors_list.append(get_sensor_using_inc_angle(sensor_name, angle_, sensor_freq_inc_angle[sensor_name][irow, i_ang]))

        for scenario_ in list_scenarios_names:
            single_sim = run_multi_instruments_retrieval(sensors_list, scenario_, veg_param_path, num_trials, md_rtrv_opt, out_folder_path, dry_run,
                                                         False, tf_save_error_history_fig)
            pbar.update()
            df.loc[irow, f'{scenario_}_rmse'] = single_sim['sm_rmse']
            df.loc[irow, f'{scenario_}_std'] = single_sim['est_sm_std']
            df.loc[irow, f'{scenario_}_ubrmse'] = single_sim['sm_ubrmse']
            df.loc[irow, f'{scenario_}_bias'] = single_sim['sm_bias']

            out_json_data[scenario_].append(single_sim)

            df.to_excel(out_xls_file_path, index=False)
            with open(json_file_path, 'w') as f:
                json.dump(out_json_data, f, cls=NumpyArrayEncoder)
    pbar.close()
    return xls_out_path


def run_multi_instruments_retrieval_from_json_standalone(in_json_path: str, list_scenarios_names: list[str], veg_param_path: str, num_trials: int,
                                                         out_folder_path: str, xls_out_file_name: str, irow: int, debug_mode: bool = False,
                                                         dry_run: bool = False, skip_if_exist: bool = False, tf_save_error_history_fig: bool = False,
                                                         verbose: bool = False):
    md_rtrv_opt = copy.copy(default_optimizer_param)
    if debug_mode:
        md_rtrv_opt['max_f_eval'] = 10

    st_time_ = dt.datetime.now()
    if verbose:
        log_print(f'Start row {irow}')

    if not os.path.isfile(in_json_path):
        raise FileNotFoundError(f'{in_json_path} file not found')
    with open(in_json_path, 'r') as f:
        sensor_freq_inc_angle: dict = json.load(f)
    for key in sensor_freq_inc_angle.keys():
        sensor_freq_inc_angle[key] = np.array(sensor_freq_inc_angle[key])
    sensor_names_list = [key for key in sensor_freq_inc_angle.keys() if len(sensor_freq_inc_angle[key].shape) > 1]
    num_rows = sensor_freq_inc_angle[sensor_names_list[0]].shape[0]

    out_json_data = {'input_param': {}}
    df = pd.DataFrame({})
    for key in sensor_names_list:
        out_json_data['input_param'][f'{key}'] = [sensor_freq_inc_angle[key][irow, :]]
        for i_ang, angle_ in enumerate(sensor_freq_inc_angle['inc_angle']):
            df[f'sensor_{key}_ang_{angle_}'] = [sensor_freq_inc_angle[key][irow, i_ang]]
            out_json_data['input_param'][f'sensor_{key}_ang_{angle_}'] = [sensor_freq_inc_angle[key][irow, i_ang]]
    for scenario_ in list_scenarios_names:
        df[f'{scenario_}_rmse'] = [np.nan]
        df[f'{scenario_}_ubrmse'] = [np.nan]
        df[f'{scenario_}_std'] = [np.nan]
        df[f'{scenario_}_bias'] = [np.nan]
        out_json_data[scenario_] = list()

    fn_no_ext_ = xls_out_file_name.split('.xls')[0]
    if os.path.isfile(os.path.join(out_folder_path, fn_no_ext_ + '.json')) and skip_if_exist:
        with open(os.path.join(out_folder_path, fn_no_ext_ + '.json'), 'r') as f:
            data_ = json.load(f)
            is_sel_row = np.zeros((len(data_[list_scenarios_names[0]]), len(sensor_names_list)), dtype=bool)
            for ikey, key in enumerate(sensor_names_list):
                is_sel_row[:, ikey] = np.prod(np.array(data_['input_param'][key]) == sensor_freq_inc_angle[key][irow, :], axis=1)
        idx_ = np.where(np.prod(is_sel_row, axis=1))[0]
        del data_, is_sel_row
        if idx_.size > 0:
            log_print(f'data point exist in row {idx_[0]}')
            return True

    # create file with random suffix
    while True:
        rand_tag = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
        rand_tag = f'{irow:04d}' + rand_tag[4:]

        json_file_name = f'{fn_no_ext_}_{rand_tag}.json'
        if not os.path.isfile(json_file_name):
            break
    xls_out_file_name = f'{fn_no_ext_}_{rand_tag}.xlsx'

    out_xls_file_path = os.path.join(out_folder_path, xls_out_file_name)
    json_file_path = os.path.join(out_folder_path, json_file_name)
    if verbose:
        log_print(f'json file path: {json_file_path}')
        log_print(f'xls file path: {out_xls_file_path}')

    sensors_list = []
    for sensor_name in sensor_names_list:
        for i_ang, angle_ in enumerate(sensor_freq_inc_angle['inc_angle']):
            if sensor_freq_inc_angle[sensor_name][irow, i_ang] > 0:
                sensors_list.append(get_sensor_using_inc_angle(sensor_name, angle_, sensor_freq_inc_angle[sensor_name][irow, i_ang]))

    for scenario_ in list_scenarios_names:
        t_ = dt.datetime.now()
        single_sim = run_multi_instruments_retrieval(sensors_list, scenario_, veg_param_path, num_trials, md_rtrv_opt, out_folder_path, dry_run,
                                                     verbose, tf_save_error_history_fig, rand_tag)
        df.loc[0, f'{scenario_}_rmse'] = single_sim['sm_rmse']
        df.loc[0, f'{scenario_}_std'] = single_sim['est_sm_std']
        df.loc[0, f'{scenario_}_ubrmse'] = single_sim['sm_ubrmse']
        df.loc[0, f'{scenario_}_bias'] = single_sim['sm_bias']

        out_json_data[scenario_].append(single_sim)

        df.to_excel(out_xls_file_path, index=False)
        with open(json_file_path, 'w') as f:
            json.dump(out_json_data, f, cls=NumpyArrayEncoder)
        dur_ = dt.datetime.now() - t_
        if verbose:
            log_print(f'{scenario_} duration: {convert_timedelta2str(dur_)}')

    if verbose:
        log_print(f'Finished row {irow} in {convert_timedelta2str(dt.datetime.now() - st_time_)}')
    return xls_out_path


def join_standalone_files(out_folder_path: str, xls_out_file_name: str, list_scenarios: list[str]):
    img_out_path = os.path.join(out_folder_path, 'img')
    check_folder(img_out_path)

    file_name_base = xls_out_file_name.split('.')[0]
    list_json_files = []
    for f in os.listdir(out_folder_path):
        f_path = os.path.join(out_folder_path, f)
        if os.path.isfile(f_path) and f_path.endswith('.json') and '_'.join(f.split('.json')[0].split('_')[:-1]) == file_name_base \
                and len(f.split('_')[-1]) == 13:
            list_json_files.append(f)
    if not list_json_files:
        return None
    list_json_files = np.sort(list_json_files)
    unique_files = np.sort(np.unique(np.array([file[0:len(file_name_base) + 5] for file in list_json_files])))
    json_data = {}
    for file_base in tqdm(unique_files, desc='Joining files: ', unit='file'):
        data_ = []
        for filename_ in list_json_files[np.char.rfind(list_json_files, file_base) == 0]:
            file_path = os.path.join(out_folder_path, str(filename_))
            with open(file_path, 'r') as f:
                try:
                    data_.append(json.load(f))
                except json.decoder.JSONDecodeError as e:
                    log_print(f'Error in reading file {file_path}')
        scenarios_exist = np.zeros((len(list_scenarios), len(data_)), dtype=bool)
        for idata, single_data in enumerate(data_):
            for i_scenm, scenario in enumerate(list_scenarios):
                try:
                    scenarios_exist[i_scenm, idata] = bool(len(single_data[scenario]))
                except (KeyError, ValueError) as e:
                    scenarios_exist[i_scenm, idata] = False
        if not np.all(np.sum(scenarios_exist, axis=1)):
            log_print(f'data of {file_base.split("_")[-1]} is not complete')
        if len(data_) > 1:
            json_sf_ = {f'{scenario_}': list() for scenario_ in list_scenarios}
            if np.sum(np.all(scenarios_exist, axis=0)) > 1:  # More than one full file
                log_print(f'Error: found multiple full files, {file_base}')
            elif np.sum(np.all(scenarios_exist, axis=0)) == 1:  # only one full file
                json_sf_ = data_[np.where(np.all(scenarios_exist, axis=0))[0][0]]
            elif np.all(np.sum(scenarios_exist, axis=1)):
                first_arg = np.argmax(np.sum(scenarios_exist, axis=0))
                for key in data_[first_arg].keys():
                    json_sf_[key] = data_[first_arg][key]
                for ikey, key in enumerate(json_sf_.keys()):
                    if not json_sf_[key]:
                        i_data_ = np.argwhere(scenarios_exist[ikey, :])[0][0]
                        json_sf_[key] = data_[i_data_][key]
            else:
                log_print(f'there are multiple partial files, {file_base}')
        else:  # one file exist
            json_sf_ = data_[0]

        for key in json_sf_.keys():
            if json_data.get(key) is None:
                json_data[key] = json_sf_[key]
            else:
                if key == 'input_param':
                    for key2 in json_sf_['input_param'].keys():
                        if 'sensor' not in key2 and len(np.array(json_sf_[key][key2]).shape) < 2:  # fix a bug in the code
                            if len(np.array(json_data[key][key2]).shape) < 2:
                                json_data[key][key2] = [json_data[key][key2]]
                            json_data[key][key2] += [json_sf_[key][key2]]
                        else:
                            json_data[key][key2] += json_sf_[key][key2]
                else:
                    json_data[key] += json_sf_[key]
    # check data len
    data_len = np.zeros(len(json_data))
    for ikey, key in enumerate(json_data.keys()):
        if key == 'input_param':
            len_temp = np.unique(np.array([len(param) for param in json_data['input_param'].values()]))
            if len(len_temp) == 1:
                data_len[ikey] = len_temp
        else:
            data_len[ikey] = len(json_data[key])
    if len(np.unique(data_len)) > 1:
        raise RuntimeError('Expected all the data to be the same size. Got different sizes')

    out_json_file_path = os.path.join(out_folder_path, file_name_base + '.json')
    out_xls_file_path = os.path.join(out_folder_path, file_name_base + '.xlsx')
    with open(out_json_file_path, 'w') as f:
        json.dump(json_data, f, cls=NumpyArrayEncoder)

    df = pd.DataFrame({})
    for key in json_data['input_param'].keys():
        if 'sensor_' in key:
            df[key] = json_data['input_param'][key]
    num_rows = len(json_data[list_scenarios[0]])
    for scenario_ in list_scenarios:
        df[f'{scenario_}_rmse'] = [np.nan] * num_rows
        df[f'{scenario_}_ubrmse'] = [np.nan] * num_rows
        df[f'{scenario_}_std'] = [np.nan] * num_rows
        df[f'{scenario_}_bias'] = [np.nan] * num_rows
        for irow in range(num_rows):
            df.loc[irow, f'{scenario_}_rmse'] = json_data[scenario_][irow]['sm_rmse']
            df.loc[irow, f'{scenario_}_std'] = json_data[scenario_][irow]['est_sm_std']
            df.loc[irow, f'{scenario_}_ubrmse'] = json_data[scenario_][irow]['sm_ubrmse']
            df.loc[irow, f'{scenario_}_bias'] = json_data[scenario_][irow]['sm_bias']

    analysis_ = {}
    for scenario in list_scenarios:
        for _stat_merit in ['rmse', 'ubrmse', 'bias']:
            max_id = df[f'{scenario}_{_stat_merit}'].idxmax()
            min_id = df[f'{scenario}_{_stat_merit}'].idxmin()
            analysis_[f'{scenario}_{_stat_merit}'] = [df[f'{scenario}_{_stat_merit}'][max_id],
                                                      df[f'{scenario}_{_stat_merit}'][min_id],
                                                      df[f'{scenario}_{_stat_merit}'].mean()]
            analysis_[f'{scenario}_{_stat_merit}_inst'] = [' '.join(f'{id_}' for id_ in df.iloc[max_id, 0:4].to_numpy(dtype=int)),
                                                           ' '.join(f'{id_}' for id_ in df.iloc[min_id, 0:4].to_numpy(dtype=int)), '']
            stat_img_save_tag = f'{_stat_merit}_{file_name_base}_{scenario}'
            plt_hist_single_scenario_metrics(df[f'{scenario}_{_stat_merit}'].to_numpy(), _stat_merit, stat_img_save_tag, img_out_path)
    plt.close('all')
    plt_hist_all_scenario_metrics(df, list_scenarios_names, file_name_base, img_out_path)
    plt.close('all')

    analysis_df = pd.DataFrame(analysis_)
    analysis_df.index = ['max', 'min', 'mean']
    writer = pd.ExcelWriter(out_xls_file_path)
    df.to_excel(writer, sheet_name='data', index=False)
    analysis_df.T.to_excel(writer, sheet_name='analysis')
    writer.close()

    return out_xls_file_path


if __name__ == '__main__':
    # Create the parser
    pars = argparse.ArgumentParser(description='soil moisture retrieval performance metrics')

    # Add the arguments
    pars.add_argument('-g', '--veg_param_path', nargs='?', type=str, help='path to the vegetation parameters folder')
    pars.add_argument('-o', '--out_folder', nargs='?', metavar='output_main_path', default=None, type=str, help='path of the config file')
    pars.add_argument('--num_trials', nargs='?', metavar='num_trials', default=10, type=int, help='for each sample, how many trials?')
    pars.add_argument('-d', '--row_idx', nargs='+', metavar='ddm_idx', type=int, help='Enter row idx, leave it empty to go over all rows',
                      default=None)
    pars.add_argument('--standalone', action='store_true', default=False, help='if Ture will save the results in folders with random tag')
    pars.add_argument('--dry_run', action='store_true', default=False, help='If true, no retrieval is performed')
    pars.add_argument('--debug', '--debugging', action='store_true', default=False, help='limit Num of fun eval')
    pars.add_argument('--in_json_file', nargs='?', default='', type=str, help='path to xls file that contain instruments with operation mode')
    pars.add_argument('--scenario', nargs='+', type=str, default=None, help='list of simulation scenario')
    pars.add_argument('--out_xls_name', nargs='?', type=str, help='out xls file name, it should end with .xlsx')
    pars.add_argument('--join_files', action='store_true', default=False, help='join the multiple json files produced by --standalone')
    pars.add_argument('--plt_hist', action='store_true', default=False, help='plot histogram of the error. Only works with --join_files')
    pars.add_argument('--skip_if_exist', action='store_true', default=False, help='only for standalone, skip if results exist')
    pars.add_argument('--plt_error_history', action='store_true', default=False, help='plot error per iteration')
    pars.add_argument('--verbose', action='store_true', default=False, help='Print details in the screen')

    args = pars.parse_args()
    xls_out_path = os.path.join(args.out_folder, args.out_xls_name)

    check_folder(args.out_folder)

    veg_parameters_folder = args.veg_param_path
    list_scenarios_names = args.scenario
    if list_scenarios_names is None:
        list_scenarios_names = ['walnut_gulch', 'tonzi_ranch', 'metolius', 'las_cruces', 'yanco']
    if args.join_files:
        join_standalone_files(args.out_folder, args.out_xls_name, list_scenarios_names)
    else:
        if args.standalone:
            if args.row_idx is None:
                log_print('No row was selected, running all the rows')
                with open(args.in_json_file, 'r') as f:
                    sensor_freq_inc_angle: dict = json.load(f)
                list_len = [len(sensor_freq_inc_angle[key]) for key in sensor_freq_inc_angle.keys() if
                            isinstance(sensor_freq_inc_angle[key], list) and isinstance(sensor_freq_inc_angle[key][0], list)]
                del sensor_freq_inc_angle
                row_list = range(max(list_len))
            else:
                row_list = args.row_idx
            for irow in row_list:
                run_multi_instruments_retrieval_from_json_standalone(args.in_json_file, list_scenarios_names, veg_parameters_folder,
                                                                     args.num_trials, args.out_folder, args.out_xls_name, irow, args.debug,
                                                                     args.dry_run, args.skip_if_exist, args.plt_error_history, args.verbose)
        else:
            run_multi_instruments_retrieval_from_json(args.in_json_file, list_scenarios_names, veg_parameters_folder, args.num_trials,
                                                      args.out_folder, args.out_xls_name, args.debug, args.dry_run, args.plt_error_history)
