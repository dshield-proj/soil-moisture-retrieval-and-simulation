from helpper.download_smap_data import download_smap_l4_3h_9km_grid_data
from helpper.experiment_data_processing import get_list_biomes, extract_epoch_id, extracting_operation_modes, unpack_gb_list, get_plans_cvs_files, \
    extract_metadata_from_grid_csv, get_pointing_options_dic
from helpper.smap_helper import get_smap_file_name, NO_SM_VAL, smap_l4_default_folder
from tqdm import tqdm
from typing import Optional
from util.run_util import check_folder, log_print
from util.save_data import NumpyArrayEncoder
import argparse
import datetime
import datetime as dt
import h5py
import json
import numpy as np
import os
import pandas as pd

"""
  DESCRIPTION
            Generate simulated soil moisture from plan files, this is part of running the experiment.
                         
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-06-06
   Updated  2022-06-22: clean up and add comments
 
   Copyright 2022 University of Southern California
"""


def datetime_mean(dates: list[datetime.datetime]) -> Optional[datetime.datetime]:
    """
    Calculate the averaged time in a list of datetime.

    :param dates: list of datetime
    :return: None if empty list, or averaged time
    """
    if len(dates) < 1:
        return None
    any_reference_date = dates[0]
    return any_reference_date + sum([date - any_reference_date for date in dates], datetime.timedelta()) / len(dates)


def generate_out_sim_soil_moisture(experiment_folder: str, experiment_run_id: str, smap_data_folder=None, download_smap_data=True):
    """
    Generate simulated soil moisture data using SMAP L4 soil moisture values and the plans of the experiment.
    results of each 3 hrs will be in a file, the output will be in experiment_folder/science_data/

    :param experiment_folder: experiment folder path
    :param experiment_run_id: experiment run_id
    :param smap_data_folder: SMAP data folder path, if None will get the folder from $SMAP_L4_PATH
    :param download_smap_data: Download SMAP data if not available?
    :return:
    """
    # parameters
    max_duration = dt.timedelta(hours=24)
    out_files_duration = dt.timedelta(hours=3)

    if smap_data_folder is None:
        smap_data_folder = smap_l4_default_folder

    plans_folder = os.path.join(experiment_folder, 'plan', experiment_run_id)
    plans_files_path = get_plans_cvs_files(plans_folder)
    num_out_files = int((max_duration / out_files_duration))
    files_epoch_list = [extract_epoch_id(plan_fp) for plan_fp in plans_files_path]
    first_epoch = min(files_epoch_list)
    epoch_list = [first_epoch + idx * out_files_duration for idx in range(num_out_files)]

    # download SMAP data
    if download_smap_data:
        download_smap_l4_3h_9km_grid_data(smap_data_folder, min(epoch_list) - out_files_duration/2, max(epoch_list) + out_files_duration/2,
                                          force=False, quiet=False)

    # generate output files
    for i_outfile, current_epoch in enumerate(epoch_list):
        log_print(f'epoch {i_outfile+1} out of {len(epoch_list)}: {current_epoch}')

        start_time = dt.datetime.utcnow()
        gp_data = {'gp': [],
                   'lat': [],
                   'lon': [],
                   'time': [],
                   'smap_grid_row': [],
                   'smap_grid_col': [],
                   'igbp_class': [],
                   'biome_id': [],
                   'p_band_pointing_options': [],
                   'l_band_pointing_options': [],
                   'p_band_data_time': [],
                   'l_band_data_time': [],
                   'p_band_operation_mode': [],
                   'l_band_operation_mode': [],
                   'max_time_obs_span_hr': [],
                   'sm_noise_val': [],
                   'smap_soil_moisture': [],
                   'simulated_soil_moisture': []}

        for ifile, plan_fp in tqdm(enumerate(plans_files_path), desc='Ingesting Plans', total=len(plans_files_path), unit='file'):
            file_epoch = files_epoch_list[ifile]
            step_size = pd.read_csv(plan_fp, skiprows=[0, 1, 2], nrows=1, header=None).astype(str)  # 4th row contains the stepsize
            step_size = float(step_size[0][0].split()[5])

            # read in the plan
            plan = pd.read_csv(plan_fp, skiprows=[0, 1, 2, 3], index_col=None)

            # check if the last time in the file before the current epoch start time
            if file_epoch + dt.timedelta(seconds=int(plan['TP'][plan.shape[0]-1]) * step_size) < current_epoch:
                continue

            for index, row in plan.iterrows():
                l_band_obs = False
                p_band_obs = False
                if row['mode'] == 'obs':
                    obs_time = file_epoch + dt.timedelta(seconds=int(row['TP']) * step_size)
                    if obs_time > current_epoch + out_files_duration:
                        break  # break if obs time > out file max time, note time increases
                    if obs_time < current_epoch:
                        continue
                    if not (pd.isnull(row.iloc[2])):  # check for L-band operation
                        if not (row.iloc[2][-1] == 'S' or row.iloc[2][-1] == 'E'):
                            l_band_obs = True

                    if not (pd.isnull(row.iloc[3])):  # check for P-band operation
                        if not (row.iloc[3][-1] == 'S' or row.iloc[3][-1] == 'E'):
                            p_band_obs = True
                    if l_band_obs:
                        l_band_gp_list = unpack_gb_list(row['L-band GPs'])
                        for gp_val in l_band_gp_list:
                            if gp_val in gp_data['gp']:
                                gb_arg = np.where(np.array(gp_data['gp']) == gp_val)[0][0]
                                gp_data['l_band_pointing_options'][gb_arg].append(int(row['L-band'][2:]))
                                gp_data['l_band_data_time'][gb_arg].append(obs_time)
                                if abs(obs_time - max(gp_data['l_band_data_time'][gb_arg])) > out_files_duration or abs(
                                        obs_time - min(gp_data['l_band_data_time'][gb_arg])) > out_files_duration:
                                    raise ValueError(f"all data need to be within {out_files_duration}, got {obs_time}, "
                                                     f"{max(gp_data['l_band_data_time'][gb_arg])}, {min(gp_data['l_band_data_time'][gb_arg])}")
                            else:  # create gp data
                                gp_data['gp'].append(gp_val)
                                gp_data['p_band_pointing_options'].insert(len(gp_data['gp'])-1, [])
                                gp_data['p_band_data_time'].insert(len(gp_data['gp'])-1, [])
                                gp_data['l_band_pointing_options'].insert(len(gp_data['gp'])-1, [int(row['L-band'][2:])])
                                gp_data['l_band_data_time'].insert(len(gp_data['gp'])-1, [obs_time])

                    if p_band_obs:
                        p_band_gp_list = unpack_gb_list(row['P-band GPs'])
                        for gp_val in p_band_gp_list:
                            if gp_val in gp_data['gp']:
                                gb_arg = np.where(np.array(gp_data['gp']) == gp_val)[0][0]
                                gp_data['p_band_pointing_options'][gb_arg].append(int(row['P-band'][2:]))
                                gp_data['p_band_data_time'][gb_arg].append(obs_time)
                                if abs(obs_time - max(gp_data['p_band_data_time'][gb_arg])) > out_files_duration or abs(
                                        obs_time - min(gp_data['p_band_data_time'][gb_arg])) > out_files_duration:
                                    raise ValueError(f"all data need to be within {out_files_duration}, got {obs_time}, "
                                                     f"{max(gp_data['p_band_data_time'][gb_arg])}, {min(gp_data['p_band_data_time'][gb_arg])}")

                            else:  # create gp data
                                gp_data['gp'].append(gp_val)
                                gp_data['p_band_pointing_options'].insert(len(gp_data['gp'])-1, [int(row['P-band'][2:])])
                                gp_data['p_band_data_time'].insert(len(gp_data['gp'])-1, [obs_time])
                                gp_data['l_band_pointing_options'].insert(len(gp_data['gp'])-1, [])
                                gp_data['l_band_data_time'].insert(len(gp_data['gp'])-1, [])

        if len(gp_data['gp']) < 1:  # skip empty out files
            continue
        # Get SMAP SM values
        smap_file_path = get_smap_file_name(current_epoch, smap_data_folder)
        # smap_file_path = os.path.join(smap_data_folder, smap_file_name)
        smap_h5_data = h5py.File(smap_file_path)
        _smap_soil_moisture_values = np.array(smap_h5_data['Analysis_Data']['sm_surface_analysis'], dtype=float)
        # close SMAP file
        smap_h5_data.close()

        gird_csv_file_path = os.path.join(experiment_folder, 'common', 'grid.csv')
        gp_df = pd.read_csv(gird_csv_file_path)
        gp_data['smap_data_file_name'] = os.path.basename(smap_file_path)
        pointing_potions_dic = get_pointing_options_dic(experiment_folder)

        obs_quality_df_list = {}

        for biom in get_list_biomes(experiment_folder):
            obs_quality_fn = f"table_{biom}.csv"
            obs_quality_file_path = os.path.join(experiment_folder, 'obs_quality', obs_quality_fn)
            obs_quality_df_list[biom] = pd.read_csv(obs_quality_file_path)

        for key in ['lat', 'lon', 'smap_grid_row', 'smap_grid_col', 'smap_soil_moisture', 'sm_noise_val', 'max_time_obs_span']:
            gp_data[key] = np.zeros(len(gp_data['gp']), dtype=float)
        for key in ['smap_grid_row', 'smap_grid_col', 'igbp_class', 'p_band_operation_mode', 'l_band_operation_mode']:
            gp_data[key] = np.zeros(len(gp_data['gp']), dtype=int)
        for key in ['biome_id']:
            gp_data[key] = [''] * len(gp_data['gp'])
            gp_data['time'] = [None] * len(gp_data['gp'])
        #  fill metadata
        for i_gp, gp_id in tqdm(enumerate(gp_data['gp']), desc='Filling metadata', unit='GP', total=len(gp_data['gp'])):
            gp_grid_data_ = extract_metadata_from_grid_csv(gp_id, gp_df)
            gp_data['lat'][i_gp] = gp_grid_data_['lat']
            gp_data['lon'][i_gp] = gp_grid_data_['lon']
            gp_data['smap_grid_row'][i_gp] = gp_grid_data_['row']
            gp_data['smap_grid_col'][i_gp] = gp_grid_data_['col']
            gp_data['igbp_class'][i_gp] = gp_grid_data_['igbp_class']
            gp_data['biome_id'][i_gp] = gp_grid_data_['biome_id']

            _smap_sm_val_ = _smap_soil_moisture_values[gp_data['smap_grid_row'][i_gp], gp_data['smap_grid_col'][i_gp]]
            if _smap_sm_val_ < 0.0:  # catch no data value
                _smap_sm_val_ = np.nan
            gp_data['smap_soil_moisture'][i_gp] = _smap_sm_val_

            gp_data['p_band_operation_mode'][i_gp] = extracting_operation_modes(gp_data['p_band_pointing_options'][i_gp], pointing_potions_dic, gp_id)
            gp_data['l_band_operation_mode'][i_gp] = extracting_operation_modes(gp_data['l_band_pointing_options'][i_gp], pointing_potions_dic, gp_id)

            obs_quality_df = obs_quality_df_list[gp_data['biome_id'][i_gp]]
            sel_row = obs_quality_df[(obs_quality_df['LbandSAR'] == gp_data['l_band_operation_mode'][i_gp]) & (
                    obs_quality_df['PbandSAR'] == gp_data['p_band_operation_mode'][i_gp])].reset_index()
            if len(sel_row) != 1:
                raise ValueError(f'expected one row, got {sel_row}')

            gp_data['sm_noise_val'][i_gp] = float(sel_row['Q1'][0])
            all_times_ = gp_data['p_band_data_time'][i_gp] + gp_data['l_band_data_time'][i_gp]
            gp_data['max_time_obs_span'][i_gp] = (max(all_times_) - min(all_times_)) / dt.timedelta(hours=1)
            gp_data['time'][i_gp] = datetime_mean(all_times_)

        # generate simulated data
        noise_ = np.random.randn(len(gp_data['gp']))
        gp_data['simulated_soil_moisture'] = gp_data['smap_soil_moisture'] + noise_ * gp_data['sm_noise_val']
        # replace nan by fill value
        gp_data['simulated_soil_moisture'][np.isnan(gp_data['simulated_soil_moisture'])] = NO_SM_VAL
        end_time = dt.datetime.utcnow()
        science_data_folder = os.path.join(experiment_folder, 'science_data')
        check_folder(science_data_folder)
        file_date_str = current_epoch.strftime("%Y%m%dT%H%M%S")
        partial_out_file_name = f'science_data_{experiment_run_id}_{file_date_str}'
        out_json_file_path = os.path.join(science_data_folder, f'{partial_out_file_name}_debug_data.json')
        out_csv_file_path = os.path.join(science_data_folder, f'{partial_out_file_name}.csv')

        log_print(f"Epoch time: {current_epoch}, SMAP file {os.path.basename(smap_file_path)}, No. GP with SM data: {len(gp_data['simulated_soil_moisture'] > 0.0)}")

        with open(out_json_file_path, 'w') as f:
            json.dump(gp_data, f, cls=NumpyArrayEncoder)
        df_out = pd.DataFrame({'GP index': gp_data['gp'],
                               'time': [time_.isoformat() for time_ in gp_data['time']],
                               'geophysicalValue': gp_data['simulated_soil_moisture']})
        df_out.to_csv(out_csv_file_path, index=False)

        science_data_dic = {
            "type": "ScienceData",
            "runId": "run_id",
            "createdTime": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "file_epoch": min(files_epoch_list).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration": int(out_files_duration / dt.timedelta(hours=1)),
            "geophysicalVar": {
                "property": "SOIL MOISTURE",
                "units": "m3/m3"},
            "grid": {"filePath": "common/grid.csv"},
            "observation": [],
            "debugLog": {
                "production": {
                    "startTime": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "endTime": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "processTime": f'{(end_time - start_time) / dt.timedelta(hours=1):.2}',
                    "personnelTime": 0.5,
                    "timeFactor": 1
                },
                "software": {
                    "name": "Python",
                    "version": "",
                    "gitUrl": "https://bitbucket.org/usc_mixil/multi-instruments-soil-moisture-retrieval/",
                    "commit": ""
                },
                "machine": {
                    "type": "PC",
                    "processor": "10th gen Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz",
                    "ram": 64,
                    "OS": "Ubuntu 20.04"
                }
            }
        }
        for i_gp, gp_id in enumerate(gp_data['gp']):
            science_data_dic['observation'].append({'positionId': gp_id,
                                                    'time': gp_data['time'][i_gp].isoformat(),
                                                   'geophysicalValue': gp_data['simulated_soil_moisture'][i_gp]})

        science_data_out_json = os.path.join(science_data_folder, f'{partial_out_file_name}.json')
        with open(science_data_out_json, 'w') as f:
            json.dump(science_data_dic, f, cls=NumpyArrayEncoder, indent=True)


def create_parser():
    # Create the parser
    pars = argparse.ArgumentParser(description='Generate simulated soil moisture values from a plan')
    # Add the arguments
    pars.add_argument('exp_path',  nargs='?', default=None, type=str, help='experiment directory path')
    pars.add_argument('-r', '--run_id',  nargs='?', default='RUN001', type=str, help='Run id ')

    pars.add_argument('--smap_path',  nargs='?', default=None, type=str, help='SMAP data path')
    pars.add_argument('--download_smap', action='store_true', default=False, help='Download SMAP data if not exist?')

    return pars


if __name__ == '__main__':
    pars_ = create_parser()
    args = pars_.parse_args()

    smap_l4_folder = args.smap_path
    download_smap_data_tf = args.download_smap
    experiment1_folder = args.exp_path
    run_id = args.run_id

    generate_out_sim_soil_moisture(experiment1_folder, run_id, smap_l4_folder, download_smap_data_tf)
