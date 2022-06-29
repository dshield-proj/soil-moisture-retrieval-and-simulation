from helpper.scenarios_data import scenario2vegtype
from intermediate_product_mapping import operation_mode2num_observ_inc_angle
from util.plotting import save_figure
from util.run_util import check_folder, log_print
import argparse
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

"""
  DESCRIPTION
            Generate observation quality files from intermediate JSON file
            
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-04-25
   Updated  2022-06-22: clean up and add comments
 
   Copyright 2022 University of Southern California
"""

clr_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'

plt_font_size = 12
fig_width = 10

stat2label = {'rmse': 'RMSE [$\mathrm{m}^3/\mathrm{m}^3$]',
              'ubrmse': 'ubRMSE [$\mathrm{m}^3/\mathrm{m}^3$]',
              'bias': 'Bias [$\mathrm{m}^3/\mathrm{m}^3$]'}


def scenario2table_b_name(scenario_):
    bio_id = scenario2bioid(scenario_)
    return f'table_{bio_id}.csv'


def scenario2bioid(scenario):
    dic_ = {'metolius': 'B1',
            'walnut_gulch': 'B2',
            'tonzi_ranch': 'B3',
            'yanco': 'B4',
            'las_cruces': 'B5'}

    return dic_.get(scenario)


def scenario2bio_name(scenario):
    dic_ = {'walnut_gulch': 'shrubland',
            'tonzi_ranch': 'savanna',
            'metolius': 'forest',
            'las_cruces': 'bare',
            'yanco': 'cropland'}
    return dic_.get(scenario)


def gen_obs_quality_file(in_intermediate_json_file, out_folder):
    max_op_mode = 9
    inc_angle_list = [35, 45, 55]
    num_inc_angles = len(inc_angle_list)
    num_instruments = 2
    sel_quality = 'rmse'

    with open(in_intermediate_json_file, 'r') as f:
        inter_json_data = json.load(f)
    list_scenarios_names = [key for key in inter_json_data.keys() if key not in ['input_param']]

    inst1_list = []
    inst2_list = []
    for inst1 in range(0, max_op_mode + 1):
        for inst2 in range(0, max_op_mode + 1):
            inst1_list.append(inst1)
            inst2_list.append(inst2)

    df = pd.DataFrame({'l_band': inst1_list,
                       'p_band': inst2_list})

    sensor_names_list = ['l_band', 'p_band']
    idx_list = np.zeros(df.shape[0], dtype=int)
    for irow, row_data in df.iterrows():
        num_observ_l_band_row, num_observ_p_band_row = operation_mode2num_observ_inc_angle(row_data, num_inc_angles, num_instruments)
        row_param = {'l_band': num_observ_l_band_row,
                     'p_band': num_observ_p_band_row}
        is_sel_row = np.zeros((len(inter_json_data[list_scenarios_names[0]]), len(sensor_names_list)), dtype=bool)
        for ikey, key in enumerate(sensor_names_list):
            is_sel_row[:, ikey] = np.prod(np.array(inter_json_data['input_param'][key]) == row_param[key], axis=1)
        data_idx_ = np.where(np.prod(is_sel_row, axis=1))[0]
        if data_idx_.size <= 0:
            log_print(f'row {irow}: no performance data for {row_data}')
        elif data_idx_.size == 1:
            for scenario_ in list_scenarios_names:
                for key1, key2 in zip(['rmse', 'ubrmse', 'std', 'bias'], ['sm_rmse', 'sm_ubrmse', 'est_sm_std', 'sm_bias']):
                    df.at[irow, f'{scenario_}_{key1}'] = inter_json_data[scenario_][data_idx_[0]][key2]
            idx_list[irow] = data_idx_[0]
        else:
            raise RuntimeError(f'Expected single index value, got {data_idx_}')

    # calculate processing time
    unique_idx = np.unique(idx_list)
    total_time_sec_list = np.ones(unique_idx.size, dtype=float) * np.nan
    for irow, data_idx_ in enumerate(unique_idx):
        total_time_sec_list[irow] = sum(
            [np.array(inter_json_data[scenario_][data_idx_]['retrieval_duration_sec']).sum() for scenario_ in list_scenarios_names])
    del inter_json_data

    # Export output files
    analysis_ = {}
    for scenario_ in list_scenarios_names:
        table_b_fn = scenario2table_b_name(scenario_)
        out_df = pd.DataFrame({'LbandSAR': df['l_band'],
                               'PbandSAR': df['p_band'],
                               'Q1': df[f'{scenario_}_{sel_quality}']})
        out_df.to_csv(os.path.join(out_folder, table_b_fn), index=False)
        log_print(table_b_fn)

        for _stat_merit in ['rmse', 'ubrmse', 'bias']:
            max_id = df[f'{scenario_}_{_stat_merit}'].idxmax()
            min_id = df[f'{scenario_}_{_stat_merit}'].idxmin()
            pband_only = df[f'{scenario_}_{_stat_merit}'][(df['l_band'] == 0) & (df['p_band'] > 0)]
            lband_only = df[f'{scenario_}_{_stat_merit}'][(df['p_band'] == 0) & (df['l_band'] > 0)]
            both_radars_types = df[f'{scenario_}_{_stat_merit}'][(df['p_band'] > 0) & (df['l_band'] > 0)]
            metric_per_num_inst = [[]] * 4
            metric_per_num_inst[0] = df[f'{scenario_}_{_stat_merit}'][((df['p_band'] < 4) & (df['p_band'] > 0) & (df['l_band'] == 0)) | ((df['l_band'] > 0) & (df['l_band'] < 4) & (df['p_band'] == 0))]
            metric_per_num_inst[1] = df[f'{scenario_}_{_stat_merit}'][((df['p_band'] > 3) & (df['l_band'] == 0)) | ((df['l_band'] > 3) & (df['p_band'] == 0)) | ((0 < df['p_band']) & (df['p_band'] < 4) & (0 < df['l_band']) & (df['l_band'] < 4))]
            metric_per_num_inst[2] = df[f'{scenario_}_{_stat_merit}'][((df['p_band'] > 3) & (0 < df['l_band']) & (df['l_band'] < 4)) | ((df['l_band'] > 3) & (0 < df['p_band']) & (df['p_band'] < 4))]
            metric_per_num_inst[3] = df[f'{scenario_}_{_stat_merit}'][((df['p_band'] > 3) & (df['l_band'] > 3)) | ((df['l_band'] > 3) & (df['p_band'] > 3))]
            analysis_[f'{scenario_}_{_stat_merit}'] = [scenario2vegtype(scenario_),
                                                       df[f'{scenario_}_{_stat_merit}'][max_id],
                                                       df[f'{scenario_}_{_stat_merit}'][min_id],
                                                       df[f'{scenario_}_{_stat_merit}'].mean(),
                                                       df[f'{scenario_}_{_stat_merit}'].std(), sum(df[f'{scenario_}_{_stat_merit}'] <= 0.01),
                                                       pband_only.mean(),
                                                       lband_only.mean(),
                                                       both_radars_types.mean(),
                                                       metric_per_num_inst[0].mean(),
                                                       metric_per_num_inst[1].mean(),
                                                       metric_per_num_inst[2].mean(),
                                                       metric_per_num_inst[3].mean()]
            analysis_[f'{scenario_}_{_stat_merit}_inst'] = ['', ' '.join(f'{id_}' for id_ in df.iloc[max_id, 0:4].to_numpy(dtype=int)),
                                                            ' '.join(f'{id_}' for id_ in df.iloc[min_id, 0:4].to_numpy(dtype=int)),
                                                            sum(~df[f'{scenario_}_{_stat_merit}'].isnull()), '', '',
                                                            len(pband_only),
                                                            len(lband_only),
                                                            len(both_radars_types),
                                                            len(metric_per_num_inst[0]),
                                                            len(metric_per_num_inst[1]),
                                                            len(metric_per_num_inst[2]),
                                                            len(metric_per_num_inst[3])]
    out_xls_file_path = os.path.join(out_folder, 'analysis_summary_2_sensors_LbandSAR_PbandSAR.xlsx')
    analysis_df = pd.DataFrame(analysis_)
    analysis_df.index = ['vegetation_type', 'max', 'min', 'mean', 'std', 'less than 0.01', 'Pband', 'Lband', 'both_bands', 'single_inst', 'two_inst',
                         'three_inst', 'four_inst']
    writer = pd.ExcelWriter(out_xls_file_path)
    df.to_excel(writer, sheet_name='data', index=False)
    analysis_df.T.to_excel(writer, sheet_name='analysis')
    writer.close()

    # plot stats per instrument
    inst_num_list = np.arange(1, 5).astype(float) - 0.5
    bars_width = float((inst_num_list[1] - inst_num_list[0])) / (len(list_scenarios_names) + 1)

    for _stat_merit in ['rmse', 'ubrmse', 'bias']:
        fig, ax = plt.subplots(1)
        for i_scenario, scenario in enumerate(list_scenarios_names):
            data_ = analysis_[f'{scenario}_{_stat_merit}'][-4:]
            ax.bar(inst_num_list + bars_width * i_scenario + bars_width/2, data_, width=bars_width, align='edge', label=scenario2vegtype(scenario), color=clr_list[i_scenario])
        ax.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(.42, 1.25), ncol=3, fontsize=plt_font_size, handlelength=1, columnspacing=1)
        ax.set_ylabel(stat2label.get(_stat_merit))
        ax.set_xlabel('Number of observations')
        ax.set_xticks(np.arange(1, 5))
        ax.set_xlim([inst_num_list[0], inst_num_list[-1] + 1])
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        file_name_base = '2_sensors_LbandSAR_PbandSAR'
        save_figure(fig, out_folder, f'per_inst_{_stat_merit}_{file_name_base}_bar_all_scenario.png', tf_save_fig=True, fig_save_types=['png', 'pdf'])
    plt.close('all')

    # plot stat for L/P band or both
    bands_idx_list = np.arange(1, 4).astype(float) - 0.5
    bars_width = float((bands_idx_list[1] - bands_idx_list[0])) / (len(list_scenarios_names) + 1)
    for _stat_merit in ['rmse', 'ubrmse', 'bias']:
        fig, ax = plt.subplots(1)
        for i_scenario, scenario in enumerate(list_scenarios_names):
            data_ = analysis_[f'{scenario}_{_stat_merit}'][-len(inst_num_list) - 3:-len(inst_num_list)]
            ax.bar(bands_idx_list + bars_width * i_scenario + bars_width/2, data_, width=bars_width, align='edge', label=scenario2vegtype(scenario), color=clr_list[i_scenario])
        ax.grid()
        plt.legend(loc="upper center", bbox_to_anchor=(.42, 1.25), ncol=3, fontsize=plt_font_size, handlelength=1, columnspacing=1)
        ax.set_ylabel(stat2label.get(_stat_merit))
        ax.set_xlim([bands_idx_list[0], bands_idx_list[-1] + 1])
        plt.xticks([1, 2, 3], ['P-band', 'L-band', 'Both'])
        plt.tight_layout()
        plt.tight_layout()
        plt.tight_layout()
        file_name_base = '2_sensors_LbandSAR_PbandSAR'
        save_figure(fig, out_folder, f'type_inst_{_stat_merit}_{file_name_base}_bar_all_scenario.png', tf_save_fig=True, fig_save_types=['png', 'pdf'])
    plt.close('all')

    # plot histogram
    use_max_val_in_hist_plt = True
    max_val_dic = {'rmse': 0.06, 'ubrmse': 0.06, 'bias': 0.03}
    max_val_plt_dic = {'rmse': 0.04, 'ubrmse': 0.03, 'bias': 0.015}
    for _stat_merit in ['rmse', 'ubrmse', 'bias']:
        fig, ax = plt.subplots(1)
        _hist_bin = np.linspace(0.0, max_val_dic[_stat_merit], 13)
        for i_scenario, scenario in enumerate(list_scenarios_names):
            data_ = df[f'{scenario}_{_stat_merit}'].to_numpy()
            data_ = data_[~np.isnan(data_)]
            if use_max_val_in_hist_plt:
                hist_, hist_edg = np.histogram(data_, _hist_bin)
            else:
                hist_, hist_edg = np.histogram(data_)
            hist_cb = hist_edg[:-1] + (hist_edg[1] - hist_edg[0]) / 2
            markerline, stemlines, baseline = ax.stem(hist_cb, hist_, label=scenario2vegtype(scenario), basefmt='None', linefmt=clr_list[i_scenario])
            markerline.set_markeredgecolor(clr_list[i_scenario])
            markerline.set_markerfacecolor(clr_list[i_scenario])
        ax.grid()
        ax.legend()
        if use_max_val_in_hist_plt:
            ax.set_xlim([0.0, max_val_plt_dic[_stat_merit]])
        ax.set_xlabel(stat2label.get(_stat_merit))
        ax.set_ylabel('Counts')
        plt.tight_layout()
        ax.set_ylim([0, ax.get_ylim()[-1]])
        plt.tight_layout()

        file_name_base = '2_sensors_LbandSAR_PbandSAR'
        save_figure(fig, out_folder, f'hist_{_stat_merit}_{file_name_base}_all_scenario.png', tf_save_fig=True, fig_save_types=['png', 'pdf'])

        # plot hist bar
        fig, ax = plt.subplots(1)
        _hist_bin = np.linspace(0.0, max_val_dic[_stat_merit], 13)
        bars_width = abs((_hist_bin[1] - _hist_bin[0])) / len(list_scenarios_names)
        for i_scenario, scenario in enumerate(list_scenarios_names):
            data_ = df[f'{scenario}_{_stat_merit}'].to_numpy()
            data_ = data_[~np.isnan(data_)]
            if use_max_val_in_hist_plt:
                hist_, hist_edg = np.histogram(data_, _hist_bin)
            else:
                hist_, hist_edg = np.histogram(data_)
            ax.bar(hist_edg[:-1] + bars_width * i_scenario, hist_, width=bars_width, align='edge', label=scenario2vegtype(scenario), color=clr_list[i_scenario])
        ax.grid()
        ax.legend()
        if use_max_val_in_hist_plt:
            ax.set_xlim([0.0, max_val_plt_dic[_stat_merit]])
        ax.set_xlabel(stat2label.get(_stat_merit))
        ax.set_ylabel('Counts')
        plt.tight_layout()
        plt.tight_layout()
        file_name_base = '2_sensors_LbandSAR_PbandSAR'
        save_figure(fig, out_folder, f'hist_{_stat_merit}_{file_name_base}_bar_all_scenario.png', tf_save_fig=True, fig_save_types=['png', 'pdf'])
    log_print(f'No. unique modes: {len(total_time_sec_list)}')
    log_print(f'Total processing time [sec]: {sum(total_time_sec_list)}')
    log_print(f'Total processing time [hr]: {sum(total_time_sec_list) / 3600.0}')


def create_parser():
    global pars
    # Create the parser
    pars = argparse.ArgumentParser(description='Generate observation quality files')
    # Add the arguments
    pars.add_argument('-o', '--out_folder', nargs='?', metavar='output_main_path', default=None, type=str, help='out folder path')
    pars.add_argument('--inter_json_file', nargs='?', default=None, type=str, help='intermediate product json file path [input]')
    return pars


if __name__ == '__main__':
    pars = create_parser()
    args = pars.parse_args()
    if args.out_folder is None:
        args.out_folder = 'obs_quality'
    check_folder(args.out_folder)
    if args.inter_json_file is None:
        args.inter_json_file = 'instruments_inc_angles_and_observations.json'
    gen_obs_quality_file(args.inter_json_file, args.out_folder)
