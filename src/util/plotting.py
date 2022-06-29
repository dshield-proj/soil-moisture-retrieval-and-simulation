"""
  DESCRIPTION
            Plotting module

   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-04-26
   Updated  2022-06-22: move all plots to this module

   Copyright 2022 University of Southern California
"""

import os
import random
import string
import numpy as np
from matplotlib import pyplot as plt
from helpper.scenarios_data import scenario2vegtype
import matplotlib as mpl

plt_font_size = 12  # font size
plt_title = False  # plot figures titles
plot_db_range = 20  # default dynamic range of dB plots
default_save_type = ['png', 'pdf']  # default save image types

SMALL_SIZE = plt_font_size
MEDIUM_SIZE = plt_font_size
BIGGER_SIZE = plt_font_size
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'


clr_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
            '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
stat2label = {'rmse': 'RMSE [$\mathrm{m}^3/\mathrm{m}^3$]',
              'ubrmse': 'ubRMSE [$\mathrm{m}^3/\mathrm{m}^3$]',
              'bias': 'Bias [$\mathrm{m}^3/\mathrm{m}^3$]'}


def save_figure(fig, fig_out_folder, img_save_name, tf_save_fig=True, fig_save_types=None):
    """
    save figure

    :param fig: figure object
    :type fig: plt.figure
    :param fig_out_folder: image saving folder
    :type fig_out_folder: str
    :param img_save_name: image save name
    :type img_save_name: str
    :param tf_save_fig: save the figure?
    :type tf_save_fig: bool
    :param fig_save_types: type of image? png, eps, pdf, svg ...
    :type fig_save_types: list of str or str
    :return: True if image is saved, else False
    :rtype: bool
    """
    if not tf_save_fig:
        return False
    if fig_save_types is None:
        fig_save_types = default_save_type
    elif type(fig_save_types) is str:
        fig_save_types = [fig_save_types]
    for fig_type in fig_save_types:
        name = f"{img_save_name.split('.')[0]}.{fig_type}"
        fig.savefig(os.path.join(fig_out_folder, name), format=fig_type, bbox_inches='tight')
    return True


def plt_sm_hist(true_sm: np.ndarray, est_sm: np.ndarray, img_out_path: str, img_name: str):
    fig, ax = plt.subplots(1, figsize=(5, 3))
    error_ = np.sqrt((true_sm - est_sm) ** 2).flatten()
    plt.hist(error_)
    ax.grid()
    ax.set_xlabel('|Error| [$\mathrm{m}^3/\mathrm{m}^3$]')
    ax.set_ylabel('Counts')
    plt.tight_layout()
    fig.savefig(os.path.join(img_out_path, f'{img_name}.png'))
    return fig


def plt_sm_error_history(sm_error_history_list, sm_list, num_trials, img_save_tag, history_fig_rand_tag, out_folder_path, tf_save_error_history_fig):
    fig, ax = plt.subplots(1)
    for i_sm, sm in enumerate(sm_list):
        for i_trial in range(num_trials):
            sm_error = sm_error_history_list[i_sm][i_trial]
            ax.plot(np.arange(len(sm_error)), sm_error, linewidth=2, color=clr_list[(i_sm * num_trials + i_trial) % len(clr_list)])
    ax.grid()
    ax.set_xlabel('Iteration (including internal)')
    ax.set_ylabel('Error in soil moisture')
    ax.set_ylim([-0.03, 0.03])
    plt.tight_layout()
    if history_fig_rand_tag is None:
        history_fig_rand_tag = ''.join(random.choice(string.ascii_lowercase) for _ in range(8))
    img_save_name = f'plt_sm_error_history_{img_save_tag}_{history_fig_rand_tag}'
    fig_out_folder = os.path.join(out_folder_path, 'error_history')
    if not os.path.isdir(fig_out_folder):
        os.makedirs(fig_out_folder)
    save_figure(fig, fig_out_folder, img_save_name, tf_save_error_history_fig, 'png')
    return fig


def plt_hist_all_scenario_metrics(df_data, list_scenarios_names, file_name_base, img_out_path):
    for _stat_merit in ['rmse', 'ubrmse', 'bias']:
        fig, ax = plt.subplots(1)
        for i_scenario, scenario in enumerate(list_scenarios_names):
            hist_, hist_edg = np.histogram(np.nan_to_num(df_data[f'{scenario}_{_stat_merit}'].to_numpy()))
            hist_cb = hist_edg[:-1] + (hist_edg[1] - hist_edg[0]) / 2
            ax.plot(hist_cb, hist_, linewidth=1.5, label=scenario2vegtype(scenario), marker='o', color=clr_list[i_scenario])
        ax.grid()
        ax.legend()
        ax.set_xlabel(stat2label.get(_stat_merit))
        ax.set_ylabel('Counts')
        plt.tight_layout()
        plt.tight_layout()
        save_figure(fig, img_out_path, f'hist_{_stat_merit}_{file_name_base}_all_scenario.png', tf_save_fig=True)


def plt_hist_single_scenario_metrics(stat_data, metric_name, stat_img_save_tag, img_out_path):
    fig, ax = plt.subplots(1)
    plt.hist(stat_data)
    ax.grid()
    ax.set_xlabel(stat2label.get(metric_name))
    ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.tight_layout()
    fig.savefig(os.path.join(img_out_path, f'all_hist_{stat_img_save_tag}.png'))
