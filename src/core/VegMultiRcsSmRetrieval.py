from core.forest_fwd_model import forest_fwd_model
from multi_sa_py.MultiSimAnn import MultiSimAnn
from typing import Optional
from util.plotting import save_figure
import copy
import matplotlib.pyplot as plt
import numpy as np

"""
  DESCRIPTION
            Soil moisture value retrieval from multiple radar measurements. 
             
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-04-14
   Updated  2022-06-22: clean up and add comments
 
   Copyright 2022 University of Southern California
"""


class VegRcsSmRetrieval(MultiSimAnn):
    def __init__(self, veg_parameters: dict, radar_parameters_list: list[dict], num_rcs_avg, itr: int, n_t: int = 1, n_s: int = 20,
                 n_mul: int = 100, temper: float = 10.0, r_t: float = 0.85, c_j: float = 2.0,
                 step_lim: float = 0.0001, f_stop: float = 3.3e-7, overflow_cost: float = 100, up_limit: float = 1, down_limit: float = 0,
                 overflow_option: int = 2, max_n_function_eval: Optional[int] = None, dev_mode: Optional[bool] = False,
                 silent_mode: Optional[bool] = True, cost_fun='l2'):

        """

        :param veg_parameters: vegetation parameter (from fwd_veg_module.import_config_file()
        :param radar_parameters_list: list of instruments parameters
        :param itr: Number of iterations
        :param n_t: Number of step length adjustment at each temperature
        :param n_s: Number of iterations after each step length adjustment
        :param n_mul: Number of iterations of multidirectional search
        :param temper: temperature
        :param r_t: temperature step size
        :param c_j: ... step size
        :param step_lim:
        :param f_stop:
        :param overflow_cost: overflow cost (default: None)
        :param up_limit: upper limit of x
        :param down_limit: lower limit of x
        :param overflow_option: for choosing the cost value out of x limitations (check self.fcost() for details)
        :param dev_mode: if True, the random number will be taken from "rand_seq.mat" (default: False)
        """
        try:
            len(veg_parameters['soilcontent'])
        except TypeError:
            veg_parameters['soilcontent'] = np.array([float(veg_parameters['soilcontent'])])
        self.target_sm = veg_parameters['soilcontent']

        n_unknown = len(veg_parameters['soilcontent'])
        super().__init__(n_unknown, itr, n_t, n_s, n_mul, temper, r_t, c_j, step_lim, f_stop, overflow_cost, up_limit, down_limit, overflow_option,
                         max_n_function_eval, dev_mode, silent_mode, cost_fun)
        self.n_lyr = len(veg_parameters['soilcontent'])
        measurements_weight = []
        list_veg_param = [copy.copy(veg_parameters) for _ in radar_parameters_list]
        for i_radar, radar_param in enumerate(radar_parameters_list):
            list_veg_param[i_radar]['freq'] = radar_param['center_freq']  # Hz
            inc_angle = np.deg2rad(radar_param['inc_angle_deg'])
            list_veg_param[i_radar]["thetaiList"] = [inc_angle]
            list_veg_param[i_radar]["phiiList"] = [0]
            list_veg_param[i_radar]["thetasList"] = [inc_angle]
            list_veg_param[i_radar]["phisList"] = [np.pi]
            list_veg_param[i_radar]["polarization"] = radar_param["polarization"]
            list_veg_param[i_radar]["num_observation"] = radar_param["num_observation"]
            if 'vh' in radar_param["polarization"] or 'hv' in radar_param["polarization"]:
                _w = len(radar_param["polarization"]) / (len(radar_param["polarization"]) * 2 - 1)
            else:
                _w = 1/2
            for pol in radar_param["polarization"]:
                if pol in ['vv', 'hh']:
                    measurements_weight += [2 * _w]
                elif pol in ['vh', 'hv']:
                    measurements_weight += [_w]
        self.list_veg_param = list_veg_param
        self.num_rcs_avg = num_rcs_avg
        self.radar_parameters_list = radar_parameters_list
        self.measurements_weight = np.array(measurements_weight)
        self.num_measurements = len(measurements_weight)
        self.num_instruments = len(radar_parameters_list)

    def get_initial_x(self):
        """
        Return initial guess of x

        :return: initial guess of x
        """

        x_initial = 0.5 * np.ones(self.n_unknown)
        return x_initial

    def get_measured_fx(self):
        """
        measurement
        :return:
        """
        sfr = np.sqrt(4 / np.pi - 1)  # 0.523 (signal-to-fluctuation-ratio for single look)

        measured_sigma_ = []
        for i_radar, (radar_param, veg_param) in enumerate(zip(self.radar_parameters_list, self.list_veg_param)):
            kp = 10 ** (radar_param['noise_sigma0_db'] / 10)
            n_looks = radar_param['n_looks_per_km2']
            n_obsrv = radar_param['num_observation']
            sigma_no_noise = forest_fwd_model.sigma0_fwd_model(veg_param, use_parallel=False)
            wgn = np.random.randn(2, n_obsrv, len(radar_param['polarization']))  # white Gaussian noise
            measured_sigma_ += (sigma_no_noise * np.mean((1 + sfr/np.sqrt(n_looks) * wgn[0, ...]) + kp * wgn[1, ...], axis=0)).tolist()
        measured_sigma = np.array(measured_sigma_) * self.measurements_weight
        if np.isnan(measured_sigma).any():
            print(f'Nan found, {measured_sigma}')
        return measured_sigma

    def bm_solver(self, x):
        """
        forward model

        :param x:
        :return:
        """
        self.n_feval += 1
        for i_radar in range(self.num_instruments):
            self.list_veg_param[i_radar]['soilcontent'] = x
        sigma0_ = []
        for i_radar, veg_param in enumerate(self.list_veg_param):
            sigma0_ += forest_fwd_model.sigma0_fwd_model(veg_param, use_parallel=False).tolist()
        sigma0 = np.array(sigma0_) * self.measurements_weight

        if np.isnan(sigma0).any():
            print(f'Nan found, {sigma0}')
        return sigma0

    def show_result(self):
        for i_radar in range(self.num_instruments):
            self.list_veg_param[i_radar]['soilcontent'] = self.x_opt

        print(f'Final Value of Cost Function={self.f_opt:g}')
        print(f'In-situ SM ={np.array2string(self.target_sm[0:self.n_lyr] * 100, separator=", ", precision=3)[1:-1]} %vol')
        print(f'Retrieved SM={np.array2string(self.x_opt[0:self.n_lyr] * 100, separator=", ", precision=3)[1:-1]} %vol')

    def get_sm_error_history(self):
        _iter = self.x_opt_history.shape[1]
        x_opt_hist = np.reshape(self.x_opt_history, (self.n_unknown, _iter * self.n_t * self.n_s * self.n_mul))
        x_hist = x_opt_hist[:, np.sum(np.isnan(x_opt_hist), axis=0) < self.n_unknown]
        sm_error_history = x_hist[0, :] - self.target_sm[0]
        return sm_error_history

    def plot_x_history(self, title='Soil_moisture vs iterations', tf_save_fig=False, fig_out_folder='out_images', img_save_name='sm_iteration.png'):

        sm_error = self.get_sm_error_history()
        fig, ax = plt.subplots(1)
        ax.plot(np.arange(len(sm_error)), sm_error, linewidth=1.5)
        ax.grid()
        ax.set_xlabel('Iteration (including internal)')
        ax.set_ylabel('Error in soil moisture')
        plt.tight_layout()
        save_figure(fig, fig_out_folder, img_save_name, tf_save_fig=tf_save_fig)

        return sm_error

    @staticmethod
    def l2_norm_cost_fun(measures, simulated):
        diff_norm = (measures - simulated) / measures
        return np.sum(np.abs(diff_norm) ** 2) / np.size(diff_norm)
