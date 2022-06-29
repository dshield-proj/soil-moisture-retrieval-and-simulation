#!/usr/bin/env python
from typing import Union, Optional
from collections.abc import Callable
import matplotlib.pyplot as plt
import numpy as np
"""
  DESCRIPTION
    Multidirectional-Search-based Simulating annealing method
    
    A. Etminan and M. Moghaddam, "Electromagnetic Imaging of Dielectric Objects Using a Multidirectional-Search-Based Simulated Annealing,
    " in IEEE Journal on Multiscale and Multiphysics Computational Techniques, vol. 3, pp. 167-175, 2018. doi: 10.1109/JMMCT.2018.2875107
                         
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-04-21
   Updated  2021-08-05 Added typing decoration (required Python < 3.9)
            2022-06-22 add comments
            
   Copyright 2022 University of Southern California
"""


class MultiSimAnn:
    """
    Base class for Multidirectional-Search-based Simulating annealing method
    Reference:
    A. Etminan and M. Moghaddam, "Electromagnetic Imaging of Dielectric Objects Using a Multidirectional-Search-Based Simulated Annealing,
    " in IEEE Journal on Multiscale and Multiphysics Computational Techniques, vol. 3, pp. 167-175, 2018. doi: 10.1109/JMMCT.2018.2875107

    Note:
    cost_fun_factory:
            either string or a function with only two inputs: (measures, simulated)
            The string options are:
                'l2': l2 norm
                'l1': l1 norm
                'log': np.sum(((10 * np.log10(simulated) - 10 * np.log10(measures)) / (10 * np.log10(measures))) ** 2)

    """

    def __init__(self, n_unknown: int, itr: int, n_t: int = 1, n_s: int = 20, n_mul: int = 100, temper: float = 10.0, r_t: float = 0.85,
                 c_j: float = 2.0, step_lim: float = 0.0001, f_stop: float = 3.3e-7,
                 overflow_cost: Optional[float] = None, up_limit: Union[float, np.ndarray] = 250., down_limit: Union[float, np.ndarray] = -250.,
                 overflow_option: int = 2, max_n_function_eval: Optional[int] = None, dev_mode: bool = False, silent_mode: bool = False,
                 cost_fun: Union[str, Callable[[np.ndarray, np.ndarray], float]] = 'l2'):
        """

        :param n_unknown: Number of unknowns
        :param itr: Number of Insertions
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
        :param max_n_function_eval: Maximum number of functions evaluations, if None, there is no maximum
        :param dev_mode: if True, the random number will be taken from "rand_seq.mat" (default: False)
        :param silent_mode: if true no output to the screen
        :param cost_fun: The cost function, 'l1', 'l2', 'log', or function with two inputs see note for more info
        """
        self.silent_mode = silent_mode
        self.n_unknown = n_unknown
        self.overflow_cost = 1000.0 if (overflow_cost is None) else overflow_cost
        if 1 < np.size(up_limit) < n_unknown:
            raise ValueError(f'up_limit len should be 1 or n_unknown, got {np.size(up_limit)}, n_unknown is {n_unknown}')

        self.up_limit = up_limit if (np.size(up_limit) > 1) else np.ones(n_unknown) * up_limit
        self.down_limit = down_limit if (np.size(down_limit) > 1) else np.ones(n_unknown) * down_limit
        self.overflow_option = overflow_option  # for choosing the cost value out of x limitations (check fcost for details)
        self.n_feval = None  # number of function evaluations (Will be reset at the start of running the algorithm)
        self.scale_rate = np.ones(n_unknown)
        self.num_iteration = int(itr)
        self.itr = int(itr)
        self.n_t = int(n_t)  # Number of step length adjustment at each temperature
        self.n_s = int(n_s)  # Number of iterations after each step length adjustment
        self.n_mul = int(n_mul)  # Number of iterations of multidirectional search
        self.xx_history = np.ones((self.n_unknown, self.n_unknown, self.itr, self.n_t, self.n_s, self.n_mul)) * np.nan
        self.x_opt_history = np.ones((self.n_unknown, self.itr, self.n_t, self.n_s, self.n_mul)) * np.nan
        self.start_temper = float(temper)
        self.temper = float(temper)
        self.r_t = float(r_t)
        self.c_j = float(c_j)
        self.step_len_adj = (self.up_limit - self.down_limit)
        self.step_lim = float(step_lim)
        self.f_stop = float(f_stop)
        self.x_tar = None
        self.x_opt = None
        self.f_opt = None
        self.dev_mode = dev_mode
        self.max_n_eval = np.inf if (max_n_function_eval is None) else max_n_function_eval
        if type(cost_fun) in [str, np.str_]:
            if cost_fun == 'l2':
                self.cost_fun_factory: Callable[[np.ndarray, np.ndarray], float] = self.l2_norm_cost_fun
            elif cost_fun == 'l1':
                self.cost_fun_factory: Callable[[np.ndarray, np.ndarray], float] = self.l1_norm_cost_fun
            elif cost_fun == 'log':
                self.cost_fun_factory: Callable[[np.ndarray, np.ndarray], float] = self.log_cost_fun
            else:
                raise ValueError(f'Not implemented cost function, {cost_fun}')
        elif callable(cost_fun):
            self.cost_fun_factory: Callable[[np.ndarray, np.ndarray], float] = cost_fun
        else:
            raise ValueError(f'Not implemented cost function, {cost_fun}')
        if dev_mode:
            from scipy.io import loadmat
            mat_file = loadmat('randseq.mat')  # pre-fixed random sequence (for comparison between python and Matlab)
            self.rand_seq = mat_file['rand_seq'][0]
            self.i_rand_seq = 0

    def reset(self):
        self.x_tar = None
        self.x_opt = None
        self.f_opt = None
        self.n_feval = None
        self.temper = self.start_temper
        self.step_len_adj = (self.up_limit - self.down_limit)
        self.itr = self.num_iteration
        self.xx_history = np.ones((self.n_unknown, self.n_unknown, self.itr, self.n_t, self.n_s, self.n_mul)) * np.nan
        self.x_opt_history = np.ones((self.n_unknown, self.itr, self.n_t, self.n_s, self.n_mul)) * np.nan

    def get_initial_x(self) -> np.ndarray:
        """
        Return initial guess of x

        :return: initial guess of x
        """
        x_initial = 100. * np.ones(self.n_unknown)
        return x_initial

    def get_rand_num(self) -> float:
        """
        return a random number

        :return: random number
        """
        if self.dev_mode:
            if self.i_rand_seq >= self.rand_seq.size:
                self.i_rand_seq -= self.rand_seq.size

            rand_num = self.rand_seq[self.i_rand_seq]
            self.i_rand_seq += 1
        else:
            rand_num = np.random.random()

        return rand_num

    def get_measured_fx(self) -> np.ndarray:
        """
        Return the measured f_measured(x)
        :return: measured f_measured(x)
        """
        x_tar = np.zeros(self.n_unknown)  # x measured
        measures = self.bm_solver(x_tar)
        return measures

    def run_multi_sim_ann(self):
        """
        This is the main method for running the method

        :return:
        """
        # initialization
        self.n_feval: int = 0  # number of function evaluations
        self.temper = self.start_temper
        self.itr = self.num_iteration

        costf = np.zeros(self.itr + 1)
        # x_opt_history = np.zeros((self.n_unknown, self.itr))
        # xx = np.zeros((self.n_unknown, self.n_unknown))
        # fxx = np.zeros(self.n_unknown)
        # n_ref_exp_con = np.zeros((self.itr, self.n_t, 3))

        measures = self.get_measured_fx()  # measurements [xxx]
        x_accept = self.get_initial_x()  # initial guess of x
        x_opt = x_accept

        f_x_accept = self.bm_fcost(x_accept, measures)
        f_opt = f_x_accept
        self.overflow_cost = f_opt * 10

        # Iterations
        i: int = 0
        while f_opt > self.f_stop and i < self.itr:
            if self.n_feval > self.max_n_eval:
                break
            if not self.silent_mode:
                print(f'iteration={i + 1:d}')
            # print('f_opt={} max_pixel error={}'.format(f_opt,np.max(np.abs(x_tar-x_opt))))
            costf[i] = f_opt
            # x_opt_history[:, i] = x_opt
            # n_t
            for j in range(self.n_t):  # iterations of step length updates
                if self.n_feval > self.max_n_eval:
                    break

                numerator_update = np.zeros(self.n_unknown)
                denumerator_update = np.zeros(self.n_unknown)
                n_update = np.zeros(self.n_unknown)
                # n_s
                for jj in range(self.n_s):  # iterations after each step length adjustment
                    if self.n_feval > self.max_n_eval:
                        break
                    (fxx, xx, x_accept, f_x_accept, x_opt, f_opt, x_zero, f_x_zero, n_update) = self.bm_initialize4_xx(
                        x_accept, f_x_accept, measures, x_opt, f_opt, n_update)
                    # iterations of multi-directional search
                    k: int = 0
                    k_stop = 0
                    while k < self.n_mul and k_stop == 0:
                        if self.n_feval > self.max_n_eval:
                            break
                        self.xx_history[:, :, i, j, jj, k] = np.array(xx)
                        self.x_opt_history[:, i, j, jj, k] = np.array(x_opt)
                        # Reflection
                        f_opt, f_x_accept, fxx, minval, x_accept, x_opt, xx = self.reflection(denumerator_update, f_opt, f_x_accept,
                                                                                              f_x_zero, k, measures, numerator_update,
                                                                                              x_accept, x_opt, x_zero, xx)

                        # Expansion
                        if minval < f_x_zero:
                            f_opt, f_x_accept, fxx, x_accept, x_opt, xx = self.expansion(f_opt, f_x_accept, fxx, measures, x_accept, x_opt,
                                                                                         x_zero, xx)

                        # contraction
                        else:
                            f_opt, f_x_accept, fxx, x_accept, x_opt, xx = self.contraction(f_opt, f_x_accept, fxx, measures, x_accept,
                                                                                           x_opt, x_zero, xx)

                        minval = np.min(fxx)
                        minloc = np.argmin(fxx)
                        if minval < f_x_zero:
                            xdummy = x_zero
                            x_zero = np.array(xx[:, minloc])
                            xx[:, minloc] = xdummy
                            fdummy = f_x_zero
                            f_x_zero = np.array(fxx[minloc])
                            fxx[minloc] = fdummy

                        # Stopping criteria
                        if k > 0:
                            # max_dis=max(sum(abs(repmat(x_zero,1,n_unknown)-xx)));
                            max_dis = np.max(np.abs(np.tile(np.reshape(x_zero, (self.n_unknown, 1)), (1, self.n_unknown)) - xx))
                            # if max_dis/max(1,sum(abs(x_zero)))<=step_lim
                            if max_dis <= self.step_lim:
                                k_stop = 1

                        k += 1

                self.bm_step_len_adj_update(n_update)

            x_accept = x_opt
            f_x_accept = f_opt
            self.temper = self.temper * self.r_t
            i += 1

        self.itr = i
        costf[self.itr] = f_opt
        # self.x_tar = x_tar * self.scale_rate
        self.x_opt = x_opt * self.scale_rate
        self.f_opt = f_opt

    def contraction(self, f_opt: float, f_x_accept: float, fxx: np.ndarray, measures: np.ndarray, x_accept: np.ndarray, x_opt: np.ndarray,
                    x_zero: np.ndarray, xx: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        fxx, xx = self.bm_contract2(xx, x_zero, measures)
        for kk in range(self.n_unknown):
            f_x_perturb = np.array(fxx[kk])
            x_perturb = np.array(xx[:, kk])
            if f_x_perturb <= f_x_accept:
                x_accept = x_perturb
                f_x_accept = f_x_perturb
                if f_x_perturb < f_opt:
                    x_opt = x_perturb
                    f_opt = f_x_perturb

            else:  # accepting by probability
                selection = self.get_rand_num()
                if selection <= np.exp(-(f_x_perturb - f_x_accept) / self.temper):
                    x_accept = x_perturb
                    f_x_accept = f_x_perturb
        return f_opt, f_x_accept, fxx, x_accept, x_opt, xx

    def expansion(self, f_opt: float, f_x_accept: float, fxx: np.ndarray, measures: np.ndarray, x_accept: np.ndarray, x_opt: np.ndarray,
                  x_zero: np.ndarray, xx: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        fxxe, xxe = self.bm_expand2(xx, x_zero, measures)
        if np.min(fxxe) < np.min(fxx):
            xx = xxe
            fxx = fxxe
        for kk in range(self.n_unknown):
            f_x_perturb = fxxe[kk]
            x_perturb = xxe[:, kk]
            if f_x_perturb <= f_x_accept:
                x_accept = x_perturb
                f_x_accept = f_x_perturb
                if f_x_perturb < f_opt:
                    x_opt = x_perturb
                    f_opt = f_x_perturb

            else:  # accepting by probability
                selection = self.get_rand_num()
                if selection <= np.exp(-(f_x_perturb - f_x_accept) / self.temper):
                    x_accept = x_perturb
                    f_x_accept = f_x_perturb
        return f_opt, f_x_accept, fxx, x_accept, x_opt, xx

    def reflection(self, denominator_update: np.ndarray, f_opt: float, f_x_accept: float, f_x_zero: np.ndarray, k: int, measures: np.ndarray,
                   numerator_update: np.ndarray, x_accept: np.ndarray, x_opt: np.ndarray, x_zero: np.ndarray, xx: np.ndarray) -> \
            tuple[float, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:

        fxx, xx = self.bm_reflect2(xx, x_zero, measures)
        minval = np.min(fxx)
        for kk in range(self.n_unknown):
            f_x_perturb = fxx[kk]
            x_perturb = np.array(xx[:, kk])
            if k == 0:
                denominator_update += np.sign(np.abs(x_zero - x_perturb))
                if f_x_perturb <= f_x_zero:
                    numerator_update += np.sign(np.abs(x_zero - x_perturb))
                else:
                    selection = self.get_rand_num()
                    if selection <= np.exp(-(f_x_perturb - f_x_zero) / self.temper):
                        numerator_update += np.sign(np.abs(x_zero - x_perturb))

            if f_x_perturb <= f_x_accept:
                x_accept = x_perturb
                f_x_accept = f_x_perturb
                if f_x_perturb < f_opt:
                    x_opt = x_perturb
                    f_opt = f_x_perturb

            else:  # accepting by probability
                selection = self.get_rand_num()
                if selection <= np.exp(-(f_x_perturb - f_x_accept) / self.temper):
                    x_accept = x_perturb
                    f_x_accept = f_x_perturb
        return f_opt, f_x_accept, fxx, minval, x_accept, x_opt, xx

    def show_result(self):
        print(f'Final Value of Cost Function={self.f_opt}')
        print(f'Target Values={self.x_tar}')
        print(f'Retrieved Values={self.x_opt}')

    def bm_fcost(self, x_in: np.ndarray, measures: np.ndarray) -> float:

        if np.any(x_in < self.down_limit) or np.any(x_in > self.up_limit):
            if self.overflow_option == 1:  # constant for assignment
                f_out = self.overflow_cost
            elif self.overflow_option == 2:  # higher for more violations
                f_out = 0.0
                for tt in range(self.n_unknown):
                    if x_in[tt] < self.down_limit[tt] or x_in[tt] > self.up_limit[tt]:
                        f_out += self.overflow_cost

            elif self.overflow_option == 3:  # higher for more violations
                f_out = 1.0
                for tt in range(self.n_unknown):
                    if x_in[tt] > self.up_limit[tt]:
                        f_out += (self.overflow_cost * (x_in[tt] - self.up_limit[tt]))
                    elif x_in[tt] < self.down_limit[tt]:
                        f_out += (self.overflow_cost * (self.down_limit[tt] - x_in[tt]))
            else:
                raise ValueError(f"overflow_option should have values 1-3, found {self.overflow_option}")
        else:
            simulated = self.bm_solver(x_in)
            f_out = self.cost_fun_factory(measures, simulated)

        return f_out

    def bm_solver(self, x: np.ndarray) -> np.ndarray:
        """
        forward model, f(x)
        :param x: input parameter
        :return: f(x)
        """

        self.n_feval += 1
        # Rosenbrock function
        y = 0.0
        for jj in range(self.n_unknown):
            y += x[jj] ** 2

        return np.array([y])

    def bm_initialize4_xx(self, x_accept: np.ndarray, f_x_accept: float, measures: np.ndarray, x_opt: np.ndarray, f_opt: float, n_update: np.ndarray)\
            -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:

        xx = np.zeros((self.n_unknown, self.n_unknown))
        dim2 = xx.shape[1]
        fxx = np.zeros(dim2)
        x_zero = x_accept
        f_x_zero = f_x_accept
        for q in range(self.n_unknown):
            xx[:, q] = x_accept
            rand_num = self.get_rand_num()
            xx[q, q] = xx[q, q] + (2 * rand_num) * self.step_len_adj[q]
            fxx[q] = self.bm_fcost(xx[:, q], measures)
            x_perturb = np.array(xx[:, q])
            f_x_perturb = fxx[q]
            if f_x_perturb <= f_x_accept:
                n_update[q] += 1
                x_accept = x_perturb
                f_x_accept = f_x_perturb
                if f_x_perturb < f_opt:
                    x_opt = x_perturb
                    f_opt = f_x_perturb

            else:  # accepting by probability
                selection = self.get_rand_num()
                if selection <= np.exp(-(f_x_perturb - f_x_accept) / self.temper):
                    n_update[q] += 1
                    x_accept = x_perturb
                    f_x_accept = f_x_perturb

        minval = np.min(fxx)
        minloc = np.argmin(fxx)

        if minval < f_x_zero:
            xdummy = x_zero
            x_zero = np.array(xx[:, minloc])
            xx[:, minloc] = xdummy
            fdummy = f_x_zero
            f_x_zero = np.array(fxx[minloc])
            fxx[minloc] = fdummy

        return fxx, xx, x_accept, f_x_accept, x_opt, f_opt, x_zero, f_x_zero, n_update

    def bm_reflect2(self, xx: np.ndarray, x_zero: np.ndarray, measures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        rho = 1.0
        rand_size = 0.5  # determines the randomness size we add to multi-directional steps
        rand_mean = 1.0  # determines the randomness mean we add to multi-directional steps

        xxr = np.zeros_like(xx)
        p_dir = np.zeros_like(xx)

        (dim1, dim2) = xx.shape
        for i in range(dim2):
            # xxe(:,i)=x_zero+(mu*(xx(:,i)-x_zero));
            p_dir[:, i] = rho * (x_zero - xx[:, i])
            # alpha_max = 2.0
            # for j in np.arange(0, dim1, dtype=np.int):
            #    if p_dir[j, i] > 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / p_dir[j, i]))
            #    elif p_dir[j, i] < 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / (-p_dir[j, i])))

            alpha_max = 1.0
            # xxr[:, i] = x_zero + (0.5 + np.random.random()) * alpha_max * p_dir[:, i]
            rand_num = self.get_rand_num()
            xxr[:, i] = x_zero + (rand_mean - rand_size + (rand_size * rand_num * 2)) * alpha_max * p_dir[:, i]

        # mu = -rho
        # xxr = self.cal_xxe(mu, rand_mean, rand_size, x_zero, xx)
        fxxr = self.bm_calculate_fxx(xxr, measures)
        return fxxr, xxr

    def bm_step_len_adj_update(self, n_update: np.ndarray):

        dim = self.step_len_adj.shape[0]
        for cc in range(dim):  # step length update
            update_ratio = n_update[cc] / self.n_s
            if update_ratio > 0.6:
                aa = self.step_len_adj[cc] * (1. + (self.c_j * (update_ratio - 0.6) / 0.4))
                # if aa<=up_limit && aa>=down_limit
                if aa <= (self.up_limit[cc] - self.down_limit[cc]):
                    self.step_len_adj[cc] = aa
                else:
                    self.step_len_adj[cc] = self.up_limit[cc] - self.down_limit[cc]

            elif update_ratio < 0.4:
                aa = self.step_len_adj[cc] / (1 + (self.c_j * (0.4 - update_ratio) / 0.4))
                # if aa<=up_limit && aa>=down_limit
                self.step_len_adj[cc] = aa

    def bm_calculate_fxx(self, xx: np.ndarray, measures: np.ndarray) -> np.ndarray:

        dim2 = xx.shape[1]
        fxx = np.zeros(dim2)
        for i in range(dim2):
            fxx[i] = self.bm_fcost(xx[:, i], measures)

        return fxx

    def bm_expand2(self, xx: np.ndarray, x_zero: np.ndarray, measures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        mu: float = 2.0
        rand_size: float = 0.5  # determines the randomness size we add to multi-directional steps
        rand_mean: float = 1.0  # determines the randomness mean we add to multi-directional steps
        # xxe = self.cal_xxe(mu, rand_mean, rand_size, x_zero, xx)
        xxe = np.zeros_like(xx)
        p_dir = np.zeros_like(xx)

        (dim1, dim2) = xx.shape
        for i in range(dim2):
            # xxe(:,i)=x_zero+(mu*(xx(:,i)-x_zero));
            p_dir[:, i] = mu * (xx[:, i] - x_zero)
            # alpha_max = 2.0
            # for j in np.arange(0, dim1, dtype=np.int):
            #    if p_dir[j, i] > 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / p_dir[j, i]))
            #    elif p_dir[j, i] < 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / (-p_dir[j, i])))

            alpha_max = 1.0
            rand_num = self.get_rand_num()
            xxe[:, i] = x_zero + (rand_mean - rand_size + (rand_size * rand_num * 2)) * alpha_max * p_dir[:, i]

        fxxe = self.bm_calculate_fxx(xxe, measures)
        return fxxe, xxe

    def cal_xxe(self, mu: float, rand_mean: float, rand_size: float, x_zero: np.ndarray, xx: np.ndarray) -> np.ndarray:

        xxe = np.zeros_like(xx)
        p_dir = np.zeros_like(xx)

        (dim1, dim2) = xx.shape
        for i in range(dim2):
            # xxe(:,i)=x_zero+(mu*(xx(:,i)-x_zero));
            p_dir[:, i] = mu * (xx[:, i] - x_zero)
            # alpha_max = 2.0
            # for j in np.arange(0, dim1, dtype=np.int):
            #    if p_dir[j, i] > 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / p_dir[j, i]))
            #    elif p_dir[j, i] < 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / (-p_dir[j, i])))

            alpha_max = 1.0
            rand_num = self.get_rand_num()
            xxe[:, i] = x_zero + (rand_mean - rand_size + (rand_size * rand_num * 2)) * alpha_max * p_dir[:, i]

        return xxe

    def bm_contract2(self, xx: np.ndarray, x_zero: np.ndarray, measures: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        theta = 0.5
        rand_size = 0.5  # determines the randomness size we add to multi-directional steps
        rand_mean = 1.0  # determines the randomness mean we add to multi-directional steps
        xxc = np.zeros_like(xx)
        p_dir = np.zeros_like(xx)

        (dim1, dim2) = xx.shape
        for i in range(dim2):
            p_dir[:, i] = theta * (x_zero - xx[:, i])
            # alpha_max = 2.0
            # for j in np.arange(0, dim1, dtype=np.int):
            #    if p_dir[j, i] > 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / p_dir[j, i]))
            #    elif p_dir[j, i] < 0:
            #        alpha_max = np.min((alpha_max, self.step_len_adj[j] / (-p_dir[j, i])))

            alpha_max = 1.0
            rand_num = self.get_rand_num()
            xxc[:, i] = x_zero + (rand_mean - rand_size + (rand_size * rand_num * 2)) * alpha_max * p_dir[:, i]
        # mu = -theta
        # xxc = self.cal_xxe(mu, rand_mean, rand_size, x_zero, xx)
        fxxc = self.bm_calculate_fxx(xxc, measures)
        return fxxc, xxc

    def bm_plot_cost(self, costf: np.ndarray):
        fig = plt.figure()
        iter_arr = np.arange(0, self.itr)
        plt.plot(iter_arr, costf[0:self.itr])
        plt.xlabel('Iteration number')
        plt.ylabel('Cost function')
        return fig

    @staticmethod
    def _l_norm_cost_fun(measures: np.ndarray, simulated: np.ndarray, norm_type: int = 2) -> float:
        diff = simulated - measures
        if diff.size > 1:
            f_out = np.linalg.norm(diff, norm_type)
        else:
            f_out = np.abs(diff)
        return f_out

    @staticmethod
    def l2_norm_cost_fun(measures: np.ndarray, simulated: np.ndarray) -> float:
        return MultiSimAnn._l_norm_cost_fun(measures, simulated, norm_type=2)

    @staticmethod
    def l1_norm_cost_fun(measures: np.ndarray, simulated: np.ndarray) -> float:
        return MultiSimAnn._l_norm_cost_fun(measures, simulated, norm_type=1)

    @staticmethod
    def log_cost_fun(measures: np.ndarray, simulated: np.ndarray) -> float:
        return float(np.sum(((10 * np.log10(simulated) - 10 * np.log10(measures)) / (10 * np.log10(measures))) ** 2))
