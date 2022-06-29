#!/usr/bin/env python
from numba import njit
from scipy.constants import epsilon_0 as eps_0
from typing import Union
import numpy as np

"""
  DESCRIPTION
            Calculate soil dielectric constant using Mironov model

   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2020-08-28
   Updated  2022-06-22 clean up and added comments

   Copyright 2022 University of Southern California
"""


@njit(cache=True)
def cal_reflectivity(theta_i_deg: float, es: float) -> float:
    theta_i = np.deg2rad(theta_i_deg)
    cos_theta_t = np.sqrt(es - np.sin(theta_i) ** 2)
    r_vv = (es * np.cos(theta_i) - cos_theta_t) / (es * np.cos(theta_i) + cos_theta_t)
    r_hh = (np.cos(theta_i) - cos_theta_t) / (np.cos(theta_i) + cos_theta_t)
    lr_lhcp = np.abs(0.5 * (r_vv - r_hh)) ** 2
    return lr_lhcp


@njit(cache=True)
def soil_dielectric_mironov(f_hz: float, soil_moisture: Union[float, np.ndarray], clay_fraction: float) -> Union[complex, np.ndarray]:
    """
    Purpose :
    -------
    Calculate the dielectric constant of a wet soil
    Developed and validated from 1 to 10 GHz.
    adapted for a large range of soil moisture

    References :
    ----------
    [1] Mironov et al, Generalized Refractive Mixing Dielectric Model for moist soil
    IEEE Trans. Geosc. Rem. Sens., vol 42 (4), 773-785. 2004.

    [2] Mironov et al, Physically and Mineralogically Based Spectroscopic Dielectric Model for Moist Soils
    IEEE Trans. Geosc. Rem. Sens., vol 47 (7), 2059-2070. 2009.


    * This code is a direct adaptation of Patricia de Rosnay's Fortran  code collected in ECMWF's CMEM v3.0

    :param f_hz: frequency [Hz]
    :type f_hz: double
    :param soil_moisture: Soil Moisture content [0-1]
    :type soil_moisture: double
    :param clay_fraction: clay fraction [0-1]
    :type clay_fraction: double
    :return: (dielectric,
    :rtype: tuple (complex, double)
    """
    eps_winf = 4.9

    # Initializing the GRMDM spectroscopic parameters with clay fraction
    # RI & NAC of dry soils
    znd = 1.634 - 0.539 * clay_fraction + 0.2748 * clay_fraction ** 2   # [2](17)
    zkd = 0.03952 - 0.04038 * clay_fraction                             # [2](18)
    # Maximum bound water fraction
    zxmvt = 0.02863 + 0.30673 * clay_fraction                           # [2](19)
    # Bound water parameters
    zep0b = 79.8 - 85.4 * clay_fraction + 32.7 * clay_fraction ** 2     # [2](20)
    ztaub = 1.062e-11 + 3.450e-12 * clay_fraction                       # [2](21)
    zsigmab = 0.3112 + 0.467 * clay_fraction                            # [2](22)
    # Unbound (free) water parameters
    zep0u = 100                                                         # [2](24)
    ztauu = 8.5e-12                                                     # [2](25)
    zsigmau = 0.3631 + 1.217 * clay_fraction                            # [2](23)
    # Computation of epsilon water (bound & unbound)  [2](16)
    zcxb = (zep0b - eps_winf) / (1 + (2 * np.pi * f_hz * ztaub) ** 2)
    zepwbx = eps_winf + zcxb
    zepwby = zcxb * (2 * np.pi * f_hz * ztaub) + zsigmab / (2 * np.pi * eps_0 * f_hz)
    zcxu = (zep0u - eps_winf) / (1 + (2 * np.pi * f_hz * ztauu) ** 2)
    zepwux = eps_winf + zcxu
    zepwuy = zcxu * (2 * np.pi * f_hz * ztauu) + zsigmau / (2 * np.pi * eps_0 * f_hz)
    # Computation of refractive index of water (bound & unbound)

    sqrt_2 = np.sqrt(2)
    zepwby_2 = zepwby ** 2
    zepwuy_2 = zepwuy ** 2
    zepwbx_2 = zepwbx ** 2
    zepwux_2 = zepwux ** 2

    znb = np.sqrt(np.sqrt(zepwbx_2 + zepwby_2) + zepwbx) / sqrt_2       # [2](14)
    zkb = np.sqrt(np.sqrt(zepwbx_2 + zepwby_2) - zepwbx) / sqrt_2       # [2](15)
    znu = np.sqrt(np.sqrt(zepwux_2 + zepwuy_2) + zepwux) / sqrt_2       # [2](14)
    zku = np.sqrt(np.sqrt(zepwux_2 + zepwuy_2) - zepwux) / sqrt_2       # [2](15)
    # Computation of soil refractive index (nm & km): xmv can be a vector
    zxmvt2 = np.minimum(soil_moisture, zxmvt)
    zflag = (soil_moisture >= zxmvt)
    znm = znd + (znb - 1) * zxmvt2 + (znu - 1) * (soil_moisture - zxmvt) * zflag   # [2](12)
    zkm = zkd + zkb * zxmvt2 + zku * (soil_moisture - zxmvt) * zflag               # [2](13)
    # Computation of soil dielectric constant:
    zepmx = znm ** 2 - zkm ** 2                                                    # [2](11)
    zepmy = znm * zkm * 2                                                          # [2](11)
    eps = zepmx + 1j * zepmy
    return eps
