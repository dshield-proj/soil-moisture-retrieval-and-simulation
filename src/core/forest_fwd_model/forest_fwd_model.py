from core.forest_fwd_model import forest_f90  # Fortran function
from core.forest_fwd_model.soil_dielectric import soil_dielectric_mironov
from multiprocessing import Pool
from scipy.constants import speed_of_light
from typing import Optional
import numpy as np

"""
  DESCRIPTION
            Wrapper to the Fortran code forest_f90. 
            This calculate the NRCS of ground and vegetation for backscatter radars
            It is based on 
            M. Burgin, D. Clewley, R. M. Lucas and M. Moghaddam, "A Generalized Radar Backscattering Model Based on Wave Theory for Multilayer 
            Multispecies Vegetation," in IEEE Transactions on Geoscience and Remote Sensing,
            vol. 49, no. 12, pp. 4832-4845, Dec. 2011, doi: 10.1109/TGRS.2011.2172949.

   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-06-08
   Updated  2022-06-22 clean up and added comments

   Copyright 2022 University of Southern California
"""


def db2pwr(db_val): return 10.0 ** (db_val / 10.0)


def import_config_file(file_name: str) -> dict:
    """
    import the vegetation parameters from the .dat file

    :param file_name: file name
    :return: vegetation parameters
    """
    file = open(file_name, 'r')
    input_param = dict()
    for line in file:
        str_temp = [line.replace(',', ' ').replace('(', ' ').replace(')', ' ').replace('=', ' ').split()]
        if len(str_temp[0]) > 2:  # it's a complex number
            temp_val = np.double(str_temp[0][1]) + 1j * np.double(str_temp[0][2])
        else:
            temp_val = np.double(str_temp[0][1])

        input_param[str_temp[0][0]] = temp_val
    file.close()
    return input_param


def _forest_wrap(epsr: np.ndarray, rms_height: np.ndarray, thetainc: float, wavlen: float, nd: int, itype: int, cnpyht: float, lbdiel: complex,
                 lblen: float, lbrad: float, lbradmax: float, lbdens: float, lbexp: int, lborient: int, sbdiel: complex, sblen: float, sbrad: float,
                 sbradmax: float, sbdens: float, sbexp: int, sborient: int, leafdiel: complex, leafd: float, leafrad: float, leafdens: float,
                 trunkdiel: complex, trunklen: float, trunkrad: float, trunkradmax: float, trunkdens: float):
    """
    USE sigma0_fwd_model() instead of this function
    calculate the NRCS values (sigmahh, sigmavv, sigmahv) in dB scale
    This is based on

    M. Burgin, D. Clewley, R. M. Lucas and M. Moghaddam, "A Generalized Radar Backscattering Model Based on Wave Theory for Multilayer Multispecies
    Vegetation," in IEEE Transactions on Geoscience and Remote Sensing,
    vol. 49, no. 12, pp. 4832-4845, Dec. 2011, doi: 10.1109/TGRS.2011.2172949.


    :param epsr: soil dielectric constant (complex)
    :param rms_height: rms surface roughness [m]
    :param thetainc: incidence angle [deg]
    :param wavlen: wavelength [m]
    :param nd:
    :param itype:
    :param cnpyht: canopy height [m]
    :param lbdiel: long branches dielectric constant [m]
    :param lblen:long branches length [m]
    :param lbrad: long branches radius [m]
    :param lbradmax: long branches maximum radius [m]
    :param lbdens: long branch density [#/m3]
    :param lbexp: lang branches pdf exponent
    :param lborient: long branch orientation [deg]
    :param sbdiel: short branches dielectric constant [m]
    :param sblen: short branches length [m]
    :param sbrad: short branches radius [m]
    :param sbradmax: short branches maximum radius [m]
    :param sbdens: short branch density [#/m3]
    :param sbexp: short branches pdf exponent
    :param sborient: short branch orientation [deg]
    :param leafdiel: leaves dielectric constant
    :param leafd: leaves thickness  [m]
    :param leafrad: leaves radius [m]
    :param leafdens: leaves density [#/m3]
    :param trunkdiel: trunk dielectric constant
    :param trunklen: trunk length [m]
    :param trunkrad: trunk radius [m]
    :param trunkradmax: trunk max radius [m]
    :param trunkdens: trunks density [#/m3]
    :return: (sigmahh, sigmavv, sigmahv) [dB]
    """
    if isinstance(epsr, complex):
        epsr = np.ones(10, dtype=complex) * epsr
    elif isinstance(epsr, np.ndarray):
        if len(epsr) == 1:
            epsr = np.ones(10, dtype=complex) * epsr
        elif len(epsr) != 10:
            print('ERROR: epsr need to be np.array of length 10')
    else:
        print('ERROR: epsr need to be np.array of length 10')
    if isinstance(rms_height, float):
        rms_height = np.ones(10, dtype=float) * rms_height
    elif isinstance(rms_height, np.ndarray):
        if len(epsr) == 1:
            rms_height = np.ones(10, dtype=float) * rms_height
        elif len(rms_height) != 10 or rms_height.dtype != float:
            print('ERROR: rms_height need to be np.array of length 10 and type float')
    else:
        print('ERROR: rms_height need to be np.array of length 10 and type float')

    sigmahh, sigmavv, sigmahv = forest_f90.forest(epsr, rms_height, thetainc, wavlen, nd, itype, cnpyht, lbdiel, lblen, lbrad, lbradmax, lbdens,
                                                  lbexp, lborient, sbdiel, sblen, sbrad, sbradmax, sbdens, sbexp, sborient, leafdiel, leafd, leafrad,
                                                  leafdens, trunkdiel, trunklen, trunkrad, trunkradmax, trunkdens)

    return sigmahh, sigmavv, sigmahv


def sigma0_fwd_model(veg_par: dict, use_parallel: bool = True, num_cores: Optional[int] = None) -> np.ndarray:
    """
    calculate the NRCS values (sigmahh, sigmavv, sigmahv) in dB scale
    This is based on

    M. Burgin, D. Clewley, R. M. Lucas and M. Moghaddam, "A Generalized Radar Backscattering Model Based on Wave Theory for Multilayer Multispecies
    Vegetation," in IEEE Transactions on Geoscience and Remote Sensing,
    vol. 49, no. 12, pp. 4832-4845, Dec. 2011, doi: 10.1109/TGRS.2011.2172949.

    :param veg_par: vegetation parameters + incident angles
    :param use_parallel: use parallel processing? (default: True)
    :param num_cores: number of course used in the parallel processing? if None use all the cores (default: None)
    :return: NRCS,  np.ndarray of size (len(veg_par['thetaiList']), len(veg_par['polarization']))
    """
    wavelen = speed_of_light / veg_par['freq']
    eps_r = soil_dielectric_mironov(veg_par['freq'], veg_par['soilcontent'], veg_par['clay_frac'])
    if isinstance(veg_par['polarization'], str):
        veg_par['polarization'] = [veg_par['polarization']]
    n_angles = np.size(veg_par['thetaiList'])

    if use_parallel and n_angles > 1:
        args = ((eps_r, veg_par['soilht'], np.rad2deg(theta_i), wavelen, veg_par['nd'], veg_par['itype'], veg_par['cnpyht'], veg_par['lbdiel'],
                 veg_par['lblen'], veg_par['lbrad'], veg_par['lbradmax'], veg_par['lbdens'], veg_par['lbexp'], veg_par['lborient'], veg_par['sbdiel'],
                 veg_par['sblen'], veg_par['sbrad'], veg_par['sbradmax'], veg_par['sbdens'], veg_par['sbexp'], veg_par['sborient'],
                 veg_par['leafdiel'], veg_par['leafd'], veg_par['leafrad'], veg_par['leafdens'], veg_par['trunkdiel'], veg_par['trunklen'],
                 veg_par['trunkrad'], veg_par['trunkradmax'], veg_par['trunkdens']) for theta_i in veg_par['thetaiList'])
        with Pool(processes=num_cores) as pool:
            sigmahh_db, sigmavv_db, sigmahv_db = pool.starmap(_forest_wrap, args)
        sigmahh_db = np.array(sigmahh_db, dtype=float)
        sigmavv_db = np.array(sigmavv_db, dtype=float)
        sigmahv_db = np.array(sigmahv_db, dtype=float)
    else:
        sigmahh_db = np.zeros(n_angles, dtype=float)
        sigmavv_db = np.zeros(n_angles, dtype=float)
        sigmahv_db = np.zeros(n_angles, dtype=float)
        for i_ang, theta_i in enumerate(veg_par['thetaiList']):
            theta_i_deg = np.rad2deg(theta_i)
            sigmahh_db[i_ang], sigmavv_db[i_ang], sigmahv_db[i_ang] = \
                _forest_wrap(eps_r, veg_par['soilht'], theta_i_deg, wavelen, veg_par['nd'], veg_par['itype'], veg_par['cnpyht'], veg_par['lbdiel'],
                             veg_par['lblen'], veg_par['lbrad'], veg_par['lbradmax'], veg_par['lbdens'], veg_par['lbexp'], veg_par['lborient'],
                             veg_par['sbdiel'], veg_par['sblen'], veg_par['sbrad'], veg_par['sbradmax'], veg_par['sbdens'], veg_par['sbexp'],
                             veg_par['sborient'], veg_par['leafdiel'], veg_par['leafd'], veg_par['leafrad'], veg_par['leafdens'],
                             veg_par['trunkdiel'], veg_par['trunklen'], veg_par['trunkrad'], veg_par['trunkradmax'], veg_par['trunkdens'])

    sigma_db = np.zeros((n_angles, len(veg_par['polarization'])), dtype=float)
    for ipol, pol in enumerate(veg_par['polarization']):
        if pol == 'vv':
            sigma_db[:, ipol] = sigmavv_db
        elif pol == 'hh':
            sigma_db[:, ipol] = sigmahh_db
        elif pol in ['hv', 'vh']:
            sigma_db[:, ipol] = sigmahv_db
        else:
            sigma_db[:, ipol] = np.nan * np.ones(n_angles)

    if n_angles == 1 and len(veg_par['polarization']) == 1:
        return db2pwr(float(sigma_db[0, 0]))
    elif n_angles == 1:
        return db2pwr(sigma_db[0, :])
    elif len(veg_par['polarization']) == 1:
        return db2pwr(sigma_db[:, 0])
    else:
        return db2pwr(sigma_db)
