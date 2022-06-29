from typing import Union
"""
  DESCRIPTION
            instruments data and processing functions.

   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-06-22
   Updated

   Copyright 2022 University of Southern California
"""


def list_inc_angles2op_mode(inc_angles: list):
    """
    get list of operation modes from list of science data incidence angles.
    This function generate a single operation mode value from inc_angles

    :param inc_angles: list of incidence angles
    :return:
    """

    inc_angle_list = {35: 0, 45: 0, 55: 0}
    for angle in inc_angles:
        inc_angle_list[angle] += 1
    operation_mode = None
    if inc_angle_list[35] == 0 and inc_angle_list[45] == 0 and inc_angle_list[55] == 0:
        operation_mode = 0
    elif inc_angle_list[35] == 1 and inc_angle_list[45] == 0 and inc_angle_list[55] == 0:
        operation_mode = 1
    elif inc_angle_list[35] == 0 and inc_angle_list[45] == 1 and inc_angle_list[55] == 0:
        operation_mode = 2
    elif inc_angle_list[35] == 0 and inc_angle_list[45] == 0 and inc_angle_list[55] == 1:
        operation_mode = 3
    elif inc_angle_list[35] == 2 and inc_angle_list[45] == 0 and inc_angle_list[55] == 0:
        operation_mode = 4
    elif inc_angle_list[35] == 0 and inc_angle_list[45] == 2 and inc_angle_list[55] == 0:
        operation_mode = 5
    elif inc_angle_list[35] == 0 and inc_angle_list[45] == 0 and inc_angle_list[55] == 2:
        operation_mode = 6
    elif inc_angle_list[35] == 1 and inc_angle_list[45] == 1 and inc_angle_list[55] == 0:
        operation_mode = 7
    elif inc_angle_list[35] == 1 and inc_angle_list[45] == 0 and inc_angle_list[55] == 1:
        operation_mode = 8
    elif inc_angle_list[35] == 0 and inc_angle_list[45] == 1 and inc_angle_list[55] == 1:
        operation_mode = 9

    if operation_mode is None:
        print(f'Not supported mode, {inc_angles}')
        operation_mode = -100
    return operation_mode


def get_sensor_using_using_opmode(sensor_id: Union[int, str], op_mode: int):
    sensors_list = {1: {'center_freq': 1275.7 * 1e6,
                        'polarization': ['vv', 'hh', 'vh'],
                        'instrument_id': 1},
                    2: {'center_freq': 1275.7 * 1e6,
                        'polarization': ['vv', 'hh', 'vh'],
                        'instrument_id': 2},
                    3: {'center_freq': 435 * 1e6,
                        'polarization': ['vv', 'hh', 'vh'],
                        'instrument_id': 3},
                    4: {'center_freq': 435 * 1e6,
                        'polarization': ['vv', 'hh', 'vh'],
                        'instrument_id': 4},
                    9: {'center_freq': (1217 + 1298) / 2 * 1e6,  # SMAP
                        'n_looks_per_km2': 10,
                        'noise_sigma0_db': -30,
                        'inc_angle_deg': 40.0,
                        'num_observation': 1,
                        'instrument_id': 9},
                    'SMAP': {'center_freq': (1217 + 1298) / 2 * 1e6,  # SMAP
                             'n_looks_per_km2': 10,
                             'noise_sigma0_db': -30,
                             'inc_angle_deg': 40.0,
                             'num_observation': 1,
                             'instrument_id': 9}}
    out = sensors_list.get(sensor_id)
    if sensor_id in [9, 'SMAP']:
        smap_op_mode2pol = {1: ['vv', 'hh'], 2: ['vv', 'hh', 'vh']}
        if op_mode not in smap_op_mode2pol.keys():
            raise ValueError(f'expected op_mode {str(smap_op_mode2pol.keys())}, got {op_mode}')
        out['polarization'] = smap_op_mode2pol.get(op_mode)
    elif sensor_id in [1, 2, 3, 4]:
        inc_angles_list = {1: 35, 2: 45, 3: 55, 4: 35, 5: 45, 6: 55}
        num_observation_list = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
        inst12_nesz = {1: -40.69, 2: -37.29, 3: -32.87, 4: -40.69, 5: -37.29, 6: -32.87}
        inst34_nesz = {1: -41.45, 2: -38.29, 3: -35.38, 4: -41.45, 5: -38.29, 6: -35.38}

        inst12_n_looks = {1: 411.14, 2: 506.86, 3: 587.18, 4: 411.14, 5: 506.86, 6: 587.18}
        inst34_n_looks = {1: 4213.17, 2: 5194.91, 3: 6018.15, 4: 4213.17, 5: 5194.91, 6: 6018.15}
        if op_mode not in inc_angles_list.keys():
            raise ValueError(f'expected operation modes are {str(inc_angles_list.keys())}, got {op_mode}')

        out['inc_angle_deg'] = inc_angles_list.get(op_mode)
        out['num_observation'] = num_observation_list.get(op_mode)
        if sensor_id in [1, 2]:
            out['noise_sigma0_db'] = inst12_nesz.get(op_mode)
            out['n_looks_per_km2'] = inst12_n_looks.get(op_mode)
        else:
            out['noise_sigma0_db'] = inst34_nesz.get(op_mode)
            out['n_looks_per_km2'] = inst34_n_looks.get(op_mode)
    else:
        raise RuntimeError(f'sensor Id should be one of the following {str(sensors_list.keys())}, got {sensor_id}')
    return out


def get_sensor_using_inc_angle(sensor_name: Union[int, str], inc_angle: int, num_observation: int = 1):
    inc_angle = int(inc_angle)
    sensors_list = {'l_band': {'center_freq': 1275.7 * 1e6,
                               'polarization': ['vv', 'hh', 'vh'],
                               'instrument_id': 1,
                               'num_observation': num_observation,
                               'name': 'l_band'},
                    'p_band': {'center_freq': 435 * 1e6,
                               'polarization': ['vv', 'hh', 'vh'],
                               'instrument_id': 3,
                               'num_observation': num_observation,
                               'name': 'p_band'},
                    'smap': {'center_freq': (1217 + 1298) / 2 * 1e6,  # SMAP
                             'n_looks_per_km2': 10,
                             'noise_sigma0_db': -30,
                             'inc_angle_deg': 40.0,
                             'num_observation': num_observation,
                             'instrument_id': 9,
                             'name': 'smap',
                             'polarization': ['vv', 'hh']}}

    out = sensors_list.get(sensor_name)
    if sensor_name in ['l_band', 'p_band']:
        name2nesz = {'l_band': {35: -40.69, 45: -37.29, 55: -32.87},
                     'p_band': {35: -41.45, 45: -38.29, 55: -35.38}}
        name2num_look = {'l_band': {35: 411.14, 45: 506.86, 55: 587.18},
                         'p_band': {35: 4213.17, 45: 5194.91, 55: 6018.15}}
        try:
            out['noise_sigma0_db'] = name2nesz[sensor_name][inc_angle]
            out['n_looks_per_km2'] = name2num_look[sensor_name][inc_angle]
            out['inc_angle_deg'] = float(inc_angle)
        except KeyError as e:
            raise ValueError(f'not recognised sensor name and incidence angle, got {sensor_name} and {inc_angle}')
    return out
