from core.forest_fwd_model.forest_fwd_model import import_config_file
from util.igbp_veg_class import igbp_class2filename, igbp_class2name
import copy
import os
"""
  DESCRIPTION
            Simulation scenarios.

   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2022-06-22
   Updated

   Copyright 2022 University of Southern California
"""


def get_scenario(scenario_name, veg_param_path):
    scenario_name2igbp = {'walnut_gulch': 7,
                          'tonzi_ranch': 8,
                          'metolius': 1,
                          'las_cruces': 16,
                          'yanco': 12}
    scenario2sm_values = {'walnut_gulch': [0.40016, 0.36879, 0.35471, 0.3383],
                          'tonzi_ranch': [0.23378, 0.22762, 0.22055, 0.2164],
                          'metolius': [0.2124, 0.202, 0.1981, 0.1952],
                          'las_cruces': [0.171, 0.168, 0.167, 0.163],
                          'yanco': [0.3036, 0.2908, 0.2789, 0.2733]}
    scenario2clay_fraction = {'walnut_gulch': 0.19,
                              'tonzi_ranch': 0.20,
                              'metolius': 0.11,  # need to be updated
                              'las_cruces': 0.11,  # need to be updated
                              'yanco': .251}
    if scenario_name not in scenario2sm_values.keys():
        raise ValueError(f'scenario_name expected to be {", ".join(scenario2sm_values.keys())}. Got {scenario_name}')

    veg_file_name = igbp_class2filename(scenario_name2igbp.get(scenario_name))
    veg_file_path = os.path.join(veg_param_path, veg_file_name)
    veg_param = import_config_file(file_name=veg_file_path)
    veg_param['clay_frac'] = scenario2clay_fraction.get(scenario_name)
    veg_param['soilht'] = 0.02

    return {'name': scenario_name,
            'vegetation': copy.copy(veg_param),
            'igbp': scenario_name2igbp.get(scenario_name),
            'sm_list': scenario2sm_values.get(scenario_name)}


def scenario2vegtype(scenario):
    scenario_name2igbp = {'walnut_gulch': 7,
                          'tonzi_ranch': 8,
                          'metolius': 1,
                          'las_cruces': 16,
                          'yanco': 12}
    igbp_ = scenario_name2igbp[scenario]
    class_name = igbp_class2name(igbp_)
    if len(class_name.split(' ')) > 2:
        class_name = ' '.join(class_name.split(' ')[:-1])
    return class_name
