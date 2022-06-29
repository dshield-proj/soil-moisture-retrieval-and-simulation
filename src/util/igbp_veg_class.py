from typing import Optional


"""
  DESCRIPTION
            Descriptions of the classes are found in 
            http://www.eomf.ou.edu/static/IGBP.pdf
             
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-09-29
   Updated  2022-06-22: clean up and add comments
 
   Copyright 2022 University of Southern California
"""


def igbp_class2name(class_id: int) -> Optional[str]:
    igbp_class_name = {1: 'Evergreen needleleaf forests',
                       2: 'Evergreen broadleaf forests',
                       3: 'Deciduous needleleaf forests',
                       4: 'Deciduous broadleaf forests',
                       5: 'Mixed forests',
                       6: 'Closed shrublands',
                       7: 'Open shrublands',
                       8: 'Woody savannas',
                       9: 'Savannas',
                       10: 'Grasslands',
                       11: 'Permanent wetlands',
                       12: 'Croplands',
                       13: 'Urban and built-up lands',
                       14: 'Cropland/natural vegetation mosaics',
                       15: 'Snow and ice',
                       16: 'Barren',
                       17: 'Water bodies'}
    return igbp_class_name.get(class_id)


def igbp_class2filename(class_id: int) -> Optional[str]:
    igbp_class_fn = {1: 'EvergreenNeedle_OJPYJP.dat',
                     2: 'EvergreenBroadleaf_LaSelvaAspen.dat',
                     3: 'DeciduousNeedle_OJPYJP.dat',
                     4: 'DeciduousBroadleaf_Aspen.dat',
                     5: 'MixedForest_OJPAspen.dat',
                     6: 'ClosedShrub_DwyerSouth58.dat',
                     7: 'OpenShrub_DwyerNorth24.dat',
                     8: 'WoodySavanna_Bowen47.dat',
                     9: None,
                     10: 'Grasslands.dat',
                     11: 'Wetlands_Fen.dat',
                     12: 'Grasslands.dat',
                     13: None,
                     14: None,
                     15: None,
                     16: 'Barren_no_vegetation.dat',
                     17: None}
    return igbp_class_fn.get(class_id)

