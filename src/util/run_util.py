import datetime as dt
import os
import shutil
"""
  DESCRIPTION
            Functions assist running the codes
            
   AUTHOR   Amer Melebari
            Microwave Systems, Sensors and Imaging Lab (MiXiL)
            University of Southern California (USC)
   EMAIL    amelebar@usc.edu
   CREATED  2021-06-22
   Updated  
 
   Copyright 2022 University of Southern California
"""


def check_folder(folder_path: str):
    """
    Check if folder folder_path exist, if not create it
    """
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def backup_file(file_path):
    """
    create a backup of the file by adding ".bakX" where X is a number increasing from 0, the selected value depends on existing backups


    """
    if os.path.isfile(file_path):
        dest_path = file_path
        idx = 0
        while os.path.isfile(dest_path):
            dest_path = f'{file_path}.bak{idx:d}'
            idx += 1
        shutil.move(file_path, dest_path)


def log_print(msg: str):
    """
    Print log to the screen
    """
    print(f'{str(dt.datetime.now())}  {msg}')


def convert_timedelta2str(duration: dt.timedelta):
    """
    Convert duration of timedelta to string
    """
    seconds = duration.total_seconds()
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    out_str = ''
    if hours > 0:
        out_str += f'{int(hours)} hr '
    if minutes > 0:
        out_str += f'{int(minutes)} min '
    out_str += f'{int(seconds)} sec'
    return out_str
