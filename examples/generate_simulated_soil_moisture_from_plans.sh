#  DESCRIPTION
#            Generate simulated soil moisture values from plans files
#            Expected run time (without SMAP data download) is about 10 minutes.
#
#   AUTHOR   Amer Melebari
#            Microwave Systems, Sensors and Imaging Lab (MiXiL)
#            University of Southern California (USC)
#   EMAIL    amelebar@usc.edu
#   CREATED  2022-06-28
#   Updated  
#
#   Copyright 2022 University of Southern California

experiment_path='experiment1'  # Experiment folder relative to src
SMAP_L4_DATA_PATH='/media/amer/Data/smap_l4_sm'  # SMAP L4 data folder


c_dir=$(pwd)

cd ../src || { echo "cd to src dir failed"; exit 127; }

python3 gen_science_data.py "$experiment_path" -r RUN001 --smap_path "$SMAP_L4_DATA_PATH"  --download_smap
echo Outputs are in science_data/RUN001 folder
cd "$c_dir" || { echo "cd to original dir failed"; exit 127; }
