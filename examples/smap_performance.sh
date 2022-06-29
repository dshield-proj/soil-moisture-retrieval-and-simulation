#  DESCRIPTION
#            Calculate SMAP radar performance metrics.
#            Expected run time is 57 minutes using Intel® Core™ i7-10700K processor, which has 16 cores. The RAM requirement is normal less than 16 GB.
#
#   AUTHOR   Amer Melebari
#            Microwave Systems, Sensors and Imaging Lab (MiXiL)
#            University of Southern California (USC)
#   EMAIL    amelebar@usc.edu
#   CREATED  2022-06-26
#   Updated
#
#   Copyright 2022 University of Southern California

veg_path='/media/amer/myfiles/vegetation_param/vegetation-parameters'  # vegetation parameters path

c_dir=$(pwd)

cd ../src || { echo "cd to src dir failed"; exit 127; }
python3 intermediate_product_mapping.py -o dshield_input --gen_empty_inter_file_smap
python3 estimate_sm_performance.py -g  $veg_path -o out_dshield_data --num_trials 10 --in_json_file dshield_input/smap_observations_input.json --out_xls_name smap_sim.xlsx
echo 'Performance metrics of SMAP radar are  in src/out_dshield_data/smap_sim.json and src/out_dshield_data/smap_sim.xlsx'
cd "$c_dir" || { echo "cd to original dir failed"; exit 127; }
