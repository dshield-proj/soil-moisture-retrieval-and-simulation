#  DESCRIPTION
#            Calculate SHIELD radars performance metrics.
#            Expected run time is 232 Hours
#
#            set $veg_path to the path of vegetation parameters folder
#            use debug=true to make sure the code is running
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
debug='y'  # Y
out_folder='out_dshield_data'

c_dir=$(pwd)

cd ../src || { echo "cd to src dir failed"; exit 127; }

python3 intermediate_product_mapping.py -o dshield_input --gen_empty_inter_file
# This will take most of the time
if [[ "$debug" == 'Y' || "$debug" == 'y' ]]
then
echo "Debugging mode, results are not accurate. This should be used for testing the code only."
python3 estimate_sm_performance.py -g  $veg_path -o "$out_folder" --num_trials 10 --in_json_file dshield_input/instruments_inc_angles_and_observations.json --out_xls_name dshield_sim.xlsx --standalone --skip_if_exist --verbose --debug

else
python3 estimate_sm_performance.py -g  $veg_path -o "$out_folder" --num_trials 10 --in_json_file dshield_input/instruments_inc_angles_and_observations.json --out_xls_name dshield_sim.xlsx --standalone --skip_if_exist --verbose
fi
 
# Joining the files
python3 estimate_sm_performance.py -g  $veg_path -o "$out_folder" --num_trials 10 --in_json_file dshield_input/instruments_inc_angles_and_observations.json --out_xls_name dshield_sim.xlsx --join_files

python3 generate_obs_quality_file.py -o obs_quality --inter_json_file "$out_folder"/dshield_sim.json

echo "Performance metrics of D_SHIELD radars are  in src/$out_folder/dshield_sim.json and src/$out_folder/dshield_sim.xlsx"


cd "$c_dir" || { echo "cd to original dir failed"; exit 127; }

