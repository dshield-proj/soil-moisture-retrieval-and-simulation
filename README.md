# README #

This repository is for generating soil moisture retrieval performance metrics for D-SHIELD instruments specs and SMAP radar specs.  
It uses a forward model and an optimizer to find the performance metric 

For more info   
Amer Melebari, Sreeja Nag, Vinay Ravindra, Mahta Moghaddam, "Soil Moisture Retrieval from Multi-Instrument and Multi-Frequency Simulated Measurements in Support of Future Earth Observing Systems" IGARSS 2022

This repository does not contain the forward model code. Please contact the author to get the code.  
## System requirement  
This code required both `FORTRAN` and `Python 3.9+`.  
The code should work with any operating system, however, it has been tested on Ubuntu 20.4 and 22.4  
## Requirement installation

You can install the required Python packages using the following code

```bash   
conda install -c conda-forge --file conda-requirements.txt
```

You need to compile the FORTRAN. If you don't have a FORTRAN compiler, the method of installing it in Ubuntu is
```bash
sudo apt-get update
sudo apt-get install gfortran
```

To compile the FORTRAN code (forward model) use one of the following methods:   
Run the following shell code  
```bash  
bash compile_fortran_codes.sh
```

## Example of running the code

### Generate retrieval performance metric   
This generates soil moisture performance metrics using Carlo simulations. The metrics are ubRMSE, RMSE, and bias. The code calculate the mean value 
and the standard deviation for each metric. These metrics are generated for 4 vegetation types.  

#### Generate retrieval performance metric for SMAP radar specs   
First generate empty intermediate JSON file using the following   
```bash   
python3 intermediate_product_mapping.py -o dshield_input --gen_empty_inter_file_smap  
```
`dshield_input` is the path to the output file. This code will generate a JSON file in `dshield_input/smap_observations_input.json`  

Secondly, generate soil moisture retrieval metrics using Mont Carlo simulations using the following code   
```bash 
python3 estimate_sm_performance.py -g VEG_TABLE_PATH -o out_dshield_data --num_trials 10 --in_json_file dshield_input/smap_observations_input.json --out_xls_name smap_sim.xlsx  
```  
where ``VEG_TABLE_PATH`` is the path to the vegetation parameters table, `out_dshield_data` is the output folder path. The output will be `smap_sim.xlsx` and ``smap_sim.json``  
Note: ``--num_trials`` is the number of trials in the Mont Carlo simulation.

#### Generate retrieval performance metric for DSHIELD radars 
First generate empty intermediate JSON file using the following
```bash   
python3 intermediate_product_mapping.py -o dshield_input --gen_empty_inter_file_smap
```
`dshield_input` is the path to the output file. This code will generate JSON file in `instruments_inc_angles_and_observations.json`.  
In this version, the code can generate the number of incidence angles for only the p_band and the l_band radar with three incidence angles: 35, 45, and 55 degrees.  

You can supply the code with list of operation modes of 4 radars: radars 1 & 2 are p_band and radars 3 & 4 are l_band. The format need to be the similar to 
the sample file `sample_instruments_operation_modes.xlsx`.
To run the code with this option use the following example
```bash   
python3 intermediate_product_mapping.py -o dshield_input --gen_empty_inter_file_smap --op_mode_xls_path OM_XLS_PATH
```
`OM_XLS_PATH` is the path to the operation mode Excel file.  

Secondly, generate soil moisture retrieval metrics using Mont Carlo simulations using the following code
```bash 
python3 estimate_sm_performance.py -g VEG_TABLE_PATH -o out_dshield_data --num_trials 10  --in_json_file dshield_input/instruments_inc_angles_and_observations.json --out_xls_name dshield_sim.xlsx --standalone  --skip_if_exist --verbose 
```
where ``VEG_TABLE_PATH`` is the path to the vegetation parameters table, `out_dshield_data` is the output folder path, and ``--num_trials`` is the number of trials in the Mont Carlo simulation. 
The `--standalone` option estimates the performance of each combination in a separate file, each file ends with random code. This is useful when parallelizing the code, as each row can run in a separate machine.  
The option `--skip_if_exist` make the code skip a row if the result of this row is in `dshield_sim.json`.  The `--verbose` option make the code write to the screen the progress details.

The output files will be `dshield_sim_*.xlsx` and ``dshield_sim_*.json``. The * is an 8 character with the first 4 character are the row number and the rest are random characters.   
If you run the code without the option `--standalone`, only two files will be generated; `dshield_sim.json` and `dshield_sim.xlsx`. 

To generate a single file from all these files, run the code with option `--join_files`, i.e.  
```bash 
python3 estimate_sm_performance.py -g VEG_TABLE_PATH -o out_dshield_data --num_trials 10  --out_xls_name dshield_sim.xlsx --join_files 
```
This will generate `dshield_sim.json` and `dshield_sim.xlsx` files.

To generate the observation quality files, run the following command 
```bash 
python3 generate_obs_quality_file.py --inter_json_file dshield_sim.json  -o OUT_FOLDER
```
THe output files are used in generating simulated soil moisture values from experiment plans
### Generate simulated soil moisture values from experiment plans  
This can be done using the following command
```bash
python gen_science_data.py EXPERIMENT_PATH -r RUN001 --smap_path SMAP_L4_DATA_PATH  --download_smap
```
`EXPERIMENT_PATH` is experiment directory path, `RUN001` is the run id value, `SMAP_L4_DATA_PATH` is the path to SMAP L4 data  
If you use option `--download_smap`, SMAP data will be downloaded if it doesn't exist. You need NASA [EARTHDATA](https://urs.earthdata.nasa.gov/) credentials to download the data.

## Who do I talk to?
Amer Melebari  
amelebar@usc.edu  
747-272-4376