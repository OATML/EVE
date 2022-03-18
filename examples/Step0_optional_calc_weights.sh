#!/bin/bash
#SBATCH -c 10                             # Request ten cores for parallel weights calc
#SBATCH -N 1                              # Request one node (if you request more than one core with -c, also using
                                          # -N 1 means all cores will be on the same node)
#SBATCH -t 0-5:59                         # Runtime in D-HH:MM format
#SBATCH -p short                          # Partition to run in
#SBATCH --mem=30G                          # Memory total in MB (for all cores)

#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user="lodevicus_vanniekerk@hms.harvard.edu"

#SBATCH --job-name="tmp_eve_weights"
# Job array-specific
# Nice tip: using %3a to pad job array number to 3 digits (23 -> 023)
#SBATCH --output=logs/slurm_files/slurm-lvn-%A_%3a-%x.out  # File to which STDOUT + STDERR will be written, %A: jobID, %a: array task ID, %x: jobname
#SBATCH --array=0-5%10  		  # Job arrays (e.g. 1-100 with a maximum of 5 jobs at once)
##SBATCH --array=0			      # Resubmitting / testing only first job

hostname
pwd
module load gcc/6.2.0

export MSA_data_folder='./data/MSA'
export MSA_list='./data/mappings/example_mapping.csv'
export MSA_weights_location='./data/weights_check/'
export VAE_checkpoint_location='./results/VAE_parameters'
export model_name_suffix='Jan1_PTEN_example'
export model_parameters_location='./EVE/default_model_params.json'
export training_logs_location='./logs/'
export protein_index=$SLURM_ARRAY_TASK_ID

python calc_weights.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index ${protein_index} \
    --MSA_weights_location ${MSA_weights_location} \
    --VAE_checkpoint_location ${VAE_checkpoint_location} \
    --model_name_suffix ${model_name_suffix} \
    --model_parameters_location ${model_parameters_location} \
    --training_logs_location ${training_logs_location}
