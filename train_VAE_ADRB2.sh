#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --output=./slurm_stdout/slurm-pn-%j.out
#SBATCH --error=./slurm_stdout/slurm-pn-%j.err
#SBATCH --job-name="train_vae_lood_adrb2"

#SBATCH --nodelist=oat0
#SBATCH --array=0  # Used for protein index, useful when creating job arrays?
#SBATCH --partition=msc

#SBATCH --mail-type=ALL
#SBATCH --mail-user="lood.vanniekerk@cs.ox.ac.uk"

export CONDA_ENVS_PATH=/scratch-ssd/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch-ssd/$USER/conda_pkgs

# used to be /scratch-ssd/oatml/scripts/run_locked.sh
/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f protein_env.yml
source /scratch-ssd/oatml/miniconda3/bin/activate protein_env

export EVE_REPO=/users/ms20lvn/EVE

export MSA_data_folder=$EVE_REPO/data/MSA/
export MSA_list=$EVE_REPO/data/mappings/mapping_adrb2.csv
export MSA_weights_location=$EVE_REPO/data/weights/
export VAE_checkpoint_out=$EVE_REPO/results/VAE_checkpoints/
export model_name_suffix='lood19May'
export model_parameters_location=$EVE_REPO/EVE/default_model_params.json
export training_logs_location=$EVE_REPO/logs/$model_name_suffix

srun \
    echo "testing stdout" && \
    echo $SLURM_ARRAY_TASK_ID && \
    python train_VAE.py \
        --MSA_data_folder ${MSA_data_folder} \
        --MSA_list ${MSA_list} \
        --MSA_weights_location ${MSA_weights_location} \
        --protein_index $SLURM_ARRAY_TASK_ID \
        --VAE_checkpoint_location ${VAE_checkpoint_out} \
        --model_name_suffix ${model_name_suffix} \
        --model_parameters_location ${model_parameters_location} \
        --training_logs_location ${training_logs_location} \