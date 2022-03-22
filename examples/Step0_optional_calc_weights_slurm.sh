#!/bin/bash
#SBATCH -c 2                             # Request ten cores for parallel weights calc
#SBATCH -N 1                              # Request one node (if you request more than one core with -c, also using
                                          # -N 1 means all cores will be on the same node)
#SBATCH -t 0-1:59                         # Runtime in D-HH:MM format
#SBATCH -p short                          # Partition to run in
#SBATCH --mem=30G                          # Memory total in MB (for all cores)

#SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS
#SBATCH --mail-user="lodevicus_vanniekerk@hms.harvard.edu"

#SBATCH --job-name="eve_weights_2cpu"
# Job array-specific
# Nice tip: using %3a to pad job array number to 3 digits (23 -> 023)
#SBATCH --output=logs/slurm_files/slurm-lvn-%A_%3a-%x.out  # File to which STDOUT + STDERR will be written, %A: jobID, %a: array task ID, %x: jobname
#SBATCH --array=0-73%10  		  # Job arrays (e.g. 1-100 with a maximum of 5 jobs at once)

set -e # fail fully on first line failure (from Joost slurm_for_ml)

# Send python outputs (like print) directly to terminal/log without buffering
export PYTHONUNBUFFERED=1

echo "hostname: $(hostname)"
echo "Running from: $(pwd)"
module load gcc/6.2.0

export CONDA_BIN=/home/lov701/miniconda3/bin/

# Assumes that the conda environment is up to date
echo "Assuming the conda environment protein_env is up to date"
# Note: this sometimes fails on the cluster, maybe I should just use the conda_bin/python3 directly..
#source "$CONDA_BIN"/activate protein_env

export MSA_data_folder="/n/groups/marks/users/lood/DeepSequence_runs/msa_tkmer_20220227"
export MSA_list='./data/mappings/mapping_msa_tkmer_20220227.csv'
export num_cpus="${SLURM_CPUS_PER_TASK}"
export MSA_weights_location='./data/weights_'$num_cpus'cpu/'
export protein_index=$SLURM_ARRAY_TASK_ID

srun /home/lov701/miniconda3/envs/protein_env/bin/python3 calc_weights.py \
    --MSA_data_folder ${MSA_data_folder} \
    --MSA_list ${MSA_list} \
    --protein_index "${protein_index}" \
    --MSA_weights_location "${MSA_weights_location}" \
    --num_cpus "$num_cpus" \
    --calc_method evcouplings
#    --skip_existing