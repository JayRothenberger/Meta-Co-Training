#!/bin/bash
#SBATCH --partition=disc
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Thread count:
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
# memory in MB
#SBATCH --mem=500G
# The %04a is translated into a 4-digit number that encodes the SLURM_ARRAY_TASK_ID
#SBATCH --output=/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/slurm/1percent_out_%a.txt
#SBATCH --error=/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/slurm/1percent_err_%a.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=1percent
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/examples
#SBATCH --array=[0-31]%4
#################################################

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | grep -oE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b")

echo Node IP: $head_node_ip


. /home/fagg/tf_setup.sh
conda activate /home/jroth/.conda/envs/mct
wandb login 6ac799cb76304b17ce74f5161bc27f7a80b6ecee

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 2 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "$head_node_ip:64425" \
ensemble_1percent.py
