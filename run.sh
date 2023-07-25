#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-sml-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=digress
#SBATCH -t 1-00
#SBATCH --output=/nfs/data_chaos/sazzolin/DiGress_fork/sbatch_outputs/sbm.txt
#SBATCH --error=/nfs/data_chaos/sazzolin/DiGress_fork/sbatch_outputs/sbm.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@studenti.unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

set -e
export PATH="/nfs/data_chaos/sazzolin/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate digress
wandb login 2cad8a8279143c69ce071f54bf37c1f5a5f4e5ff

python src/main.py dataset=sbm

echo DONE