#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=digress
#SBATCH -t 6:00:00
#SBATCH --output=/home/steve.azzolin/DiGress_fork/sbatch_outputs/sbm_bigger.txt
#SBATCH --error=/home/steve.azzolin/DiGress_fork/sbatch_outputs/sbm_bigger.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`


# TODO: enabled wandb when required + generate more graph without TF32

set -e
export PATH="/nfs/data_chaos/sazzolin/miniconda3/bin:$PATH"
export WANDB_CONFIG_DIR=/home/steve.azzolin/wandb
export WANDB_API_KEY=2cad8a8279143c69ce071f54bf37c1f5a5f4e5ff
export HYDRA_FULL_ERROR=1
eval "$(conda shell.bash hook)"
conda activate digress
wandb login

DATASET=sbm
BIGGER=200
python src/main.py \
    dataset=${DATASET} \
    general.test_only=/home/steve.azzolin/DiGress_fork/checkpoints/checkpoint_${DATASET}.ckpt \
    general.name=${DATASET}_${BIGGER} \
    general.sample_bigger_graphs=${BIGGER}


echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes