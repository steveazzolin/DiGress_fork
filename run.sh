#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=grid_small
#SBATCH -t 2-00:00
#SBATCH --output=/home/steve.azzolin/DiGress_fork/sbatch_outputs/grid_smaller.txt
#SBATCH --error=/home/steve.azzolin/DiGress_fork/sbatch_outputs/grid_smaller.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`


set -e
export PATH="/nfs/data_chaos/sazzolin/miniconda3/bin:$PATH"
export WANDB_CONFIG_DIR=/home/steve.azzolin/wandb
export WANDB_API_KEY=2cad8a8279143c69ce071f54bf37c1f5a5f4e5ff
export HYDRA_FULL_ERROR=1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate digress_ligh2.0.4
wandb login
wandb disabled

DATASET=grid
BIGGER=0
python src/main.py \
    dataset=${DATASET} \
    general.name=${DATASET}_smaller_${BIGGER}_fullprec \
    general.sample_bigger_graphs=${BIGGER} \
    model.n_layers=4 \
    model.hidden_mlp_dims.X=16 \
    model.hidden_mlp_dims.E=16 \
    model.hidden_mlp_dims.y=16 \
    model.hidden_dims.dx=64 \
    model.hidden_dims.n_head=4 \
    model.hidden_dims.dim_ffX=64 \
    train.n_epochs=1000 \
    train.batch_size=32

#Note: Changed hyper-params just for Grid for OOM

#for just testing trained model
#     general.test_only=/home/steve.azzolin/DiGress_fork/checkpoints/checkpoint_${DATASET}.ckpt \

echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes