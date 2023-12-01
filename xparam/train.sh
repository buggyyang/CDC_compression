#!/bin/bash

#SBATCH --job-name=d_eori     # Job name
#SBATCH --gres=gpu:1             # how many gpus would you like to use (here I use 1)
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ruihan.yang@uci.edu  # Where to send mail	(for notification only)
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=5            # Number of CPU cores per task
#SBATCH --mem=16G                  # Job memory request
#SBATCH --time=7-00:00:00              # Time limit hrs:min:sec
#SBATCH --partition=ava_m.p          # partition name
#SBATCH --exclude=ava-m4          # select your node (or not)
#SBATCH --output=logs/job_%j.log   # output log
#SBATCH -a 0


pairs=("l2 cosine")
item=${pairs[$SLURM_ARRAY_TASK_ID]}
lt="${item% *}"
vs="${item#* }"
/home/ruihay1/miniconda3/envs/exp_pytorch/bin/python train.py --iteration_step 8193 \
    --device 0 --loss_type $lt --var_schedule $vs --pred_mode "noise" --beta 0.000001 \
    --aux_weight 0 --reverse_context_dim_mults 4 3 2 1 --ae_path "" --embd_type "01" --load_model --load_step