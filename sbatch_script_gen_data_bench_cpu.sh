#!/bin/bash
#SBATCH --job-name OCC_REGIS
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --account=hpeng1
#SBATCH --mail-type=END,FAILNE
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3g
#SBATCH --get-user-env

### # SBATCH --cpus-per-task=1
# conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
# conda activate tp36dup
conda activate pt14
# CUDA_VISIBLE_DEVICES=0 

### This is to print the host name
ipnip=$(hostname -i)
echo ipnip=$ipnip

python -u gen_data_for_bench.py 'configs/pointcloud/ovn_cluster_gen_data.yaml' #--no-cuda