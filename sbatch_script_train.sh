#!/bin/bash
#SBATCH --job-name OCC_TRAIN
#SBATCH --nodes=1
#SBATCH --time=150:00:00
#SBATCH --account=hpeng1
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAILNE
#####SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-cpu=2g
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

# python train.py configs/pointcloud/onet_vn_cluster.yaml
# python train.py configs/pointcloud/onet_cluster.yaml
# python train.py configs/pointcloud/onet_vn_cluster_modelnet40_24bs.yaml
# python train.py configs/pointcloud/onet_vn_cluster_modelnet40_24bs_ball_svd.yaml
# python train.py configs/pointcloud/onet_vn_cluster_modelnet40_24bs_ball_svd_dual.yaml
python train.py configs/pointcloud/ovn_cluster.yaml

# python train.py configs/pointcloud/onet_vn_cluster_modelnet40_513c_512pts.yaml
# python train.py configs/pointcloud/onet_vn_cluster_modelnet40_512pts.yaml
# python train.py configs/pointcloud/onet_vn_cluster_modelnet40_513c.yaml