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

# python -u register.py configs/pointcloud/onet_vn_pretrained_cluster.yaml --no-cuda
# python -u register_modelnet.py configs/pointcloud/onet_vn_pretrained_cluster.yaml

# python -u register.py 'configs/pointcloud/30/onet_vn_pretrained_cluster_modelnet40_513512.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/30/onet_vn_pretrained_cluster_modelnet40_513512(1).yaml' --no-cuda
# python -u register.py 'configs/pointcloud/30/onet_vn_pretrained_cluster_modelnet40_24bs.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/30/onet_vn_pretrained_cluster_modelnet40_24bs(1).yaml' --no-cuda

# python -u register.py 'configs/pointcloud/all_angles/onet_vn_pretrained_cluster_modelnet40_24bs_60.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/all_angles/onet_vn_pretrained_cluster_modelnet40_24bs_90.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/all_angles/onet_vn_pretrained_cluster_modelnet40_24bs_120.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/all_angles/onet_vn_pretrained_cluster_modelnet40_24bs_150.yaml' --no-cuda

# python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_00.yaml' --no-cuda
# # python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_30.yaml' --no-cuda
# # python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_60.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_90.yaml' --no-cuda
# # python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_120.yaml' --no-cuda
# # python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_150.yaml' --no-cuda
# python -u register.py 'configs/pointcloud/nonoise/onet_vn_pretrained_cluster_modelnet40_24bs_180.yaml' --no-cuda


# python -u register.py 'configs/pointcloud/centralize/onet_vn_pretrained_cluster_modelnet40_24bs_180.yaml' --no-cuda
python -u register.py 'configs/pointcloud/density/onet_vn_pretrained_cluster_modelnet40_24bs_90.yaml' #--no-cuda