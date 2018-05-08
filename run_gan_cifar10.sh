#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=5:59:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=cifar_10_resnet
#SBATCH --mail-user=qc510@nyu.edu
#SBATCH --output=/home/qc510/ds1008_project/wgan_with_group_normalization/outputs_log/slurm_%j.out

module purge
module load h5py/intel/2.7.0rc2
module load cuda/8.0.44
#module load python3/intel/3.6.3
module swap python/intel  python3/intel/3.6.3
module load pytorch/python3.6/0.3.0_4
module load torchvision/python3.5/0.1.9
#module load anaconda3/4.3.1
#module load scikit-learn/intel/0.18.1
module load numpy/python3.6/intel/1.14.0
module load tensorflow/python3.6/1.3.0
module load scipy/intel/0.19.1

cd /home/qc510/ds1008_project/wgan_with_group_normalization
python gan_cifar10.py --NORMALIZATION groupnorm --RESNET True
