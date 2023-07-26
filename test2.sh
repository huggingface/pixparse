#!/bin/bash
#SBATCH --job-name=pixparse-new-dataset
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=11
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --exclusive
#SBATCH --partition=production-cluster
##SBATCH --partition=dev-cluster
#SBATCH --open-mode=append
#SBATCH --output=/fsx/pablo/slurm_logs/%x_%j.out
#SBATCH --exclude=ip-26-0-144-189,ip-26-0-150-12,ip-26-0-150-9
set -x -e


echo "START TIME: $(date)"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12808
echo $MASTER_ADDR

export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO 
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export NCCL_PROTO=simple
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_LOG_LEVEL=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ens

cd /fsx/pablo/git/pixparse/src
source /admin/home/pablo/.bashrc
conda activate pixparse201

srun --wait=60 --cpu_bind=v python -m pixparse.app.train \
  --data.train.source "pipe:aws s3 cp s3://pixparse-datasets2/IDL_fghjk_pretraining/idl_shard-{00000..02999}.tar -" \
  --data.train.batch-size 32 \
  --data.train.num-samples 3201303 \
  --data.train.num-workers 8 \
  --model-name cruller_base \
  --task.opt.clip-grad-value 1.0 \
  --task.opt.clip-grad-mode norm \
  --task.opt.learning-rate 3e-4 \
  --task.opt.grad-accum-steps 1 \
  --task.opt.betas 0.9 0.99 \
  --task.dtype bfloat16 \
  --task.num-intervals 50 \
  --task.num-warmup-intervals 5 \
  --train.checkpoint-dir /fsx/pablo/training_pixparse/ \
  --train.output-dir /fsx/pablo/training_pixparse/outputs/ \
  --train.experiment cruller_July25th_small_bs32_full_IDL_data_5 \
  --train.tensorboard True \
  --train.log-eval-data False \
  --train.wandb False \
  --train.log-filename out.log


echo "END TIME: $(date)"
