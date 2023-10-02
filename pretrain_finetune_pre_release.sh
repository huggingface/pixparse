# Launch scripts examples for each use case. 


# PRETRAIN CRULLER

python -m pixparse.app.train \
  --task-name cruller_pretrain \
  --model-name cruller_swin_384_to_1920 \
  --train-data.source "pipe:aws s3 cp s3://pixparse-datasets2/IDL_fghjk_pretraining/idl_shard-{00000..02999}.tar -" \
  --train-data.batch-size 4 \
  --train-data.num-samples 3201303 \
  --train-data.num-workers 8 \
  --task.opt.clip-grad-value 1.0 \
  --task.opt.clip-grad-mode norm \
  --task.opt.learning-rate 3e-5 \
  --task.opt.grad-accum-steps 1 \
  --task.opt.betas 0.9 0.98 \
  --task.dtype bfloat16 \
  --task.transforms nougat \
  --task.num-intervals 30 \
  --task.num-warmup-intervals 3 \
  --train.output-checkpoint-dir /fsx/pablo/training_pixparse/ \
  --train.output-dir /fsx/pablo/training_pixparse/outputs/ \
  --train.tensorboard True \
  --train.log-eval-data False \
  --train.wandb False \
  --train.log-filename out.log

# FINETUNE ON CORD

# EVAL ON CORD

# FINETUNE ON RVLCDIP

# EVAL ON RVLCDIP

# FINETUNE ON DOCVQA (train)

# EVAL ON DOCVQA (val)

# FINETUNE ON DOCVQA (train + val)

# EVAL ON DOCVQA (test)
# no code