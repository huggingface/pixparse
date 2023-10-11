# Launch scripts examples for each use case. 
# Default config for pretrain looks like
<<comment
TrainCfg:
    train_data:
        source: 'pipe:aws s3 cp s3://pixparse-datasets2/IDL_fghjk_pretraining/idl_shard-{00000..02999}.tar -'
        num_samples: 3201303
        batch_size: 4
        split: ''
        format: 'webdataset'
        num_workers: 4
    eval_data: None
    task:
        TaskCrullerPretrainCfg:
            dtype: None
            amp: True
            model:
                ModelArgs:
                    name: 'cruller_base'
                    image_fmt: None
                    image_size: None
                    text_max_length: None
                    cfg:
                        ModelCfg:
                            type: 'cruller'
                            image_encoder:
                                ImageEncoderCfg:
                                    name: 'vit_base_patch16_224'
                                    image_fmt: 'L'
                                    image_size: 
                                        - 576
                                        - 448
                                    needs_image_size: True
                                    pretrained: True
                            text_decoder:
                                TextDecoderCfg:
                                    name: 'facebook/bart-base'
                                    pretrained: True
                                    num_decoder_layers: 4
                                    max_length: 1024
                                    pad_token_id: None
            tokenizer:
                TokenizerCfg:
                    name: 'facebook/bart-base'
                    pretrained: True
            num_intervals: 100
            num_warmup_intervals: 5
            log_frequency: 100
            metrics_frequency: 1000
            eval_frequency: None
            opt:
                OptimizationCfg:
                    optimizer: 'adamw'
                    scheduler: 'cosine'
                    learning_rate: 0.0005
                    warmup_learning_rate: 0.0
                    weight_decay: 0.02
                    eps: 1e-06
                    clip_grad_value: None
                    clip_grad_mode: None
                    grad_accum_steps: 1
                    momentum: None
                    betas: None
                    layer_decay: None
    experiment: '20231002-130348-task_cruller_pretrain-model_cruller_base-lr_5.0e-04-b_4'
    output_dir: './output'
    log_filename: 'out.log'
    resume_path: ''
    checkpoint_path: ''
    output_checkpoint_dir: './output/20231002-130348-task_cruller_pretrain-model_cruller_base-lr_5.0e-04-b_4/checkpoints'
    seed: 42
    wandb: False
    wandb_project: 'unknown'
    tensorboard: False
    log_eval_data: False
comment


CASE="${1:-cruller_pretrain}"

# PRETRAIN CRULLER

case $CASE in 
"cruller_pretrain")
python -m pixparse.app.train \
  --task cruller_pretrain \
  --model.name cruller_swin_384_to_1920 \
  --train-data.source "pipe:aws s3 cp s3://pixparse-datasets2/IDL_fghjk_pretraining/idl_shard-{00000..02999}.tar -" \
  --train-data.batch-size 8 \
  --train-data.num-samples 3201303 \
  --train-data.num-workers 8 \
  --clip-grad-value 1.0 \
  --clip-grad-mode norm \
  --learning-rate 3e-5 \
  --grad-accum-steps 1 \
  --betas 0.9 0.98 \
  --dtype bfloat16 \
  --num-intervals 30 \
  --num-warmup-intervals 3 \
  --output-checkpoint-dir /fsx/pablo/training_pixparse/ \
  --output-dir /fsx/pablo/training_pixparse/outputs/ \
  --tensorboard True \
  --log-eval-data False \
  --wandb False \
  --log-filename out.log
;;
"cruller_finetune_cord")
python -m pixparse.app.train \
  --task cruller_finetune_cord \
  --model.name cruller_base \
  --train-data.source naver-clova-ix/cord-v2 \
  --train-data.format hf_dataset \
  --train-data.split train \
  --train-data.batch-size 8 \
  --train-data.num-samples 800 \
  --train-data.num-workers 0 \
  --clip-grad-value 1.0 \
  --clip-grad-mode norm \
  --learning-rate 3e-5 \
  --grad-accum-steps 1 \
  --betas 0.9 0.999 \
  --dtype bfloat16 \
  --num-intervals 30 \
  --num-warmup-intervals 3 \
  --resume True \
  --checkpoint-path /fsx/pablo/training_pixparse/20230911-194651-task_cruller_pretrain-model_cruller_larger_6layers-lr_3.0e-05-b_6/checkpoint-12.pt \
  --output-checkpoint-dir /fsx/pablo/training_pixparse/ \
  --output-dir /fsx/pablo/training_pixparse/outputs/ \
  --tensorboard True \
  --log-eval-data False \
  --wandb False \
  --log-filename out.log
;;
"cruller_finetune_rvlcdip")
python -m pixparse.app.train \
  --task cruller_finetune_rvlcdip \
  --model.name cruller_base \
  --train-data.source aharley/rvl_cdip \
  --train-data.format hf_dataset \
  --train-data.split train \
  --train-data.batch-size 8 \
  --train-data.num-samples 320000 \
  --train-data.num-workers 8 \
  --clip-grad-value 1.0 \
  --clip-grad-mode norm \
  --learning-rate 2e-5 \
  --grad-accum-steps 1 \
  --betas 0.9 0.99 \
  --dtype bfloat16 \
  --num-intervals 100 \
  --num-warmup-intervals 1 \
  --resume True \
  --checkpoint-path /fsx/pablo/training_pixparse/20230911-194651-task_cruller_pretrain-model_cruller_larger_6layers-lr_3.0e-05-b_6/checkpoint-18.pt \
  --output-checkpoint-dir /fsx/pablo/training_pixparse/ \
  --output-dir /fsx/pablo/training_pixparse/outputs/ \
  --tensorboard True \
  --log-eval-data False \
  --wandb False \
  --log-filename out.log
;;
*)
echo "No test case selected" 
;;
esac

# EVAL ON CORD

# FINETUNE ON RVLCDIP

# EVAL ON RVLCDIP

# FINETUNE ON DOCVQA (train)

# EVAL ON DOCVQA (val)

# FINETUNE ON DOCVQA (train + val)

# EVAL ON DOCVQA (test)
# no code