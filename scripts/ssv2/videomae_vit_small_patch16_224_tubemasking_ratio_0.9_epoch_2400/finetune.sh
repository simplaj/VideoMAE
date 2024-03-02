# Set the path to save checkpoints
OUTPUT_DIR='test_results/pd_finetune'
# path to SSV2 annotation file (train.csv/val.csv/test.csv)
DATA_PATH='/root/proj/VideoMAE/scripts/ssv2/videomae_vit_small_patch16_224_tubemasking_ratio_0.9_epoch_2400'
# path to pretrain model
MODEL_PATH='/root/proj/VideoMAE/scripts/ssv2/videomae_vit_small_patch16_224_tubemasking_ratio_0.9_epoch_2400/checkpoint.pth'
# MODEL_PATH='scripts/ssv2/videomae_vit_small_patch16_224_tubemasking_ratio_0.9_epoch_2400/pd_finetune/checkpoint-19.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python  \
    run_class_finetuning.py \
    --model vit_small_patch16_224 \
    --data_set SSV2 \
    --nb_classes 3 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 4 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 5 \
    --num_frames 16 \
    --opt adamw \
    --lr 7e-4 \
    --layer_decay 0.7 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --epochs 40 \
    --test_num_segment 2 \
    --test_num_crop 3 \
    --mixup 0 \
    --cutmix 0 \
    --smoothing 0 \
    # --eval
    # --dist_eval \
    # --enable_deepspeed 

