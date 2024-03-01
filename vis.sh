# Set the path to save video
OUTPUT_DIR='vis_results/pd_finetune'
# path to video for visualization
VIDEO_PATH='/root/proj/PD/MAEVideo/videos_all/train/slight/ill_98_4.mp4'
# path to pretrain model
# MODEL_PATH='TODO/videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0.9_e1600/checkpoint-1599.pth'
MODEL_PATH=/root/proj/VideoMAE/scripts/ssv2/videomae_vit_small_patch16_224_tubemasking_ratio_0.9_epoch_2400/pd_finetune/checkpoint-19.pth

python3 run_videomae_vis.py \
    --mask_ratio 0.9 \
    --mask_type tube \
    --decoder_depth 4 \
    --model vit_small_patch16_224 \
    ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}