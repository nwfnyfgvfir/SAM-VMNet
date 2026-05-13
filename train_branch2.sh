#!/usr/bin/env bash

# ==========================================
# Branch 2 Training Script
# work_dir: results for saving models and logs
# data_path: path to the dataset root
# medsam_path: path to the checkpoint of MedSAM
# branch1_model_path: path to the checkpoint of Branch 1 VM-UNet
# ==========================================

CUDA_VISIBLE_DEVICES=0 python train_branch2.py \
    --batch_size 4 \
    --gpu_id "0" \
    --epochs 5 \
    --work_dir "./result_branch2/" \
    --data_path "./data/vessel/" \
    --medsam_path "./pre_trained_weights/medsam_vit_b.pth" \
    --branch1_model_path "./pre_trained_weights/best-epoch159-loss0.3441.pth"
