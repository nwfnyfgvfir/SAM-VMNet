#!/usr/bin/env bash

# ==========================================
# Branch 1 Training Script
# work_dir: results for saving models and logs
# data_path: path to the dataset root
# ==========================================

CUDA_VISIBLE_DEVICES=0 python train_branch1.py \
    --batch_size 8 \
    --gpu_id "0" \
    --epochs 200 \
    --work_dir "./result_branch1/" \
    --data_path "./data/vessel/"
