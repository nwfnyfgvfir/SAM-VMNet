#!/usr/bin/env bash

# ==========================================
# Branch 1 Testing Script
# data_path: path to the dataset root
# checkpoint_path: checkpoint of the Branch 1 VM-UNet
# output_dir: path for saving raw prediction masks
# ==========================================

CUDA_VISIBLE_DEVICES=0 python test_branch1.py \
    --data_path "./data/vessel/" \
    --checkpoint_path "./pre_trained_weights/best-epoch142-loss0.3230.pth" \
    --output_dir "./data/vessel/test/pred_masks" \
    --gpu_id "0"
