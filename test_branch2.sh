#!/usr/bin/env bash

# ==========================================
# Branch 2 Testing Script
# data_path: path to the dataset root
# checkpoint_path: checkpoint of the Branch 2 SAM-VMNet
# work_dir: path for saving logs and output masks
# medsam_path: path to the checkpoint of MedSAM
# branch1_model_path: path to the checkpoint of Branch 1 VM-UNet
# ==========================================

SPLIT="${1:-test}"

CUDA_VISIBLE_DEVICES=0 python test_branch2.py \
    --data_path "./data/vessel/" \
    --checkpoint_path "./pre_trained_weights/best-epoch169-loss0.3444.pth" \
    --work_dir "./result_branch2_${SPLIT}/" \
    --gpu_id "0" \
    --medsam_path "./pre_trained_weights/medsam_vit_b.pth" \
    --branch1_model_path "./pre_trained_weights/best-epoch159-loss0.3441.pth" \
    --split "${SPLIT}"
