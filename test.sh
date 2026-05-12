# ==========================================
# Branch 1 Testing Script
# data_path: path to the training dataset
# pretrained_weight: checkpoint of the branch1 pure VM-UNet
# output_dir: path to the prediction of test set
# ==========================================
python test.py \
    --data_path "./data/vessel/" \
    --pretrained_weight "./pre_trained_weights/best-epoch142-loss0.3488.pth" \
    --device "cuda:0" \
    --output_dir "./data/vessel/test/pred_masks"