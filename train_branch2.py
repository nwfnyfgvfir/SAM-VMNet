import torch
from torch.utils.data import DataLoader
import timm
import time
import os
import sys
import argparse

from dataset import Branch2_datasets
from tensorboardX import SummaryWriter
from models.vmunet.samvmnet import SAMVMNet
from engine_branch2 import *
from feature_processor import generate_branch1_pred_masks, process_images, validate_branch2_inputs

import matplotlib.pyplot as plt

from utils import *
from configs.config_setting import setting_config
import shutil
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Branch2')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--work_dir', type=str, default='./work_dir/branch2', help='work directory')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--medsam_path', type=str, required=True, help='path to MedSAM model')
    parser.add_argument('--branch1_model_path', type=str, required=True, help='path to trained Branch1 model')
    return parser.parse_args()


def main(config, args):

    config.work_dir = args.work_dir if args.work_dir.endswith('/') else args.work_dir + '/'
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.gpu_id = args.gpu_id
    config.epochs = args.epochs
    config.medsam_path = args.medsam_path
    config.branch1_model_path = args.branch1_model_path

    medsam_model_path = args.medsam_path
    branch1_model_path = args.branch1_model_path

    print('#----------GPU init----------#')
    gpu_id = int(config.gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Generating Branch1 pred_masks----------#')
    generate_branch1_pred_masks(config.data_path, branch1_model_path, device)

    print('#----------Processing images----------#')
    process_images(config.data_path, medsam_model_path)
    validate_branch2_inputs(config.data_path)

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------Preparing dataset----------#')
    train_dataset = Branch2_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = Branch2_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)
    test_dataset = Branch2_datasets(config.data_path, config, train=False, test=True)
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    model = SAMVMNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )

    model.load_from()
    model = model.to(device)

    cal_params_flops_branch2(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    best_loss = float('inf')
    best_miou = -1.0
    best_f1 = -1.0
    start_epoch = 1
    best_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        best_loss = checkpoint.get('best_loss', checkpoint.get('min_loss', float('inf')))
        best_epoch = checkpoint.get('best_epoch', checkpoint.get('min_epoch', 1))
        best_miou = checkpoint.get('best_miou', -1.0)
        best_f1 = checkpoint.get('best_f1_or_dsc', checkpoint.get('best_f1', -1.0))
        loss = checkpoint['loss']

        log_info = (
            f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, '
            f'best_loss: {best_loss:.4f}, best_epoch: {best_epoch}, '
            f'best_miou: {best_miou:.4f}, best_f1_or_dsc: {best_f1:.4f}, loss: {loss:.4f}'
        )
        logger.info(log_info)

    step = 0
    train_losses = []
    val_losses = []
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step, train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            device
        )
        train_losses.append(train_loss)
        val_metrics = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            device
        )
        val_loss = val_metrics['loss']
        val_miou = val_metrics['miou']
        val_f1 = val_metrics['f1_or_dsc']
        val_losses.append(val_loss)

        if (
            val_f1 > best_f1 or
            (val_f1 == best_f1 and val_miou > best_miou) or
            (val_f1 == best_f1 and val_miou == best_miou and val_loss < best_loss)
        ):
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_loss = val_loss
            best_miou = val_miou
            best_f1 = val_f1
            best_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': best_loss,
                'min_epoch': best_epoch,
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'best_miou': best_miou,
                'best_f1_or_dsc': best_f1,
                'loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(os.path.join(checkpoint_dir, 'best.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
            device
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{best_epoch}-loss{best_loss:.4f}.pth')
        )
    return train_losses, val_losses


if __name__ == '__main__':
    config = setting_config
    args = parse_args()
    main(config, args)