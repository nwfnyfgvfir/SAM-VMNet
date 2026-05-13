import torch
from torch.utils.data import DataLoader
import timm
from dataset import Branch1_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet
from engine_branch1 import *
import os
import sys
from utils import *
from configs.config_setting import setting_config
import warnings
import argparse

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Branch1')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--work_dir', type=str, default='./work_dir/branch1', help='work directory')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    return parser.parse_args()


def main(config, args):
    print('#----------Creating logger----------#')
    config.work_dir = args.work_dir if args.work_dir.endswith('/') else args.work_dir + '/'
    config.data_path = args.data_path
    config.batch_size = args.batch_size
    config.gpu_id = args.gpu_id
    config.epochs = args.epochs

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

    print('#----------GPU init----------#')
    gpu_id = int(config.gpu_id)
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = Branch1_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = Branch1_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)

    test_dataset = Branch1_datasets(config.data_path, config, train=False, test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)

    print('#----------Prepareing Pure VM-UNet----------#')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    model = model.to(device)

    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    best_loss = float('inf')
    best_dice = -1.0
    best_epoch = 1
    start_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        best_loss = checkpoint.get('best_loss', checkpoint.get('min_loss', best_loss))
        best_dice = checkpoint.get('best_dice', best_dice)
        best_epoch = checkpoint.get('best_epoch', checkpoint.get('min_epoch', best_epoch))
        loss = checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, best_loss: {best_loss:.4f}, best_dice: {best_dice:.4f}, best_epoch: {best_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        step = train_one_epoch(
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

        val_metrics = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config,
            device
        )
        loss = val_metrics['loss']
        dice = val_metrics['f1_or_dsc']

        if dice > best_dice or (dice == best_dice and loss < best_loss):
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            best_loss = loss
            best_dice = dice
            best_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'best_loss': best_loss,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'min_loss': best_loss,
                'min_epoch': best_epoch,
                'loss': loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
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


if __name__ == '__main__':
    config = setting_config
    args = parse_args()
    main(config, args)