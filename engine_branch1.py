import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def _collect_binary_metrics(preds, gts, threshold):
    preds = np.asarray(preds)
    gts = np.asarray(gts)
    y_pre = np.where(preds >= threshold, 1, 0).reshape(-1)
    y_true = np.where(gts >= 0.5, 1, 0).reshape(-1)

    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

    return {
        'confusion_matrix': confusion,
        'accuracy': float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0,
        'sensitivity': float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0,
        'specificity': float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0,
        'f1_or_dsc': float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0,
        'miou': float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0,
        'pred_min': float(preds.min()),
        'pred_max': float(preds.max()),
        'pred_mean': float(preds.mean()),
        'pred_positive_ratio': float(y_pre.mean()),
    }


def train_one_epoch(train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    epoch,
                    step,
                    logger,
                    config,
                    writer,
                    device):
    model.train()
    loss_list = []

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets = data
        images = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        out = model(images)
        loss = criterion(out, targets)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('loss', loss, global_step=step)

        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            print(log_info)
            if logger is not None:
                logger.info(log_info)
    scheduler.step()
    return step


def val_one_epoch(test_loader,
                  model,
                  criterion,
                  epoch,
                  logger,
                  config,
                  device):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            img, msk = data
            img = img.to(device, non_blocking=True).float()
            msk = msk.to(device, non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            gts.append(msk.squeeze(1).cpu().detach().numpy())
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)

    preds = np.array(preds)
    gts = np.array(gts)
    metrics = _collect_binary_metrics(preds, gts, config.threshold)
    metrics['loss'] = float(np.mean(loss_list))

    log_info = (
        f"val epoch: {epoch}, loss: {metrics['loss']:.4f}, miou: {metrics['miou']}, "
        f"f1_or_dsc: {metrics['f1_or_dsc']}, accuracy: {metrics['accuracy']}, "
        f"specificity: {metrics['specificity']}, sensitivity: {metrics['sensitivity']}, "
        f"pred_min: {metrics['pred_min']:.4f}, pred_max: {metrics['pred_max']:.4f}, "
        f"pred_mean: {metrics['pred_mean']:.4f}, pred_positive_ratio: {metrics['pred_positive_ratio']:.6f}, "
        f"confusion_matrix: {metrics['confusion_matrix']}"
    )
    print(log_info)
    logger.info(log_info)

    return metrics


def test_one_epoch(test_loader,
                   model,
                   criterion,
                   logger,
                   config,
                   device,
                   test_data_name=None):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data

            img = img.to(device, non_blocking=True).float()
            msk = msk.to(device, non_blocking=True).float()

            out = model(img)
            loss = criterion(out, msk)

            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out)
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

    preds = np.array(preds)
    gts = np.array(gts)
    metrics = _collect_binary_metrics(preds, gts, config.threshold)
    metrics['loss'] = float(np.mean(loss_list))

    if test_data_name is not None:
        log_info = f'test_datasets_name: {test_data_name}'
        print(log_info)
        logger.info(log_info)
    log_info = (
        f"test of best model, loss: {metrics['loss']:.4f}, miou: {metrics['miou']}, "
        f"f1_or_dsc: {metrics['f1_or_dsc']}, accuracy: {metrics['accuracy']}, "
        f"specificity: {metrics['specificity']}, sensitivity: {metrics['sensitivity']}, "
        f"pred_min: {metrics['pred_min']:.4f}, pred_max: {metrics['pred_max']:.4f}, "
        f"pred_mean: {metrics['pred_mean']:.4f}, pred_positive_ratio: {metrics['pred_positive_ratio']:.6f}, "
        f"confusion_matrix: {metrics['confusion_matrix']}"
    )
    print(log_info)
    logger.info(log_info)

    return metrics
