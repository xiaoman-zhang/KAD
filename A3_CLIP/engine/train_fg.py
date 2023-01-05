import json
import logging
import math
import os
import cv2
import time
import numpy as np

from PIL import Image
from contextlib import suppress
from itertools import chain
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score


import torch
import torch.nn.functional as F
from torch import nn

from factory import utils
from factory.loss import ClipLoss

try:
    import wandb
except ImportError:
    wandb = None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features


def train(model, image_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    clip_loss = ClipLoss()
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_text_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train()  
    image_encoder.train()  
    text_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_image', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    if args.use_entity_features:
        metric_logger.add_meter('loss_ce_text', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = sample['image'].to(device)  
        label = sample['label'].long().to(device)
        if args.ignore_index:
            pass
        else:
            label[label==-1]=0
        # caption = sample['caption'] #batch_size,len
        entity = sample['entity']

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        text_list = [ 'pleural effusion', 'opacity', 'pneumothorax', 'edema', 'atelectasis',  'tube', 'consolidation','enlarged cardiomediastinum','tip', 'pneumonia','line','cardiomegaly', 'fracture','calcification',
            'device','engorgement',  'nodule', 'wire',  'pacemaker', 'pleural thicken', 'marking', 'scar', 'hyperinflate', 'blunt',  'collapse', 'emphysema', 'aerate', 'mass','infiltration', 'obscure', 'deformity', 'hernia',
            'drainage', 'distention', 'shift', 'stent', 'lesion', 'hardware', 'dilation',  'aspiration']
        # text_list = ["atelectasis","cardiomegaly","consolidation","edema","enlarged cardiomediastinum","fracture","lung lesion","lung opacity","no finding","pleural effusion","pleural other","pneumonia","pneumothorax",'support devices']
        text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
        entity_features = get_text_features(text_encoder,entity,tokenizer,device,max_length=args.max_length)

        image_features,image_features_pool = image_encoder(image)
        pred_class_image = model(image_features,text_features)
        loss_ce_image = ce_loss(pred_class_image.view(-1,2),label.view(-1)) 

        if args.use_entity_features:
            pred_class_text = model(entity_features.unsqueeze(1),text_features)
            loss_ce_text = ce_loss(pred_class_text.view(-1,2),label.view(-1))
            loss_ce = loss_ce_image + loss_ce_text
        else:
            loss_ce = loss_ce_image
        # torch.Size([64, 75, 2]) torch.Size([64, 75, 2]) torch.Size([64, 75])
        # print(pred_class_text.shape,pred_class_image.shape,label.shape)

        loss_clip = clip_loss(image_features_pool,entity_features)
        loss = loss_ce * args.loss_ratio + loss_clip
        loss.backward()
        optimizer.step()    

        writer.add_scalar('loss/loss', loss, scalar_step)
        writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
        writer.add_scalar('loss/loss_ce_image', loss_ce_image, scalar_step)
        if args.use_entity_features:
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
        writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_ce=loss_ce.item())
        metric_logger.update(loss_ce_image=loss_ce_image.item())
        if args.use_entity_features:
            metric_logger.update(loss_ce_text=loss_ce_text.item())
        metric_logger.update(loss_clip=loss_clip.item())


        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        metric_logger.update(lr = scheduler._get_lr(epoch)[0])

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1
        if i % 100 == 0:
            batch_size = len(image)
            num_samples = batch_count * batch_size
            samples_per_epoch = data_loader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(loss.item(), batch_size)
            loss_clip_m.update(loss_clip.item(), batch_size)
            loss_ce_m.update(loss_ce.item(), batch_size)
            loss_ce_image_m.update(loss_ce_image.item(), batch_size)
            if args.use_entity_features:
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Loss_ce_text: {loss_ce_text_m.val:#.5g} ({loss_ce_text_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )
            else:
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size/ batch_time_m.val:#g}/s "
                    f"LR: { scheduler._get_lr(epoch)[0]:5f} "
                )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()

def valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ["atelectasis","cardiomegaly","pleural effusion","infiltration","lung mass","lung nodule","pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thicken","hernia"]
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            pred_class = model(image_features,text_features)#b,14,2
            val_loss = criterion(pred_class.view(-1,2),label.view(-1))

            pred_class = torch.softmax(pred_class, dim=-1)
            pred = torch.cat((pred, pred_class[:,:,1]), 0)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    metrics = compute_AUCs(gt, pred, n_class = 14)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss,AUROC_avg,metrics

def valid_on_cheXpert(model,image_encoder,text_encoder,tokenizer,data_loader, epoch, device, args, config, writer):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    image_encoder.eval()
    text_encoder.eval()
    text_list = ['atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'pleural effusion']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        image = sample['image'].to(device,non_blocking=True)  
        label = sample['label'].long().to(device)
        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            pred_class = model(image_features,text_features)#b,14,2
            val_loss = criterion(pred_class.view(-1,2),label.view(-1))

            pred_class = torch.softmax(pred_class, dim=-1)
            pred = torch.cat((pred, pred_class[:,:,1]), 0)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    metrics = compute_AUCs(gt, pred, n_class=5)
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss,AUROC_avg,metrics

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    metrics = {}
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    metrics[f"mean_auc"] = np.mean(np.array(AUROCs))
    if n_class == 5:
        metrics[f"auc/class_0"]=AUROCs[0]
        metrics[f"auc/class_1"]=AUROCs[1]
        metrics[f"auc/class_2"]=AUROCs[2]
        metrics[f"auc/class_3"]=AUROCs[3]
        metrics[f"auc/class_4"]=AUROCs[4]
    else:
        metrics[f"auc/class_0"]=AUROCs[0]
        metrics[f"auc/class_1"]=AUROCs[1]
        metrics[f"auc/class_2"]=AUROCs[2]
        metrics[f"auc/class_3"]=AUROCs[3]
        metrics[f"auc/class_4"]=AUROCs[4]
        metrics[f"auc/class_5"]=AUROCs[5]
        metrics[f"auc/class_6"]=AUROCs[6]
        metrics[f"auc/class_7"]=AUROCs[7]
        metrics[f"auc/class_8"]=AUROCs[8]
        metrics[f"auc/class_9"]=AUROCs[9]
        metrics[f"auc/class_10"]=AUROCs[10]
        metrics[f"auc/class_11"]=AUROCs[11]
        metrics[f"auc/class_12"]=AUROCs[12]
        metrics[f"auc/class_13"]=AUROCs[13]
    return metrics





