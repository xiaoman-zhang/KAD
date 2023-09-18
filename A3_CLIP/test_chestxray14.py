import argparse
import os
import logging
import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import csv
import math
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score,accuracy_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from transformers import AutoModel,BertConfig,AutoTokenizer

from factory import utils
from scheduler import create_scheduler
from optim import create_optimizer
from engine.train_fg import train,valid_on_cheXpert,valid_on_chestxray14
from models.clip_tqn import CLP_clinical,ModelRes,ModelDense,TQN_Model,ModelRes512
from models.vit import ModelViT
from dataset.test_dataset import Chestxray14_Dataset


def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    test_dataset = Chestxray14_Dataset(config['chestxray_test_file'],config['img_res'])
    test_dataloader =DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )
    test_dataloader.num_samples = len(test_dataset)
    test_dataloader.num_batches = len(test_dataloader) 

    if args.image_encoder_name == 'resnet':
        if config['img_res'] == 224:
            image_encoder = ModelRes(res_base_model='resnet50').to(device) 
        else:
            image_encoder = ModelRes512(res_base_model='resnet50').to(device) 
    elif args.image_encoder_name == 'densenet':
        image_encoder = ModelDense(dense_base_model='densenet121').to(device) 
    elif args.image_encoder_name == 'vit':
        image_encoder = ModelViT().to(device) 
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name).to(device=device)

    if args.bert_pretrained:
        checkpoint = torch.load(args.bert_pretrained), map_location='cpu')
        state_dict = checkpoint["state_dict"]
        text_encoder.load_state_dict(state_dict)
        print('Load pretrained bert success from: ',args.bert_pretrained)
        if args.freeze_bert:
            for param in text_encoder.parameters():
                param.requires_grad = False
    model = TQN_Model().to(device) 

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    image_state_dict = checkpoint['image_encoder']     
    image_encoder.load_state_dict(image_state_dict)    
    text_state_dict =  checkpoint['text_encoder']     
    text_encoder.load_state_dict(text_state_dict)     
    state_dict = checkpoint['model']      
    model.load_state_dict(state_dict)   
    print('load checkpoint from %s'%args.checkpoint)

    print("Start testing")
    test(model,image_encoder, text_encoder, tokenizer, test_dataloader,device,args,config)

def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features

def test(model,image_encoder, text_encoder, tokenizer, data_loader,device,args,config):
    text_list = ["atelectasis","cardiomegaly","pleural effusion","infiltration","lung mass","lung nodule","pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thicken","hernia"]
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    
    save_result_path = os.path.join(args.output_dir,'result_chestxray14_official.csv')
    dist_csv_col =  ['metric',"atelectasis","cardiomegaly","pleural effusion","infiltration","lung mass","lung nodule","pneumonia","pneumothorax","consolidation","edema","emphysema","fibrosis","pleural thicken","hernia",'mean']
    f_result = open(save_result_path,'w+',newline='')
    wf_result = csv.writer(f_result)
    wf_result.writerow(dist_csv_col)

    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        img_path = sample['img_path']
        image = sample['image'].to(device) 
        label = sample['label'].float().to(device) #batch_size,num_class
        gt = torch.cat((gt, label), 0)

        with torch.no_grad():
            image_features,image_features_pool = image_encoder(image)
            pred_class = model(image_features,text_features)
            pred_class = torch.softmax(pred_class, dim=-1)

            pred = torch.cat((pred, pred_class[:,:,1]), 0) 
    
    AUROCs = compute_AUCs(gt.cpu().numpy(), pred.cpu().numpy())
    mccs,threshold = compute_mccs(gt.cpu().numpy(), pred.cpu().numpy())
    F1s = compute_F1s_threshold(gt.cpu().numpy(), pred.cpu().numpy(),threshold)
    Accs = compute_Accs_threshold(gt.cpu().numpy(), pred.cpu().numpy(),threshold)
    output = []
    output.append('threshold')
    output.append(threshold)
    wf_result.writerow(output)

    wf_result.writerow(AUROCs)
    wf_result.writerow(F1s)
    wf_result.writerow(mccs)
    wf_result.writerow(Accs)
    
    AUROCs_list = []
    mccs_list = []
    F1s_list = []
    Accs_list = []

    data_len = 10000#len(gt)
    for idx in  range(1000):
        randnum = random.randint(0,5000)
        random.seed(randnum)
        gt_idx = random.choices(gt.cpu().numpy(), k=data_len)
        random.seed(randnum)
        pred_idx = random.choices(pred.cpu().numpy(), k=data_len)
        gt_idx = np.array(gt_idx)
        pred_idx = np.array(pred_idx)
        
        AUROCs_idx = compute_AUCs(gt_idx, pred_idx)
        mccs_idx = compute_mccs_threshold(gt_idx, pred_idx,threshold)
        F1s_idx = compute_F1s_threshold(gt_idx,pred_idx,threshold)
        Accs_idx = compute_Accs_threshold(gt_idx,pred_idx,threshold)

        AUROCs_list.append(AUROCs_idx[1:]) #1000,5
        mccs_list.append(mccs_idx[1:])
        F1s_list.append(F1s_idx[1:])
        Accs_list.append(Accs_idx[1:])
    
    AUROCs_5,AUROCs_95,AUROCs_mean = get_sort_eachclass(AUROCs_list)
    output = []
    output.append('perclass_AUROCs_5')
    output.extend(AUROCs_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_AUROCs_95')
    output.extend(AUROCs_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_AUROCs_mean')
    output.extend(AUROCs_mean)
    wf_result.writerow(output)

    mccs_5,mccs_95,mccs_mean = get_sort_eachclass(mccs_list)
    output = []
    output.append('perclass_mccs_5')
    output.extend(mccs_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_mccs_95')
    output.extend(mccs_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_mccs_mean')
    output.extend(mccs_mean)
    wf_result.writerow(output)


    F1s_5,F1s_95,F1s_mean = get_sort_eachclass(F1s_list)
    output = []
    output.append('perclass_F1s_5')
    output.extend(F1s_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_F1s_95')
    output.extend(F1s_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_F1s_mean')
    output.extend(F1s_mean)
    wf_result.writerow(output)
    
    Accs_5,Accs_95,Accs_mean = get_sort_eachclass(Accs_list)
    output = []
    output.append('perclass_Accs_5')
    output.extend(Accs_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_Accs_95')
    output.extend(Accs_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_Accs_mean')
    output.extend(Accs_mean)
    wf_result.writerow(output)
    f_result.close()


def get_sort_eachclass(metric_list,n_class=14):
    metric_5=[]
    metric_95=[]
    metric_mean=[]
    for i in range(n_class):
        sorted_metric_list = sorted(metric_list,key=lambda x:x[i])
        metric_5.append(sorted_metric_list[50][i])
        metric_95.append(sorted_metric_list[950][i])
        metric_mean.append(np.mean(np.array(sorted_metric_list),axis=0)[i])
    mean_metric_5 = np.mean(np.array(metric_5))
    metric_5.append(mean_metric_5)
    mean_metric_95 = np.mean(np.array(metric_95))
    metric_95.append(mean_metric_95)
    mean_metric_mean = np.mean(np.array(metric_mean))
    metric_mean.append(mean_metric_mean)
    return metric_5,metric_95,metric_mean


def compute_AUCs(gt, pred, n_class=14):
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
    AUROCs.append('AUC')
    gt_np = gt 
    pred_np = pred 
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    mean_auc = np.mean(np.array(AUROCs[1:]))
    AUROCs.append(mean_auc)
    return AUROCs

def compute_F1s_threshold(gt, pred,threshold,n_class=14):
    gt_np = gt 
    pred_np = pred 

    F1s = []
    F1s.append('F1s')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        F1s.append(f1_score(gt_np[:, i], pred_np[:, i],average='binary'))#macro
    mean_f1 = np.mean(np.array(F1s[1:]))
    F1s.append(mean_f1)
    return F1s

def compute_Accs_threshold(gt, pred,threshold,n_class=14):
    gt_np = gt 
    pred_np = pred 
    Accs = []
    Accs.append('Accs')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
    mean_accs = np.mean(np.array(Accs[1:]))
    Accs.append(mean_accs)
    return Accs

def compute_mccs_threshold(gt, pred,threshold,n_class=14):
    gt_np = gt 
    pred_np = pred 
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>=threshold[i]]=1
        pred_np[:,i][pred_np[:,i]<threshold[i]]=0
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mccs = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mccs)
    return mccs


def compute_mccs(gt, pred, n_class=14):
    # get a best threshold for all classes
    gt_np = gt 
    pred_np = pred 
    select_best_thresholds =[]
    best_mcc = 0.0

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:,i][pred_np_[:,i]>=thresholds[i]]=1
            pred_np_[:,i][pred_np_[:,i]<thresholds[i]]=0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)

    for i in range(n_class):
        pred_np[:,i][pred_np[:,i]>= select_best_thresholds[i]]=1
        pred_np[:,i][pred_np[:,i]< select_best_thresholds[i]]=0
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mcc)
    return mccs,select_best_thresholds




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Res_train.yaml')
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--freeze_bert', default=False, type=bool)
    parser.add_argument('--use_entity_features', default=True, type=bool)
    parser.add_argument('--image_encoder_name', default='resnet')
    parser.add_argument('--bert_pretrained', default='')
    parser.add_argument('--bert_model_name', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=str,default='5', help='gpu')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--action', default='train')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args, config)
    # python test_chestxray14.py --image_encoder_name  --bert_model_name   --bert_pretrained  --output_dir  --checkpoint 


