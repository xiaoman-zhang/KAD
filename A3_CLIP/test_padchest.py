import argparse
import os
import logging
import ruamel.yaml as yaml
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
from models.clip_tqn import CLP_clinical,ModelRes,ModelDense,TQN_Model
from dataset.test_dataset import Padchest_Dataset



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
    test_dataset = Padchest_Dataset(config['padchest_all_test_file'],config['img_res'])
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
    print(len(test_dataset),len(test_dataloader) )

    if args.image_encoder_name == 'resnet':
        image_encoder = ModelRes(res_base_model='resnet50').to(device) 
    elif args.image_encoder_name == 'densenet':
        image_encoder = ModelDense(dense_base_model='densenet121').to(device) 
    elif args.image_encoder_name == 'vit':
        image_encoder = ModelViT().to(device) 
    
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name).to(device=device)

    if args.bert_pretrained:
        checkpoint = torch.load(args.bert_pretrained, map_location='cpu')
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
    text_list = ['normal', 'pulmonary fibrosis', 'chronic changes', 'kyphosis', 'pseudonodule', 'ground glass pattern', 'unchanged', 'alveolar pattern', 'interstitial pattern', 'laminar atelectasis', 'pleural effusion', 'apical pleural thickening', 'suture material', 'sternotomy', 'endotracheal tube', 'infiltrates', 'heart insufficiency', 'hemidiaphragm elevation', 'superior mediastinal enlargement', 'aortic elongation', 'scoliosis', 'sclerotic bone lesion', 'supra aortic elongation', 'vertebral degenerative changes', 'goiter', 'COPD signs', 'air trapping', 'descendent aortic elongation', 'aortic atheromatosis', 'metal', 'hypoexpansion basal', 'abnormal foreign body', 'central venous catheter via subclavian vein', 'central venous catheter', 'vascular hilar enlargement', 'pacemaker', 'atelectasis', 'vertebral anterior compression', 'hiatal hernia', 'pneumonia', 'diaphragmatic eventration', 'consolidation', 'calcified densities', 'cardiomegaly', 'fibrotic band', 'tuberculosis sequelae', 'volume loss', 'bronchiectasis', 'single chamber device', 'emphysema', 'vertebral compression', 'bronchovascular markings', 'bullas', 'hilar congestion', 'exclude', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'pulmonary mass', 'dual chamber device', 'increased density', 'surgery neck', 'osteosynthesis material', 'costochondral junction hypertrophy', 'segmental atelectasis', 'costophrenic angle blunting', 'calcified pleural thickening', 'hyperinflated lung', 'callus rib fracture', 'pleural thickening', 'mediastinal mass', 'nipple shadow', 'surgery heart', 'pulmonary artery hypertension', 'central vascular redistribution', 'tuberculosis', 'nodule', 'cavitation', 'granuloma', 'osteopenia', 'lobar atelectasis', 'surgery breast', 'NSG tube', 'hilar enlargement', 'gynecomastia', 'atypical pneumonia', 'cervical rib', 'mediastinal enlargement', 'major fissure thickening', 'surgery', 'azygos lobe', 'adenopathy', 'miliary opacities', 'suboptimal study', 'dai', 'mediastinic lipomatosis', 'surgery lung', 'mammary prosthesis', 'humeral fracture', 'calcified adenopathy', 'reservoir central venous catheter', 'vascular redistribution', 'hypoexpansion', 'heart valve calcified', 'pleural mass', 'loculated pleural effusion', 'pectum carinatum', 'subacromial space narrowing', 'central venous catheter via jugular vein', 'vertebral fracture', 'osteoporosis', 'bone metastasis', 'lung metastasis', 'cyst', 'humeral prosthesis', 'artificial heart valve', 'mastectomy', 'pericardial effusion', 'lytic bone lesion', 'subcutaneous emphysema', 'pulmonary edema', 'flattened diaphragm', 'asbestosis signs', 'multiple nodules', 'prosthesis', 'pulmonary hypertension', 'soft tissue mass', 'tracheostomy tube', 'endoprosthesis', 'post radiotherapy changes', 'air bronchogram', 'pectum excavatum', 'calcified mediastinal adenopathy', 'central venous catheter via umbilical vein', 'thoracic cage deformation', 'obesity', 'tracheal shift', 'external foreign body', 'atelectasis basal', 'aortic endoprosthesis', 'rib fracture', 'calcified fibroadenoma', 'pneumothorax', 'reticulonodular interstitial pattern', 'reticular interstitial pattern', 'chest drain tube', 'minor fissure thickening', 'fissure thickening', 'hydropneumothorax', 'breast mass', 'blastic bone lesion', 'respiratory distress', 'azygoesophageal recess shift', 'ascendent aortic elongation', 'lung vascular paucity', 'kerley lines', 'electrical device', 'artificial mitral heart valve', 'artificial aortic heart valve', 'total atelectasis', 'non axial articular degenerative changes', 'pleural plaques', 'calcified pleural plaques', 'lymphangitis carcinomatosa', 'lepidic adenocarcinoma', 'mediastinal shift', 'ventriculoperitoneal drain tube', 'esophagic dilatation', 'dextrocardia', 'end on vessel', 'right sided aortic arch', 'Chilaiditi sign', 'aortic aneurysm', 'loculated fissural effusion', 'fracture', 'air fluid level', 'round atelectasis', 'mass', 'double J stent', 'pneumoperitoneo', 'abscess', 'pulmonary artery enlargement', 'bone cement', 'pneumomediastinum', 'catheter', 'surgery humeral', 'empyema', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary venous hypertension', 'gastrostomy tube', 'lipomatosis']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    
    save_result_path = os.path.join(args.output_dir,'result_padchest_paperlist.csv')
    dist_csv_col =  ['metric', 'normal', 'pulmonary fibrosis', 'chronic changes', 'kyphosis', 'pseudonodule', 'ground glass pattern', 'unchanged', 'alveolar pattern', 'interstitial pattern', 'laminar atelectasis', 'pleural effusion', 'apical pleural thickening', 'suture material', 'sternotomy', 'endotracheal tube', 'infiltrates', 'heart insufficiency', 'hemidiaphragm elevation', 'superior mediastinal enlargement', 'aortic elongation', 'scoliosis', 'sclerotic bone lesion', 'supra aortic elongation', 'vertebral degenerative changes', 'goiter', 'COPD signs', 'air trapping', 'descendent aortic elongation', 'aortic atheromatosis', 'metal', 'hypoexpansion basal', 'abnormal foreign body', 'central venous catheter via subclavian vein', 'central venous catheter', 'vascular hilar enlargement', 'pacemaker', 'atelectasis', 'vertebral anterior compression', 'hiatal hernia', 'pneumonia', 'diaphragmatic eventration', 'consolidation', 'calcified densities', 'cardiomegaly', 'fibrotic band', 'tuberculosis sequelae', 'volume loss', 'bronchiectasis', 'single chamber device', 'emphysema', 'vertebral compression', 'bronchovascular markings', 'bullas', 'hilar congestion', 'exclude', 'axial hyperostosis', 'aortic button enlargement', 'calcified granuloma', 'clavicle fracture', 'pulmonary mass', 'dual chamber device', 'increased density', 'surgery neck', 'osteosynthesis material', 'costochondral junction hypertrophy', 'segmental atelectasis', 'costophrenic angle blunting', 'calcified pleural thickening', 'hyperinflated lung', 'callus rib fracture', 'pleural thickening', 'mediastinal mass', 'nipple shadow', 'surgery heart', 'pulmonary artery hypertension', 'central vascular redistribution', 'tuberculosis', 'nodule', 'cavitation', 'granuloma', 'osteopenia', 'lobar atelectasis', 'surgery breast', 'NSG tube', 'hilar enlargement', 'gynecomastia', 'atypical pneumonia', 'cervical rib', 'mediastinal enlargement', 'major fissure thickening', 'surgery', 'azygos lobe', 'adenopathy', 'miliary opacities', 'suboptimal study', 'dai', 'mediastinic lipomatosis', 'surgery lung', 'mammary prosthesis', 'humeral fracture', 'calcified adenopathy', 'reservoir central venous catheter', 'vascular redistribution', 'hypoexpansion', 'heart valve calcified', 'pleural mass', 'loculated pleural effusion', 'pectum carinatum', 'subacromial space narrowing', 'central venous catheter via jugular vein', 'vertebral fracture', 'osteoporosis', 'bone metastasis', 'lung metastasis', 'cyst', 'humeral prosthesis', 'artificial heart valve', 'mastectomy', 'pericardial effusion', 'lytic bone lesion', 'subcutaneous emphysema', 'pulmonary edema', 'flattened diaphragm', 'asbestosis signs', 'multiple nodules', 'prosthesis', 'pulmonary hypertension', 'soft tissue mass', 'tracheostomy tube', 'endoprosthesis', 'post radiotherapy changes', 'air bronchogram', 'pectum excavatum', 'calcified mediastinal adenopathy', 'central venous catheter via umbilical vein', 'thoracic cage deformation', 'obesity', 'tracheal shift', 'external foreign body', 'atelectasis basal', 'aortic endoprosthesis', 'rib fracture', 'calcified fibroadenoma', 'pneumothorax', 'reticulonodular interstitial pattern', 'reticular interstitial pattern', 'chest drain tube', 'minor fissure thickening', 'fissure thickening', 'hydropneumothorax', 'breast mass', 'blastic bone lesion', 'respiratory distress', 'azygoesophageal recess shift', 'ascendent aortic elongation', 'lung vascular paucity', 'kerley lines', 'electrical device', 'artificial mitral heart valve', 'artificial aortic heart valve', 'total atelectasis', 'non axial articular degenerative changes', 'pleural plaques', 'calcified pleural plaques', 'lymphangitis carcinomatosa', 'lepidic adenocarcinoma', 'mediastinal shift', 'ventriculoperitoneal drain tube', 'esophagic dilatation', 'dextrocardia', 'end on vessel', 'right sided aortic arch', 'Chilaiditi sign', 'aortic aneurysm', 'loculated fissural effusion', 'fracture', 'air fluid level', 'round atelectasis', 'mass', 'double J stent', 'pneumoperitoneo', 'abscess', 'pulmonary artery enlargement', 'bone cement', 'pneumomediastinum', 'catheter', 'surgery humeral', 'empyema', 'nephrostomy tube', 'sternoclavicular junction hypertrophy', 'pulmonary venous hypertension', 'gastrostomy tube', 'lipomatosis','mean']
    
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
    wf_result.writerow(AUROCs)
    
    AUROCs_list = []
    data_len = len(gt)
    for idx in  range(1000):
        randnum = random.randint(0,5000)
        random.seed(randnum)
        gt_idx = random.choices(gt.cpu().numpy(), k=data_len)
        random.seed(randnum)
        pred_idx = random.choices(pred.cpu().numpy(), k=data_len)
        gt_idx = np.array(gt_idx)
        pred_idx = np.array(pred_idx)
        
        AUROCs_idx = compute_AUCs(gt_idx, pred_idx)
        AUROCs_list.append(AUROCs_idx[1:]) #1000,5

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
    f_result.close()


def compute_Accs_threshold(gt, pred,threshold,n_class=193):
    gt_np = gt 
    pred_np = pred 
    
    pred_np[pred_np>threshold]=1
    pred_np[pred_np<threshold]=0
    Accs = []
    Accs.append('Accs')
    for i in range(n_class):
       Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
    mean_accs = np.mean(np.array(Accs[1:]))
    Accs.append(mean_accs)
    return Accs

def get_sort_eachclass(metric_list,n_class=193):
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

def compute_AUCs(gt, pred, n_class=193):
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

def compute_F1s_threshold(gt, pred,threshold,n_class=193):
    bert_f1 = 0.0
    gt_np = gt 
    pred_np = pred 
    
    pred_np[pred_np>threshold]=1
    pred_np[pred_np<threshold]=0
    F1s = []
    F1s.append('F1s')
    for i in range(n_class):
       F1s.append(f1_score(gt_np[:, i], pred_np[:, i],average='macro'))
    mean_f1 = np.mean(np.array(F1s[1:]))
    F1s.append(mean_f1)
    return F1s



def compute_mccs(gt, pred, n_class=193):
    # get a best threshold for all classes
    gt_np = gt 
    pred_np = pred 
    select_best_thresholds = 0.0
    best_mcc = 0.0

    for thresholds in np.linspace(0.0, 1.0, 21):
        pred_np_ = pred_np.copy()
        pred_np_[pred_np_>thresholds]=1
        pred_np_[pred_np_<thresholds]=0
        mccs = []
        for i in range(n_class):
            mccs.append(matthews_corrcoef(gt_np[:, i], pred_np_[:, i]))
        mean_mcc = np.mean(np.array(mccs))
        if mean_mcc > best_mcc:
            select_best_thresholds = thresholds
            best_mcc = mean_mcc

    pred_np[pred_np>select_best_thresholds]=1
    pred_np[pred_np<select_best_thresholds]=0
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
    parser.add_argument('--ignore_index', default=False, type=bool)
    parser.add_argument('--use_entity_features', default=True, type=bool)
    parser.add_argument('--image_encoder_name', default='resnet')
    parser.add_argument('--bert_pretrained', default='')
    parser.add_argument('--bert_model_name', default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
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

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    

    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args, config)

