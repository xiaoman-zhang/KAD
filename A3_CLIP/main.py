import argparse
import os
import logging
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import math
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score


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
from dataset.dataset import MIMIC_Dataset,Chestxray14_Dataset,CheXpert_Dataset

from io import BytesIO
from petrel_client.client import Client

conf_path = '~/petreloss.conf'
client = Client(conf_path) 


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

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset #### 
    print("Creating dataset")
    train_dataset = MIMIC_Dataset(config['train_entity_file'],config['train_entity_graph_file'], config['train_fg_query_file'], config['mrsty_file'])
    train_dataloader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )    
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader) 

    val_dataset = Chestxray14_Dataset(config['chestxray_valid_file'])
    val_dataloader =DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            num_workers=8,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader)     

    test_dataset = Chestxray14_Dataset(config['chestxray_test_file'])
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

    test_dataset_chexpert = CheXpert_Dataset(config['chexpert_valid_file'])
    test_dataloader_chexpert =DataLoader(
            test_dataset_chexpert,
            batch_size=config['batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=True,
        )
    test_dataloader_chexpert.num_samples = len(test_dataset_chexpert)
    test_dataloader_chexpert.num_batches = len(test_dataloader_chexpert)  

    if args.image_encoder_name == 'resnet':
        image_encoder = ModelRes(res_base_model='resnet50').to(device) 
    elif args.image_encoder_name == 'densenet':
        image_encoder = ModelDense(dense_base_model='densenet121').to(device) 

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name).to(device=device)

    if args.bert_pretrained:
        with BytesIO(client.get(args.bert_pretrained)) as buffer:
            checkpoint = torch.load(buffer, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        text_encoder.load_state_dict(state_dict)
        print('Load pretrained bert success from: ',args.bert_pretrained)
        if args.freeze_bert:
            for param in text_encoder.parameters():
                param.requires_grad = False
    model = TQN_Model().to(device) 

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model,image_encoder,text_encoder)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) 

    if args.checkpoint:    
        with BytesIO(client.get(args.checkpoint)) as buffer:
                checkpoint = torch.load(buffer, map_location='cpu')
        image_state_dict = checkpoint['image_encoder']     
        image_encoder.load_state_dict(image_state_dict)    
        text_state_dict =  checkpoint['text_encoder']     
        text_encoder.load_state_dict(text_state_dict)     
        state_dict = checkpoint['model']      
        model.load_state_dict(state_dict)    

        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']+1     
        print('load checkpoint from %s'%args.checkpoint)
    
    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir,  'log'))
    best_val_auc = 0.0
    best_val_auc_chexpert = 0.0

    for epoch in range(start_epoch, max_epoch):
        if epoch>0:
            lr_scheduler.step(epoch+warmup_steps)
        train_stats = train(model, image_encoder, text_encoder, tokenizer, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args,config,writer) 

        for k, v in train_stats.items():
            if k == 'loss':
                train_loss_epoch = v
            elif k == 'loss_ce':
                train_loss_ce_epoch = v
            elif k == 'loss_clip':
                train_loss_clip_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/train_loss_ce_epoch', float(train_loss_ce_epoch), epoch)
        writer.add_scalar('loss/train_loss_clip_epoch', float(train_loss_clip_epoch), epoch)
        writer.add_scalar('lr/leaning_rate',  lr_scheduler._get_lr(epoch)[0] , epoch)

        val_loss,val_auc,val_metrics = valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, val_dataloader,epoch,device,args,config,writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('loss/val_auc_epoch', val_auc, epoch)

        chexpert_val_loss, chexpert_val_auc, chexpert_val_metrics = valid_on_cheXpert(model, image_encoder, text_encoder, tokenizer, test_dataloader_chexpert ,epoch,device,args,config,writer)
        writer.add_scalar('loss/chexpert_val_loss_epoch', chexpert_val_loss, epoch)
        writer.add_scalar('loss/chexpert_val_auc_epoch', chexpert_val_auc, epoch)

        if best_val_auc_chexpert < chexpert_val_auc:
            log_stats = {'epoch': epoch, 'chexpert_val_loss': chexpert_val_loss.item(),
                         **{f'chexpert_val_{k}': v for k, v in chexpert_val_metrics.items()},
                        }  
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            best_val_auc_chexpert = chexpert_val_auc
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            with BytesIO() as buffer:
                torch.save(save_obj, buffer)
                client.put(os.path.join(args.aws_output_dir, f"best_valid_chexpert.pt"),buffer.getvalue())
        
        if best_val_auc < val_auc:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write("Save best valid model.\n")
            best_val_auc = val_auc
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            with BytesIO() as buffer:
                torch.save(save_obj, buffer)
                client.put(os.path.join(args.aws_output_dir, f"best_valid.pt"),buffer.getvalue())

            test_loss, test_auc, test_metrics = valid_on_chestxray14(model, image_encoder, text_encoder, tokenizer, test_dataloader,epoch,device,args,config,writer)
            writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
            writer.add_scalar('loss/test_auc_epoch', test_auc, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item(),
                         **{f'val_{k}': v for k, v in val_metrics.items()},
                         'test_loss': test_loss.item(),
                         **{f'test_{k}': v for k, v in test_metrics.items()},
                        }  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item(),
                         **{f'val_{k}': v for k, v in val_metrics.items()},
                        }  
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        if utils.is_main_process():  
            save_obj = {
                'model': model.state_dict(),
                'image_encoder': image_encoder.state_dict(),
                'text_encoder':text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            with BytesIO() as buffer:
                torch.save(save_obj, buffer)
                client.put(os.path.join(args.aws_output_dir, f"checkpoint_state.pt"),buffer.getvalue())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Res_train.yaml')
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--freeze_bert', default=False, type=bool)
    parser.add_argument('--ignore_index', default=False, type=bool)
    parser.add_argument("--use_entity_features", action="store_true")
    parser.add_argument('--image_encoder_name', default='resnet')
    parser.add_argument('--bert_pretrained', default='')
    parser.add_argument('--bert_model_name', default='')
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--aws_output_dir', default='')
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

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")
    main(args, config)

