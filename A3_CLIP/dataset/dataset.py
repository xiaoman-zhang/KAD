import csv
import json
import logging
import os
import re
import sys
from abc import abstractmethod
from itertools import islice
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader
import PIL
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from collections import defaultdict
from PIL import Image
import cv2

from dataset.randaugment import RandomAugment


def open_jpg(url):
    img = Image.opem(url)
    return img

class MIMIC_Dataset(Dataset):
    def __init__(self, umls_json_path, radgraph_json_path, csv_path, sty_path,img_res):
        self.umls_json_info = json.load(open(umls_json_path,'r'))
        self.radgraph_json_info = json.load(open(radgraph_json_path,'r'))

        self.entity_label_dict = {
            'ANAT-DP': 'Anatomy Definitely Present',
            'OBS-DP': 'Observation Definitely Present',
            'OBS-DA': 'Observation Definitely Absent',
            'OBS-U': 'Observation Uncertain',
        }

        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0]) #348900
        self.class_list = np.asarray(data_info.iloc[:,1:])#40 class for fine-grained query list
        sty_info = pd.read_csv(sty_path)
        self.sty_dict_info = self.csv_to_dict(sty_info)

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop(img_res,scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])  

    
    def csv_to_dict(self,sty_info):
        tui_list = sty_info.iloc[:,0]
        sty_list = sty_info.iloc[:,1]
        sty_dict = defaultdict(list)
        for idx in tqdm(range(len(tui_list))):
            tui_idx = tui_list[idx]
            sty_idx = sty_list[idx]
            sty_dict[tui_idx] = sty_idx
        return sty_dict
    
    def __len__(self):
        return len(self.img_path_list)
    
    def get_entity_list(self, entities):
        entity_dict = defaultdict(list)
        entities_num = len(entities)
        for idx in range(entities_num):
            entity_idx = entities[str(idx+1)]
            token_idx = entity_idx['tokens']
            label_idx = self.entity_label_dict[entity_idx['label']]
            entity_dict[token_idx] = label_idx
        return entity_dict

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        entities = self.umls_json_info[index]['entities']
        captions = self.umls_json_info[index]['caption']
        if len(entities) != 0:
            try:
                radgraph_entities = self.radgraph_json_info[img_path]['entities']
                radgraph_entity_dict = self.get_entity_list(radgraph_entities)
                entity_details = ''
                for entity in entities:
                    sub_entities = entity['entity']
                    sub_entity_details = ''
                    for sub_entity in sub_entities:
                        sub_entity_info = sub_entity['Entity']
                        sub_entity_list = sub_entity_info.split(' ')
                        if sub_entity_info in radgraph_entity_dict.keys():
                            sub_entity_details += ' [ENT] ' + sub_entity_info +  radgraph_entity_dict[sub_entity_info]
                        elif len(sub_entity_list) > 1:
                            for sub_entity_single in sub_entity_list:
                                if sub_entity_single in radgraph_entity_dict.keys():
                                    sub_entity_details += ' [ENT] ' + sub_entity_single + ' ' + radgraph_entity_dict[sub_entity_single]
                                else:
                                    sub_entity_details += ' [ENT] ' + sub_entity_single
                        else:
                            sub_entity_details += ' [ENT] ' + sub_entity_info 
                    entity_details = entity_details + sub_entity_details + ' [SEP] '
            except:
                entity_details = ''
                for entity in entities:
                    sub_entities = entity['entity']#搞错了 还不是list
                    sub_entity_details = ''
                    for sub_entity in sub_entities:
                        sub_entity_details += ' [ENT] ' + sub_entity['Entity'] 
                    entity_details = entity_details + sub_entity_details + ' [SEP] '
        else:
            entity_details = ''
            for sub_caption in captions:
                entity_details = entity_details + sub_caption + ' [SEP] '
    
        img = open_jpg(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label,
            "entity": entity_details
            }
    

class Chestxray14_Dataset(Dataset):
    def __init__(self, csv_path,img_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,3:])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize(512, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

class CheXpert_Dataset(Dataset):
    def __init__(self, csv_path,img_res):
        data_info = pd.read_csv(csv_path)
        self.img_path_list = np.asarray(data_info.iloc[:,0])
        self.class_list = np.asarray(data_info.iloc[:,[13,7,11,10,15]])

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.Resize([img_res,img_res], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])  
    
    def __getitem__(self, index):
        img_path = os.path.join('/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/CheXpert/small/',self.img_path_list[index])
        class_label = self.class_list[index] 
        img = Image.open(img_path).convert('RGB')   
        image = self.transform(img)
        return {
            "image": image,
            "label": class_label
            }
    
    def __len__(self):
        return len(self.img_path_list)

