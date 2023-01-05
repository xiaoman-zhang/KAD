import os
import cv2
import logging
import sys
import json
import random
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from dataclasses import dataclass
from multiprocessing import Value

# import braceexpand

import torch
import torchvision.datasets as datasets
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class UMLS_Dataset(Dataset):
    def __init__(self,mrdef_csv_file, umls_kg_file, umls_cui_file):
        self.mrdef_info = pd.read_csv(mrdef_csv_file)
        self.mrdef_cui_list = self.mrdef_info.iloc[:,0]
        self.mrdef_name_list = self.mrdef_info.iloc[:,1]
        self.mrdef_def_list = self.mrdef_info.iloc[:,2]

        self.umls_kg_info = pd.read_csv(umls_kg_file)
        self.umls_kg_source_list = self.umls_kg_info.iloc[:,0]
        self.umls_kg_target_list = self.umls_kg_info.iloc[:,1]
        self.umls_kg_edge_list = self.umls_kg_info.iloc[:,2]

        self.umls_cui_info = pd.read_csv(umls_cui_file)
        self.umls_cui_source_list = self.umls_cui_info.iloc[:,0]
        self.umls_cui_target_list = self.umls_cui_info.iloc[:,1]

        self.umls_data_len = len(self.umls_kg_info)
        self.mrdef_data_len = len(self.mrdef_info)
        print('UMLS data length: ',self.umls_data_len)
        print('MRDEF data length: ',self.mrdef_data_len)
        self.select_umls_ratio = self.umls_data_len/(self.umls_data_len+self.mrdef_data_len)
    
    def __len__(self):
        return int(self.umls_data_len+self.mrdef_data_len)
    
    def __getitem__(self, idx):
        if random.random() < self.select_umls_ratio:
            select_idx = random.randint(0,self.umls_data_len-1)
            text_h = self.umls_kg_source_list[select_idx]
            cui_h = self.umls_cui_source_list[select_idx]
            text_t = self.umls_kg_target_list[select_idx]
            cui_t = self.umls_cui_target_list[select_idx]
            text_r = self.umls_kg_edge_list[select_idx]
            if random.random()<0.5:
                input_text = text_h + ' [SEP] ' + text_r
                pos_text =  text_t
                cui = cui_t
            else:
                input_text = text_r + ' [SEP] ' + text_t
                pos_text =  text_h
                cui = cui_h
        else:
            select_idx = random.randint(0,self.mrdef_data_len-1)
            input_text = self.mrdef_name_list[select_idx]
            pos_text = self.mrdef_def_list[select_idx]
            cui = self.mrdef_cui_list[select_idx]
        sample = {}
        sample['input_text'] = input_text
        sample['pos_text'] = pos_text
        try: 
            if cui[0] == 'C':
                sample['cui'] = cui
            else:
                sample['cui'] = str(0)
        except:
            sample['cui'] = str(0)
        return sample
        
class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)
