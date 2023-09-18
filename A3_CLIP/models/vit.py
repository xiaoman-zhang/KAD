import re
import logging
import math
import json
import pathlib
import numpy as np

from copy import deepcopy
from pathlib import Path
from einops import rearrange
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional


import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.checkpoint import checkpoint
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from pytorch_pretrained_vit import ViT
from transformers import ViTImageProcessor, ViTModel

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=32, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class ModelViT224(nn.Module):
    def __init__(self):
        super(ModelViT224, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    
    def forward(self, x):
        outputs = self.vit(x)
        last_hidden_states = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        out_emb = last_hidden_states[:,1:]
        out_pool = last_hidden_states[:,0]
        return out_emb,out_pool

class ModelViT(nn.Module):
    def __init__(self):
        super(ModelViT, self).__init__()
        self.vit = ViT('B_16_imagenet1k', pretrained=True)
        # self.patch_embedding = PatchEmbed()
        # self.class_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.patch_embedding = self.vit.patch_embedding
        self.class_token = self.vit.class_token
        #seq_length + 1
        # self.positional_embedding = get_sinusoid_encoding_table(50,768)
        self.positional_embedding = self.vit.positional_embedding
        self.transformer = self.vit.transformer
    
    def forward(self, x):
        b, c, fh, fw = x.shape
        x = self.patch_embedding(x)  # b,d,gh,gw
        x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
        x = torch.cat((self.class_token.expand(b, -1, -1), x), dim=1) 
        x = self.positional_embedding(x)
        # x = x + self.positional_embedding.expand(b, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.transformer(x)
        out_emb = x[:,1:]
        out_pool = x[:,0]
        return out_emb,out_pool

if __name__ == "__main__":
    from transformers import ViTImageProcessor, ViTModel #pip install transformers==4.25.1
    from PIL import Image
    import requests

    image = torch.randn(1, 3, 224, 224) 
    model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    outputs = model(image)
    last_hidden_states = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    print(last_hidden_states.shape,pooler_output.shape)
    # torch.Size([1, 197, 768]) torch.Size([1, 768])
    for param in model.parameters():
        if param.requires_grad:
            pass
        else:
            print(param.requires_grad)
