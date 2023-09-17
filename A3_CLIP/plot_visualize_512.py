import argparse
import os
import cv2
import ruamel_yaml as yaml
import numpy as np
import random
import json
import math
from skimage import io
from tqdm import tqdm
from pathlib import Path
from functools import partial
from einops import rearrange

from torchvision import transforms
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from transformers import AutoModel,BertConfig,AutoTokenizer
from models.clip_tqn import CLP_clinical,ModelRes512,TQN_Model

def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features

def get_gt(syms_list,text_list):
    gt_class = np.zeros((10))
    gt = []
    for syms in syms_list:
        syms_class = text_list.index(syms)
        gt.append(syms_class)
        gt_class[syms_class] = 1
    return gt,gt_class

def main(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    seed = args.seed  
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # model
    image_encoder = ModelRes512(res_base_model='resnet50').to(device) 
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name,do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name).to(device) 

    model = TQN_Model().to(device) 
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    image_state_dict = checkpoint['image_encoder']     
    image_encoder.load_state_dict(image_state_dict)    
    text_state_dict =  checkpoint['text_encoder']     
    text_encoder.load_state_dict(text_state_dict)     
    state_dict = checkpoint['model']      
    model.load_state_dict(state_dict)  
    
    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([                        
                transforms.Resize([512,512], interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                normalize,
            ])
    
    text_list = ['Atelectasis','Calcification','Consolidation','Effusion','Emphysema','Fibrosis','Fracture','Mass','Nodule', 'Pneumothorax']
    text_features = get_text_features(text_encoder,text_list,tokenizer,device,max_length=args.max_length)
    json_info = json.load(open(args.test_path,'r'))

    gt = [] #label of each boxes
    gt_boxes = []
    gt_class = [] #0-1label
    
    for data_index in tqdm(range(len(json_info))):
        json_index = json_info[data_index]
        file_name = json_index['file_name']
        syms_list = json_index['syms']
        boxes_index = json_index['boxes']
        gt_index,gt_class_index = get_gt(syms_list,text_list)
        gt.append(gt_index)
        gt_boxes.append(boxes_index)
        gt_class.append(gt_class_index)
        # print(gt_index,gt_class_index)
        data_path = os.path.join('./ChestX-Det10-Dataset/test_data',file_name)
        img = Image.open(data_path).convert('RGB')  
        image = transform(img)
        image = image.unsqueeze(0).to(device) 
        
        with torch.no_grad():
            image_features,_ = image_encoder(image)
            pred_class_index,atten_map_index = model(image_features,text_features,return_atten=True)
            pred_class_index = torch.softmax(pred_class_index, dim=-1)
            atten_map_index = np.array(atten_map_index.cpu().numpy())
            
            if len(gt_index) == 0:
                pass
            else:
                for class_index in range(len(gt_class_index)):
                    if gt_class_index[class_index] == 1:
                        img_size = img.size
                        save_attn_path = os.path.join(os.path.join(args.output_dir,'visualize'),str(data_index)+'_'+str(class_index)+'_atten.png')
                        # print(atten_map_index.shape,class_index)
                        atten_map = rearrange(atten_map_index[0][class_index],' (w h) -> w h', w=16,h=16)
                        atten_map = 255*atten_map/np.max(atten_map)
                        atten_map = cv2.resize(atten_map, img_size, interpolation = cv2.INTER_LINEAR).astype(np.uint8)
                        atten_map = cv2.applyColorMap(atten_map,cv2.COLORMAP_JET)
                        img_array = np.asarray(img).astype(np.uint8)
                        atten_map_img = cv2.addWeighted(img_array, 0.5, atten_map, 0.5, 0)
                        atten_map_file = save_attn_path
                        io.imsave(atten_map_file,cv2.cvtColor(atten_map_img,cv2.COLOR_BGR2RGB))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/Res_train.yaml')
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--test_path', default='./ChestX-Det10-Dataset/test.json')
    parser.add_argument('--freeze_bert', default=False, type=bool)
    parser.add_argument('--ignore_index', default=False, type=bool)
    parser.add_argument("--use_entity_features", action="store_true")
    parser.add_argument('--image_encoder_name', default='resnet')
    parser.add_argument('--bert_pretrained', default='')
    parser.add_argument('--bert_model_name', default='')
    parser.add_argument('--save_result_path', default='./results/res_512/visualize')
    parser.add_argument('--output_dir', default='./results/res_512')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.save_result_path).mkdir(parents=True, exist_ok=True)
    main(args)