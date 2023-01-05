## Dependencies

To clone all files:

```
git clone 
```

To install Python dependencies:

```
pip install -r requirements.txt
```

## Data

#### **Training Dataset**   

1. Navigate to [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) to download the training dataset. Note: in order to gain access to the data, you must be a credentialed user as defined on [PhysioNet](https://physionet.org/settings/credentialing/).
    
1. preprocess MIMIC-CXR  
    preprocess dataset into csv file as ./A1_DATA/MIMIC-CXR/data_file/caption.csv
  
  `cd ./A1_DATA/MIMIC-CXR/data_preprocess`
  
  `sh run_preprocess.sh`

#### **Evaluation Dataset**   

**1. PadChest Dataset**

The PadChest dataset contains chest X-rays that were interpreted by 18 radiologists at the Hospital Universitario de San Juan, Alicante, Spain, from January 2009 to December 2017. The dataset contains 109,931 image studies and 168,861 images. PadChest also contains 206,222 study reports.

The [PadChest](https://arxiv.org/abs/1901.07441) is publicly available at https://bimcv.cipf.es/bimcv-projects/padchest. Those who would like to use PadChest for experimentation should request access to PadChest at the [link](https://bimcv.cipf.es/bimcv-projects/padchest).

**2. NIH ChestXrat14 Dataset**

NIH ChestXray14 dataset has 112,120 X-ray images with disease labels from 30,805
unique patients. Authors use natural language processing and the associated radiological reports to text-mine disease labels. There are 14 disease labels: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening and Hernia. 

The dataset and official split can be obtain at https://nihcc.app.box.com/v/ChestXray-NIHCC

**3. CheXpert Dataset**

The CheXpert dataset consists of chest radiographic examinations from Stanford Hospital, performed between October 2002 and July 2017 in both inpatient and outpatient centers. Population-level characteristics are unavailable for the CheXpert test dataset, as they are used for official evaluation on the CheXpert leaderboard.

The main data (CheXpert data) supporting the results of this study are available at https://aimi.stanford.edu/chexpert-chest-x-rays.

The CheXpert **test** dataset has recently been made public, and can be found by following the steps in the [cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels) repository.

#### 

## Model Checkpoints

**1. Med-KEBERT**

Med-KEBERT on [![Hugging Face's logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)**Hugging Face**](https://huggingface.co/) 

https://huggingface.co/xmcmic/Med-KEBERT

`from transformers import AutoModel,BertConfig,AutoTokenizer`

`bert_model_name = "xmcmic/Med-KEBERT"`

`tokenizer = AutoTokenizer.from_pretrained(bert_model_name,do_lower_case=True)`

`config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)`

`model = AutoModel.from_pretrained(bert_model_name, config=config)`



**2. Pre-trained Model**



## Pre-training

**1. Med-KEBERT**

The UMLS knowledge base file used during pretraining is in  `./A2_KEBER/data`

run the following command to perform Med-KEBERT pretraining

`cd ./A2_KEBERT`

`python main.py --pretrained microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext --batch-size 256 --max_length 128 --logs logs --name medkgbert --output_dir --aws_output_dir ` 

###### Arguments

- `--output_dir` directory to save logs
- `--aws_output_dir` directory to save checkpoints

**2. Pre-trained Model**

`cd ./A3_CLIP`

`python main.py --use_entity_features --image_encoder_name resnet  --bert_model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext   --bert_pretrained --output_dir --aws_output_dir`

###### Arguments

- `--bert_pretrained` pertained Med-KEBERT model path

- `--output_dir` directory to save logs

- `--aws_output_dir` directory to save checkpoints

  

## Zero-shot Inference

`python test_chestxray14.py --image_encoder_name  --bert_model_name   --bert_pretrained  --output_dir  --checkpoint` 

`python test_chexpert.py --image_encoder_name  --bert_model_name   --bert_pretrained  --output_dir  --checkpoint` 

`python test_padchest.py --image_encoder_name  --bert_model_name   --bert_pretrained  --output_dir  --checkpoint` 

###### Arguments

- `--bert_pretrained` pertained Med-KEBERT model path

- `--output_dir` directory to save result csv files

- `--checkpoint` directory of  pre-trained model's checkpoints

  
