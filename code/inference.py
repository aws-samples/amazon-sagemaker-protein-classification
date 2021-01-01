from __future__ import print_function

import json
import logging
import os
import sys

import torch
import torch.utils.data
from transformers import BertTokenizer

# Network definition
from model_def import ProteinClassifier
 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 512  # this is the max length of the sequence
PRE_TRAINED_MODEL_NAME = 'Rostlab/prot_bert'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, do_lower_case=False)

def model_fn(model_dir):
    logger.info('model_fn')
    print('Loading the trained model...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinClassifier(10) # pass number of classes, in our case its 10
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=device))
    return model.to(device)

def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        sequence = json.loads(request_body)
        print("Input protein sequence: ", sequence)
        encoded_sequence = tokenizer.encode_plus(
        sequence, 
        max_length = MAX_LEN, 
        add_special_tokens = True, 
        return_token_type_ids = False, 
        padding = 'max_length', 
        return_attention_mask = True, 
        return_tensors='pt'
        )
        input_ids = encoded_sequence['input_ids']
        attention_mask = encoded_sequence['attention_mask']

        return input_ids, attention_mask

    raise ValueError("Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    input_id, input_mask = input_data
    logger.info(input_id, input_mask)
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    with torch.no_grad():
        output = model(input_id, input_mask)
        _, prediction = torch.max(output, dim=1)
        return prediction