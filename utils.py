
import torch, torchvision
import json
import re
import time
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import BertTokenizer
import pandas as pd
import pickle
import argparse
from transformers import BertTokenizer, BertModel
import numpy as np
import copy
import pdb
import random

from others.utils import clean
from prepro.utils import _get_word_ngrams

random.seed(2021)

def open_json(path):
    f = open(path)
    data = json.load(f)
    f.close()
    return data

def save_json(file_path, data):
    out_file = open(file_path, "w")
    json.dump(data, out_file)
    out_file.close()

def check_tokenize_file(path):
    d = open_json(path)
    print(d[0])
    
def check_pkl_file(file):
    f = open(file, "rb")
    dataset = pickle.load(f)
    print('len dataset: ', len(dataset))
    print('example')
    print(dataset.iloc[0])

def construct_bert_input(doc_embed, input_ids, model, device=None):

    # input_ids shape: [batch size, sentence length]
    # doc embed: [batch size, 764]

    #word_embeddings = model.bert.embeddings(
    #    input_ids.to(device),
    #    token_type_ids=torch.zeros(input_ids.shape, dtype=torch.long).to(device),
    #    position_ids=torch.arange(0, input_ids.shape[1], dtype=torch.long).to(device) * torch.ones(input_ids.shape, dtype=torch.long).to(device))
    doc_embed = doc_embed.to(device)
    inputs_embeddings = model.bert.embeddings(input_ids.to(device)) # [batch, 511, 768]

    # For doc embedding
    doc_position_ids = torch.ones(1, dtype=torch.long).to(device) * torch.ones(doc_embed.shape[0], dtype=torch.long).to(device) # [1, 1]
    doc_token_type_ids = torch.ones(doc_embed.shape[0], dtype=torch.long).to(device) # [1, 1]

    # Get embeddings
    doc_position_embeds = model.bert.embeddings.position_embeddings(doc_position_ids.to(device)) # [batch, 768]
    doc_token_type_embeds = model.bert.embeddings.token_type_embeddings(doc_token_type_ids.to(device)) # [batch, 768]

    # Add them up
    doc_embeddings = doc_embed + doc_position_embeds + doc_token_type_embeds   # [batch, 768]
    doc_embeddings = doc_embeddings.unsqueeze(1)                               # [batch, 1, 768]

    # Cat and return
    return torch.cat((doc_embeddings, inputs_embeddings), dim=1) # [batch, 512, 768]

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocessed_dataset', type=str,
                        default='../data/cnn/preprocessed_ourmethod/cnn_dataset_preprocessed.pkl',
                        help='Path to dataset')
    parser.add_argument('--save_path', type=str, default='../data/cnn/preprocessed_ourmethod/',
                        help='Path to save preprocessed dataset')
    parser.add_argument("--shard_size", default=2000, type=int)
    parser.add_argument('--min_src_nsents', default=3, type=int)
    parser.add_argument('--max_src_nsents', default=100, type=int)
    parser.add_argument('--min_src_ntokens_per_sent', default=5, type=int)
    parser.add_argument('--max_src_ntokens_per_sent', default=200, type=int)
    parser.add_argument('--min_tgt_ntokens', default=5, type=int)
    parser.add_argument('--max_tgt_ntokens', default=500, type=int)



    args = parser.parse_args()
    print(f"RUN: {vars(args)}")
    return args

# if __name__ == "__main__":
    # args = create_parser()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # print('Loading dataset....')
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=3,
    #     shuffle=False,
    #     collate_fn=collate,
    # )
    # print('Dataloader length: ')
    # print(len(dataloader))
    # print('iterating!')
    # i = 0
    #
    # it = iter(dataloader)
    # first = next(it)
    # print('first')

#   FROM PICKLE FILE
# class ExtractiveSummarization(Dataset):
#     def __init__(self, path_to_dataset):
#         super(ExtractiveSummarization).__init__()
#         self.path_to_dataset = path_to_dataset
#
#         f = open(self.path_to_dataset, "rb")
#         self.dataset = pickle.load(f)
#
#     def __len__(self):
#         return len(self.dataset)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         sample = self.dataset.iloc[idx]
#
#         return (
#             torch.tensor(sample.patches).view(sample.patches.shape[0], sample.patches.shape[1]),
#             torch.tensor(sample.input_ids),
#             torch.tensor(sample.is_paired),
#             torch.tensor(sample.attention_mask)
#         )