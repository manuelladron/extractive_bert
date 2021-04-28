from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch, torchvision
import transformers
from transformers import BertTokenizer
import pandas as pd
import pickle
import random
import argparse
from transformers import BertTokenizer, BertModel
import numpy as np
import copy
import pdb
import json
import time
from utils import open_json

def generate_encoding_doc(story, tokenizer, model):
    """
    :param story: list with sentences corresponding to 1 story
    :return: contextual representation BERT
    """
    batch_sents = tokenizer(story, padding=True, truncation=True, return_tensors='pt')
    output = model(batch_sents['input_ids'])
    all_hidden = output[2]  # tuple
    all_hidden_pt = torch.stack(all_hidden)  # [13, batch, seq_len, 768]
    # Use second to last layer as embedding
    sentence_embeds = all_hidden_pt[-2, :, :] # [batch, seq_len, 768]
    doc_embed = sentence_embeds.mean(dim=1).mean(dim=0) # [768]
    return doc_embed

def get_random_sents(all_highlights, num_h,  highlight):
    """
    Recursive function to get negative highlights
    :param all_highlights: all sentences from the dataset
    :param num_h: length of all_highlights
    :param highlight: current highlight of a particular story
    :return: list with negative highlights
    """
    num_pos_h = len(highlight) # Current story's highlights
    rand_idxs = np.random.choice(num_h, num_pos_h)
    access_map = map(all_highlights.__getitem__, rand_idxs)
    neg_high = list(access_map)
    check = any(h in neg_high for h in highlight)
    if check:
        get_random_sents(all_highlights, highlight)
    return neg_high


class ExtractiveDataset(Dataset):
    def __init__(self, dataset_path, device):
        super(ExtractiveDataset).__init__()
        self.dataset_path = dataset_path

        self.dataset = open_json(self.dataset_path)
        self.num_stories = len(self.dataset)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, output_hidden_states=True)

        self.device = device

        self.all_high = []
        for s in self.dataset:
            high = s['highlights']  # list with sentences
            self.all_high += high

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        story = sample['story']
        highlights = sample['highlights']

#         print('highlights: ')
#         print(highlights)

#         print('embedding doc...')
        doc_embed = generate_encoding_doc(story, self.tokenizer, self.model)
#         print('getting random sents...')
        neg_high = get_random_sents(self.all_high, self.num_stories, highlights)

#         print('tokenizing...')
        token_story = []
        token_types_story = []
        att_mask_story = []

        token_high = []
        token_types_high = []

        token_neg_high = []
        token_types_neg_high = []

        for i, s in enumerate(story):
            tokens = self.tokenizer.encode(s)
            # tokens = self.tokenizer(s, max_length=511, truncation=True, padding = 'max_length', return_tensors = 'pt')
            token_story += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_story += types

        for i, h in enumerate(highlights):
            tokens = self.tokenizer.encode(h)
            token_high += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_high += types

        for i, nh in enumerate(neg_high):
            tokens = self.tokenizer.encode(nh)
            token_neg_high += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_neg_high += types

        story_ids = torch.LongTensor(token_story[:511]).to(self.device)
        high_ids = torch.LongTensor(token_high[:512]).to(self.device)
        neg_high_ids = torch.LongTensor(token_neg_high[:512]).to(self.device)
        att_mask_story = torch.ones_like(story_ids).to(self.device)

#         print('In get item: ')
#         print('doc embed: ', doc_embed.shape)
#         print('story ids: ', story_ids.shape)
#         print('neg_high_ids: ', neg_high_ids.shape)
#         print('att mask story: ', att_mask_story.shape)

        return doc_embed, story_ids, att_mask_story, high_ids, neg_high_ids


class ProcessText2(Dataset):
    def __init__(self, dataset_path, device):
        super(ProcessText).__init__()
        self.dataset_path = dataset_path

        f = open(self.dataset_path, "rb")
        self.dataset = pickle.load(f) # list with dictionaries
        self.num_stories = len(self.dataset)

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, output_hidden_states=True)

        self.device = device

        self.all_high = []
        for s in self.dataset:
            high = s['highlights']  # list with sentences
            self.all_high += high

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        story = sample['story']
        highlights = sample['highlights']

        print('highlights: ')
        print(highlights)

        print('embedding doc...')
        doc_embed = generate_encoding_doc(story, self.tokenizer, self.model)
        print('getting random sents...')
        neg_high = get_random_sents(self.all_high, self.num_stories, highlights)

        print('tokenizing...')
        token_story = []
        token_types_story = []

        token_high = []
        token_types_high = []

        token_neg_high = []
        token_types_neg_high = []

        for i, s in enumerate(story):
            tokens = self.tokenizer.encode(s)
            token_story += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_story += types

        for i, h in enumerate(highlights):
            tokens = self.tokenizer.encode(h)
            token_high += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_high += types

        for i, nh in enumerate(neg_high):
            tokens = self.tokenizer.encode(nh)
            token_neg_high += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_neg_high += types

        story_ids = torch.LongTensor(token_story).to(device)
        high_ids = torch.LongTensor(token_high).to(device)
        neg_high_ids = torch.LongTensor(token_neg_high).to(device)

        print('In get item: ')
        print('doc embed: ', doc_embed.shape)
        print('story ids: ', story_ids.shape)
        print('neg_high_ids: ', neg_high_ids.shape)

        return doc_embed, story_ids, high_ids, neg_high_ids

def collate(batch):
    """
    :param batch: doc_embed, story_ids, high_ids, neg_high_ids
    :return:
    """
    doc_embeds = [item[0] for item in batch]
    doc_embeds = torch.stack(doc_embeds)
#     print('doc embeds shape: ', doc_embeds.shape)

    story_ids = pad_sequence([item[1] for item in batch], batch_first=True)
    story_att_mask = pad_sequence([item[2] for item in batch], batch_first=True)
    high_ids = pad_sequence([item[3] for item in batch], batch_first=True)
    neg_high_ids = pad_sequence([item[4] for item in batch], batch_first=True)

#     print('story ids shape: ', story_ids.shape)
#     print('story att mask shape: ', story_att_mask.shape)
#     print('high_ids shape: ', high_ids.shape)
#     print('neg high ids: ', neg_high_ids.shape)

    return doc_embeds, story_ids, story_att_mask, high_ids, neg_high_ids

