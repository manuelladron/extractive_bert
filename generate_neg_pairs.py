import pickle
import json
import gzip
import pickletools
import pandas as pd
from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, output_hidden_states=True)

import torch
import numpy as np
import argparse
from tqdm import tqdm

# dataset_path = '../data/cnn/preprocessed/cnn_dataset_preprocessed.pkl'
# stories = pickle.load(open(dataset_path, 'rb'))
# print('Loaded Stories %d' % len(stories))
#
# print('\n Story 1')
# story1 = stories[0]
#
# for sent in story1['story']:
#     print('\nsent: ')
#     print(sent)
#
# print('\n----HIGHLIGHTS-----')
# print(story1['highlights'])

def save_json(file_path, data):
    out_file = open(file_path, "w")
    json.dump(data, out_file)
    out_file.close()

def generate_encoding_doc(story, tokenizer, model):
    """
    :param story: list with sentences corresponding to 1 story
    :return: contextual representation
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


def generate_pos_neg_pairs(dataset, save_data):
    stories = pickle.load(open(dataset, 'rb'))
    num_stories = len(stories)
    print('Loaded Stories %d' % num_stories)

    # Setting bert model
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, output_hidden_states=True)

    # All highlights
    all_high = []
    for s in stories:
        high = s['highlights']  # list with sentences
        all_high += high

    i = 0
    df = pd.DataFrame(columns=['id', 'story', 'highlights', 'doc_embed', 'neg_high'])
    for story in tqdm(stories):
        # Save
        if i%100 == 0 and i != 0:
            df.to_pickle(save_data + f'_{i}.pkl')
            df = pd.DataFrame(columns=['id', 'story', 'highlights', 'doc_embed', 'neg_high'])

        df = df.append({
            "id": i,
            "story": story['story'],
            "highlights": story['highlights'],
            "doc_embed": generate_encoding_doc(story['story'], tokenizer, model),
            "neg_high": get_random_sents(all_high, num_stories, story['highlights'])
        },
        ignore_index=True)
        i += 1

        # if i%10:
        #     print(f'At batch # {i}')
    # Save dataset here
    #pickle.dump(pos_neg_pairs, open(save_data, 'wb'))
    # save_json(save_data, pos_neg_pairs)

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--preprocessed_dataset', type=str,
                        default='../data/cnn/preprocessed/cnn_dataset_preprocessed.pkl',
                        help='Path to dataset')
    parser.add_argument('--save_dataset', type=str, default='../data/cnn/preprocessed/cnn_pos_neg_pairs',
                        help='Path to save '
                                                                                             'preprocessed dataset')
    args = parser.parse_args()
    print(f"RUN: {vars(args)}")
    return args


if __name__ == "__main__":
    args = create_parser()

    dataset = args.preprocessed_dataset

    generate_pos_neg_pairs(dataset, args.save_dataset)

