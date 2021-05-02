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
import glob
import gc 
import bisect 

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


########################### Lazy Dataset ########################################################

def pickle2torch(pickle_path):
    
    pkl = pickle_loader(pickle_path + 'cnn_train_4_dataloader_batch_50.pkl')
    
    print('length of pkl: ', len(pkl))
    print(pkl.iloc[0])
    for i in range(len(pkl)):
        sample = self.dataset.iloc[idx]

#     return (
#         torch.tensor(sample.patches).view(sample.patches.shape[0], sample.patches.shape[1]), 
#         torch.tensor(sample.input_ids),
#         torch.tensor(sample.is_paired),
#         torch.tensor(sample.attention_mask)
#         )
    
class LazyDataset(Dataset):
    def __init__(self, filename):
        self._filename = filename
        self.dataset = pickle_loader(filename)

    def __getitem__(self, idx):
        sample = self.dataset.iloc[idx]
        
        return (
            torch.tensor(sample.id),
            torch.tensor(sample.doc_embed),
            torch.tensor(sample.story_ids),
            torch.tensor(sample.att_mask_story),
            torch.tensor(sample.high_ids),
            torch.tensor(sample.neg_high_ids)
            )

    def __len__(self):
        return len(self.dataset)

def new_collate(batch):
    """
    :param batch: id, doc_embed, story_ids, att_masks, high_ids, neg_high_ids
    :return:
    """
    ids = [item[0] for item in batch]
    doc_embeds = [item[1] for item in batch]
    ids = torch.stack(ids)
    doc_embeds = torch.stack(doc_embeds)
#     print('ids shape: ' , ids.shape)
#     print('doc embeds shape: ', doc_embeds.shape)

    story_ids = pad_sequence([item[2] for item in batch], batch_first=True)
    story_att_mask = pad_sequence([item[3] for item in batch], batch_first=True)
    high_ids = pad_sequence([item[4] for item in batch], batch_first=True)
    neg_high_ids = pad_sequence([item[5] for item in batch], batch_first=True)

#     print('story ids shape: ', story_ids.shape)
#     print('story att mask shape: ', story_att_mask.shape)
#     print('high_ids shape: ', high_ids.shape)
#     print('neg high ids: ', neg_high_ids.shape)

    return ids, doc_embeds, story_ids, story_att_mask, high_ids, neg_high_ids
    
   

############################ Dataloader for Multiple files #######################################

def pickle_loader(file_path):
    f = open(file_path, "rb")
    dataset = pickle.load(f)
    return dataset

def ext_batch_size_fn(new, count):
    print('new: ', new)
    print('count: ', count)
    if (len(new) == 4):
        pass
    src, labels = new[0], new[4]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents = 0
        max_n_tokens = 0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    
    return src_elements

    
def load_dataset(args, corpus_type, shuffle):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pkl_file, corpus_type):
        dataset = pickle_loader(pkl_file)
        #dataset = torch.load(pt_file)
#         print('Loading %s dataset from %s, number of examples: %d' %
#                     (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.bert_data_path + '/cnn_train_4_dataloader_batch_' + '[0-9]*.pkl'))

    if pts:
        if (shuffle):
            random.shuffle(pts)
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.bert_data_path + '/cnn_train_4_dataloader_batch_' + '.pkl'
        yield _lazy_dataset_loader(pt, corpus_type)


class DataloaderMultiple(object):
    def __init__(self, args, datasets,  batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets # 
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        print('cur iter: ', self.cur_iter)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)
    
    
class DataIterator(object):
    def __init__(self, args, dataset, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])
        self._iterations_this_epoch = 0
        self.batch_size_fn = ext_batch_size_fn

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex, is_test):
        
        print('ex: ', ex)

#         src = ex['src']
#         tgt = ex['tgt'][:self.args.max_tgt_len][:-1]+[2]
#         src_sent_labels = ex['src_sent_labels']
#         segs = ex['segs']
#         if(not self.args.use_interval):
#             segs=[0]*len(segs)
#         clss = ex['clss']
#         src_txt = ex['src_txt']
#         tgt_txt = ex['tgt_txt']

#         end_id = [src[-1]]
#         src = src[:-1][:self.args.max_pos - 1] + end_id
#         segs = segs[:self.args.max_pos]
#         max_sent_id = bisect.bisect_left(clss, self.args.max_pos)
#         src_sent_labels = src_sent_labels[:max_sent_id]
#         clss = clss[:max_sent_id]
#         # src_txt = src_txt[:max_sent_id]

        if(is_test):
            return 0
            #return src, tgt, segs, clss, src_sent_labels, src_txt, tgt_txt
        else:
            return 1
            #return src, tgt, segs, clss, src_sent_labels

    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex, self.is_test)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def batch(self, data, batch_size):
        """Yield elements from data in chunks of batch_size."""
        minibatch, size_so_far = [], 0
        for ex in data:
            minibatch.append(ex)
            size_so_far = self.batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], self.batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 300):

            p_batch = sorted(buffer, key=lambda x: len(x[2]))
            p_batch = self.batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                if(len(b)==0):
                    continue
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device, self.is_test)

                yield batch
            return
        
class Batch(object):
    def _pad(self, data, pad_id, width=-1):
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_tgt = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]
            pre_src_sent_labels = [x[4] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))
            tgt = torch.tensor(self._pad(pre_tgt, 0))

            segs = torch.tensor(self._pad(pre_segs, 0))
            mask_src = 1 - (src == 0)
            mask_tgt = 1 - (tgt == 0)


            clss = torch.tensor(self._pad(pre_clss, -1))
            src_sent_labels = torch.tensor(self._pad(pre_src_sent_labels, 0))
            mask_cls = 1 - (clss == -1)
            clss[clss == -1] = 0
            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src_sent_labels', src_sent_labels.to(device))


            setattr(self, 'src', src.to(device))
            setattr(self, 'tgt', tgt.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask_src', mask_src.to(device))
            setattr(self, 'mask_tgt', mask_tgt.to(device))


            if (is_test):
                src_str = [x[-2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[-1] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size

    
if __name__ == '__main__':
    root = '../data/batches_processed/'
    files = sorted(glob.glob(root + 'cnn_train_4_dataloader_batch_' + '[0-9]*.pkl'))
    print('files: ', files)
    datasets = list(map(lambda x : LazyDataset(x), files))
    dataset = torch.utils.data.ConcatDataset(datasets)
    print('dataset: ', dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        collate_fn=new_collate,
    )
    it = iter(dataloader)
    first = next(it)