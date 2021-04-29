# Code from https://machinelearningmastery.com/prepare-news-articles-text-summarization/

### multiprocessing dependencies 
import argparse
import os
import string
import subprocess
from os import listdir
from pickle import dump
from multiprocessing import Pool, set_start_method
import multiprocessing
from os.path import join as pjoin
import glob
import psutil
from pathlib import Path

### preprocessing dependencies 
from tqdm import tqdm
import torch, torchvision
import numpy as np
import pandas as pd
import pickle
import gc
import hashlib
import json
import xml.etree.ElementTree as ET
from others.utils import clean
from prepro.utils import _get_word_ngrams

### other libraries 
import pdb
import time
import copy
import re
import random
from utils import open_json, save_json, check_pkl_file

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a document into news story and highlights
def split_story(doc):
    # find first highlight
    index = doc.find('@highlight')
    # split into story and highlights
    story, highlights = doc[:index], doc[index:].split('@highlight')
    # strip extra white space around each highlight
    highlights = [h.strip() for h in highlights if len(h) > 0]
    return story, highlights

# load all stories in a directory
def load_stories(directory):
    stories = list()
    for name in listdir(directory):
        filename = directory + '/' + name
        # load document
        doc = load_doc(filename)
        # split into story and highlights
        story, highlights = split_story(doc)
        # store
        stories.append({'story': story, 'highlights': highlights})
    return stories

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare a translation table to remove punctuation
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # strip source cnn office if it exists
        index = line.find('(CNN) -- ')
        if index > -1:
            line = line[index + len('(CNN)'):]
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctuation from each token
        line = [w.translate(table) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    cleaned = [c for c in cleaned if len(c) > 1]
    return cleaned

def load_json(p, lower):
    source = []
    tgt = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (lower):
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            tgt.append([])
            continue
        if (flag):
            tgt[-1].extend(tokens)
        else:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    tgt = [clean(' '.join(sent)).split() for sent in tgt]
    return source, tgt

def tokenize(args):
    stories_dir = os.path.abspath(args.raw_dataset)
    tokenized_stories_dir = os.path.abspath(args.save_dataset)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
               '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
               'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def format_to_lines(args):
    print('Doing first part.....')
    pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, 'TEST', 0)
    print('pt file: ', pt_file)
    pdb.set_trace()
    corpus_mapping = {}
    for corpus_type in ['valid', 'test', 'train']:
        temp = []
        for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
            temp.append(hashhex(line.strip()))
        corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
    train_files, valid_files, test_files = [], [], []
    for f in glob.glob(pjoin(args.raw_path, '*.json')):
        real_name = f.split('/')[-1].split('.')[0]
        if (real_name in corpus_mapping['valid']):
            valid_files.append(f)
        elif (real_name in corpus_mapping['test']):
            test_files.append(f)
        elif (real_name in corpus_mapping['train']):
            train_files.append(f)
        # else:
        #     train_files.append(f)

    print('Doing second part....')
    corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
    for corpus_type in ['train', 'valid', 'test']:
        a_lst = [(f, args) for f in corpora[corpus_type]]
        pool = Pool(args.n_cpus)
        dataset = []
        p_ct = 0
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            if (len(dataset) > args.shard_size):
                pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
                with open(pt_file, 'w') as save:
                    # save.write('\n'.join(dataset))
                    save.write(json.dumps(dataset))
                    p_ct += 1
                    dataset = []

        pool.close()
        pool.join()
        print('len of dataset > 0')
        if (len(dataset) > 0):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
            with open(pt_file, 'w') as save:
                # save.write('\n'.join(dataset))
                save.write(json.dumps(dataset))
                p_ct += 1
                dataset = []

def _format_to_lines(params):
    f, args = params
    print(f)
    source, tgt = load_json(f, args.lower)
    return {'src': source, 'tgt': tgt}

def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}

def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    """
    Adapted from original PreSumm code
    :param doc_sent_list:
    :param abstract_sent_list:
    :param summary_size:
    :return:
    """
    def _rouge_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    max_rouge = 0.0
    abstract = abstract_sent_list
    abstract = _rouge_clean(' '.join(abstract)).split()

    #print('In greedy selection, story: ', doc_sent_list)

    # Splits sentences into tokens to calculate ngramsqut
    sents = []
    returning_story = []
    for s in doc_sent_list:
        s_clean = _rouge_clean(s).split()
        if s_clean == '' or s_clean == [] or len(s_clean) < 2:
            continue
        else:
            sents.append(s_clean)
            returning_story.append(" ".join(s_clean))
    #sents = [_rouge_clean(s).split() for s in doc_sent_list]

    evaluated_1grams = [_get_word_ngrams(1, sent) for sent in sents]
    reference_1grams = _get_word_ngrams(1, abstract)
    evaluated_2grams = [_get_word_ngrams(2, sent) for sent in sents]
    reference_2grams = _get_word_ngrams(2, abstract)

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2

            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i

        if (cur_id == -1):
            return returning_story, selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return returning_story, sorted(selected)

def clean_and_save_dataset(dataset_path, save_path):
    """
    NOT USING!!!!!
    :param dataset_path:
    :param save_path:
    :return:
    """
    dataset_path = dataset_path
    f = open(dataset_path, "rb")
    dataset = pickle.load(f)  # list with dictionaries

    new_dataset = []
    empty_stories = 0
    for sample in dataset:
        story = sample['story']
        highlights = sample['highlights']
        if len(story) == 0 or len(highlights) == 0:
            empty_stories += 1
            continue
        for sent in story:
            if sent == [] or len(sent) < 1 or sent == '':
                print('removing sent: ', sent)
                story.remove(sent)

        for h in highlights:
            if h == [] or len(h) < 1 or h == '':
                print('removing highlight: ', h)
                highlights.remove(h)

        new_sample = {"story": story,
                      "highlights": highlights}
        new_dataset.append(new_sample)
    print('Saving in: ', save_path)
    pickle.dump(new_dataset, open(save_path + '/cnn_dataset_preprocessed_c.pkl', 'wb'))

def select_source_sents_and_partition_dataset(dataset_path, args):
    """
    Receives processed dataset generated by data_preprocess.py file and creates a train, test and val sets.
    :param dataset_path: path to preprocess dataset
    :param args:
    :return:
    """
    dataset_path = dataset_path
    f = open(dataset_path, "rb")
    dataset = pickle.load(f)  # list with dictionaries

    story_s = []
    empty_stories = 0
    i = 0
    for sample in dataset:
        story = sample['story']
        highlights = sample['highlights']
        if len(story) == 0 or len(highlights) == 0:
            empty_stories += 1
            continue

        # Clean story based on length (min and max)
        c_story = []
        for sent in story:
            if sent != [] or sent != '' or len(sent) > args.min_src_ntokens_per_sent:
                # print('removing sent: ', sent)
                c_story.append(sent)
            if len(sent) > args.max_src_ntokens_per_sent:
                sent = sent[:args.max_src_ntokens_per_sent]
                c_story.append(sent)

        story = c_story
        # print('story: ', story)
        story, sent_labels = greedy_selection(story[:args.max_src_nsents], highlights, 4)
        sorted_story = []
        _story = copy.deepcopy(story)
        for idx in sent_labels:
            sorted_story.append(story[idx])
            _story.remove(story[idx])
        sorted_story += [sent for sent in _story]
        sorted_story = sorted_story[:args.max_src_nsents]

        new_sample = {'id': i,
                      'story': sorted_story,
                      'highlights': highlights}
        story_s.append(new_sample)
        i+=1

    print('empty stories: ', empty_stories)
    random.shuffle(story_s)
    train_set = story_s[:90266]
    validation_set = story_s[90266:90266 + 1220]
    test_set = story_s[90266 + 1220:]

    print('---Dataset---')
    print('train set: ', len(train_set))
    print('val set: ', len(validation_set))
    print('test set: ', len(test_set))

    print('---- train sample')
    print(train_set[0])

    print('---Saving sets----')
    save_json(args.save_path + 'cnn_train_wid.json', train_set)
    save_json(args.save_path + 'cnn_test_wid.json', test_set)
    save_json(args.save_path + 'cnn_val_wid.json', validation_set)
    print('Saved in: ', args.save_path)

    return train_set, validation_set, test_set

####### FROM HERE WE WORK ON PREPROCESSING DATASET FROM JSON TO DATALOADER ############

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch, torchvision
import transformers
from transformers import BertTokenizer, BertModel


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

def get_random_sents(all_highlights, num_h, highlight):
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
        get_random_sents(all_highlights, num_h, highlight)
    return neg_high


########## functions for multiprocessing ####################
def partition_jsons(args):
    dataset = open_json(args.all_jsons)
    num_stories = len(dataset)
    print('num stories: ', num_stories)
    batch_size = 50
    num_batches = num_stories // batch_size
    print('num batches: ', num_batches)
    
    for b in range(num_batches):
        if b%50==0:
            path = args.save_path + f'batch_{b}'
            try: 
                os.mkdir(path) 
            except OSError as error: 
                print(error) 
          
        name = path + f'/cnn_train_{b}.json'
        #print('first idx: ', num_batches*b, 'second index: ', num_batches*(b+1))  
        save_json(name, dataset[batch_size*b : batch_size*(b+1)])
    print('Done!')
                  
class MultiprocessData():
    def __init__(self, args):
        self.args = args
        print('Loading tokenizer and model...')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, output_hidden_states=True)
        
        print('Generating all highlights...')
        self.allfiles = open_json(args.all_jsons)
        
        self.all_high = []
        for s in self.allfiles:
            high = s['highlights']  # list with sentences
            self.all_high += high
        print('Len all highlights: ', len(self.all_high))
    
        # Main directory 
        root_dir = os.path.dirname(args.save_path)
        print('MAIN DIR: ', root_dir)
        f = 0
        for folder in os.listdir(root_dir):
            st_ = time.time()
            folder_name = os.path.join(root_dir, folder)
            print('------folder: ', folder_name)
            df = self.pd_wrapper(directory=folder_name , pattern='*.json', processes=-1)
            df.to_pickle(args.save_dataset + f'cnn_{args.preprocess_mode}_4_dataloader_{folder}.pkl')
            print('\nSaved df file succesfully')
            f+=1
            del df
            gc.collect()
            print(f'Time to process folder {f}: {time.time() - st_} s')
            
    def get_files(self, directory, pattern):
        for path in Path(directory).rglob(pattern):
            yield path.absolute()
            
    def print_memory(self):
        print('cpu %: ', psutil.cpu_percent())
        vm = dict(psutil.virtual_memory()._asdict())
        print('virtual memory %: ', psutil.virtual_memory().percent)
        print('virtual memory aval. %: ', psutil.virtual_memory().available*100 / psutil.virtual_memory().total)
        
    def process_file(self, filename):
        print('filename: ', filename)
        data = open_json(filename)
        num_stories = len(data)
        print('num stories: ', num_stories)
        df = pd.DataFrame(columns=["id", "doc_embed", "story_ids", "att_mask_story", "high_ids", "neg_high_ids"])
       
        for i, sample in enumerate(data):
            #if i%3 == 0:
                #continue
            print(f'sample: {i}/{num_stories}')
            id_ = sample['id']
            story = sample['story']
            highlights = sample['highlights']

            doc_embed = generate_encoding_doc(story, self.tokenizer, self.model)
            neg_high = get_random_sents(self.all_high, num_stories, highlights)

            # All lists we need
            token_story = []
            token_types_story = []

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

            story_ids = torch.LongTensor(token_story[:511])
            high_ids = torch.LongTensor(token_high[:512])
            neg_high_ids = torch.LongTensor(token_neg_high[:512])
            att_mask_story = torch.ones_like(story_ids)

            df=df.append({'id': id_,
                            'doc_embed': doc_embed.detach().cpu().numpy(),
                            'story_ids': story_ids.detach().cpu().numpy(),
                            'att_mask_story': att_mask_story.detach().cpu().numpy(),
                            'high_ids': high_ids.detach().cpu().numpy(),
                            'neg_high_ids': neg_high_ids.detach().cpu().numpy()
                         },ignore_index=True)
            del story_ids
            del high_ids
            del neg_high_ids
            del att_mask_story
            del doc_embed
            del neg_high

            gc.collect()
            self.print_memory()
            
        print('\nReturning dataframe')
        return df
    
    def pd_wrapper(self, directory, pattern, processes=-1):
        # Decide how many proccesses will be created
        sum_size = 0
        if processes <=0:
            num_cpus = psutil.cpu_count(logical=False)
        else:
            num_cpus = processes
        print('num cpus: ', num_cpus)
        
        files = []
        # Get files based on pattern and their sum of size
        for file in self.get_files(directory=directory, pattern=pattern):
            sum_size = sum_size + os.path.getsize(file)
            files.append(file)
        
        print('files:%s,size:%s bytes, procs:%s'%(len(files),sum_size,num_cpus))
        # Create the pool
        with get_context('spawn').Pool(processes=num_cpus) as process_pool:
            #process_pool = Pool(processes=num_cpus)
            print('process pool: ', process_pool)
            start = time.time()
            # Start processes in the pool
            dfs = process_pool.map(self.process_file, files)
            print('process pool finished')
            # Concat dataframes to one dataframe
            process_pool.close()
            process_pool.join()
            print('concatenating pandas')
            data = pd.concat(dfs, ignore_index=True)
            end = time.time()
            print('Completed in: %s sec'%(end - start))

        return data

    
###############################################################        
def preprocess_dataloader(args):

    # Open dataset
    dataset = open_json(args.load_json)
    num_stories = len(dataset)

    # Set up tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', return_dict=True, output_hidden_states=True)

    # Get all highlights
    all_high = []
    for s in dataset:
        high = s['highlights']  # list with sentences
        all_high += high


    df = pd.DataFrame(columns=["id", "doc_embed", "story_ids", "att_mask_story", "high_ids", "neg_high_ids"])
    for sample in tqdm(dataset):
        id = sample['id']
        story = sample['story']
        highlights = sample['highlights']

        doc_embed = generate_encoding_doc(story, tokenizer, model)
        neg_high = get_random_sents(all_high, num_stories, highlights)

        # All lists we need
        token_story = []
        token_types_story = []

        token_high = []
        token_types_high = []

        token_neg_high = []
        token_types_neg_high = []

        for i, s in enumerate(story):
            tokens = tokenizer.encode(s)
            # tokens = self.tokenizer(s, max_length=511, truncation=True, padding = 'max_length', return_tensors = 'pt')
            token_story += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_story += types

        for i, h in enumerate(highlights):
            tokens = tokenizer.encode(h)
            token_high += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_high += types

        for i, nh in enumerate(neg_high):
            tokens = tokenizer.encode(nh)
            token_neg_high += tokens
            if i%2 == 0:
                types = [0] * len(tokens)
            else:
                types = [1] * len(tokens)
            token_types_neg_high += types

        story_ids = torch.LongTensor(token_story[:511])
        high_ids = torch.LongTensor(token_high[:512])
        neg_high_ids = torch.LongTensor(token_neg_high[:512])
        att_mask_story = torch.ones_like(story_ids)

        df=df.append({'id': id,
                            'doc_embed': doc_embed,
                            'story_ids': story_ids,
                            'att_mask_story': att_mask_story,
                            'high_ids': high_ids,
                            'neg_high_ids': neg_high_ids},ignore_index=True)

    df.to_pickle(args.save_path + f'cnn_train_{args.preprocess_mode}_4_dataloader.pkl')


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--raw_dataset', type=str, default='../data/cnn/stories', help='Path to dataset')
    parser.add_argument('--save_dataset', type=str, default='../data/', help='Path to save '
    																							 'preprocessed dataset')

    parser.add_argument('--preprocessed_dataset', type=str,
                        default='../data/cnn/preprocessed_ourmethod/cnn_dataset_preprocessed.pkl',
                        help='Path to dataset')

    parser.add_argument('--save_path', type=str, default='../data/all_jsons/',
                        help='Path to save preprocessed dataset')
    
    parser.add_argument('--all_jsons', default='../data/cnn_train_wid.json', type=str)
    parser.add_argument('--load_json', default='../data/test', type=str)
    parser.add_argument('--load_pkl', default='../data/cnn_train_4_dataloader.pkl', type=str)
    
    
    parser.add_argument('--preprocess_mode', default='train', type=str)
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


if __name__ == "__main__":
    set_start_method('spawn')
    from multiprocessing import get_context
    
    args = create_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load stories
    # directory = args.raw_dataset
    # save_directory = args.save_dataset

    # print('directory: ', directory)
    # tokenize(args)
    # format_to_lines(args)

    # stories = load_stories(directory)
    # print('Loaded Stories %d' % len(stories))

    st = time.time()
    #path = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/spring2021/11747_neural_networks_for_NLP
    # /project/data/cnn/preprocessed_lines/.test.0.json'
    # check_tokenize_file(path)
    #clean_and_save_dataset(args.preprocessed_dataset, args.save_dataset_c)
    #select_source_sents_and_partition_dataset(args.preprocessed_dataset, args)
#     preprocess_dataloader(args)
#     partition_jsons(args)
    M = MultiprocessData(args)
#     df = M.pd_wrapper(directory=args.load_json, pattern='*.json')
#     df.to_pickle(args.save_dataset + f'cnn_{args.preprocess_mode}_4_dataloader.pkl')
    #check_pkl_file(args.load_pkl)

    print('time: ', time.time()-st)

    #save to file
    # print('Saving in: ', save_directory)
    # dump(stories, open(save_directory + '/cnn_dataset_preprocessed_2.pkl', 'wb'))
