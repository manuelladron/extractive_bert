import argparse
import datetime
import gc
import psutil
import os
import signal
import sys
import json
from tqdm import tqdm
import glob
import pdb

import PIL
from PIL import Image

import transformers
from transformers import AdamW
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertOnlyMLMHead, BertLayer
from transformers import get_linear_schedule_with_warmup

from dataset import ExtractiveDataset, new_collate, load_dataset, DataloaderMultiple, LazyDataset
from utils import construct_bert_input, save_json, print_memory

import torch, torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# import torch.autograd.profiler as profiler

# torch.set_printoptions(threshold=10_000) # to print full tensor
# torch.autograd.set_detect_anomaly(True)
#PID = os.getpid()


def adaptive_loss(outputs):
    # masked_lm_loss = outputs['masked_lm_loss']
    sent_class_loss = outputs['sent_class_loss']
    doc_story_alig_loss = outputs['doc_story_loss']
    doc_high_alig_loss = outputs['doc_high_loss']
    doc_nmhigh_alig_loss = outputs['doc_nonhigh_loss']
    triplet_loss = outputs['triplet_loss']

#     G = torch.stack([sent_pred_loss, doc_story_alig_loss, doc_high_alig_loss, doc_nmhigh_alig_loss])  # [5]
    G = torch.stack([sent_class_loss, doc_story_alig_loss, doc_high_alig_loss, doc_nmhigh_alig_loss, triplet_loss])  #[5]
    w0 = 1.0
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    w4 = 1.0

    isAdaptive = True
    if isAdaptive:
        logits = torch.nn.Softmax(dim=0)(G)
        nG = logits * logits
        alpha = 1.0
        K = 5.0
        denominator = (alpha * K - nG[0]) * (alpha * K - nG[1]) + (alpha * K - nG[1]) * (alpha * K - nG[2]) + (
        alpha * K - nG[2]) * (alpha * K - nG[3]) + (alpha * K - nG[3]) * (alpha * K - nG[4]) + (alpha * K - nG[4]) * (alpha * K - nG[0])
#         denominator = (alpha * K - nG[0]) * (alpha * K - nG[1]) + (alpha * K - nG[1]) * (alpha * K - nG[2]) + (
#                 alpha * K - nG[2]) * (alpha * K - nG[3]) + (alpha * K - nG[3]) * (alpha * K - nG[0])

#         w0 = (alpha * K - nG[1]) * (alpha * K - nG[2]) * (alpha * K - nG[3]) / denominator
#         w1 = (alpha * K - nG[2]) * (alpha * K - nG[0]) * (alpha * K - nG[3]) / denominator
#         w2 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[3]) / denominator
#         w3 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[2]) / denominator

        w0 = (alpha * K - nG[1]) * (alpha * K - nG[2]) * (alpha * K - nG[3]) * (alpha * K - nG[4]) / denominator
        w1 = (alpha * K - nG[2]) * (alpha * K - nG[0]) * (alpha * K - nG[3]) * (alpha * K - nG[4]) / denominator
        w2 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[3]) * (alpha * K - nG[4]) / denominator
        w3 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[2]) * (alpha * K - nG[4]) / denominator
        w4 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[2]) * (alpha * K - nG[3]) / denominator

        #     adaptive_loss = w0 * masked_lm_loss + w1 * sent_pred_loss + w2 * doc_story_alig_loss + w3 * doc_high_alig_loss + w4 * doc_nmhigh_alig_loss
        adaptive_loss = w0 * sent_class_loss + w1 * doc_story_alig_loss + w2 * doc_high_alig_loss + w3 * doc_nmhigh_alig_loss + w4 * triplet_loss

    return adaptive_loss


class AlignmentHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.alig = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, doc_token):
        alig_score = self.alig(doc_token)
        return alig_score

class SentClassification(torch.nn.Module):
    """
    Use to calculate 1 objective:
        - Sequence prediction: whether the [CLS] tokens of each sent should be included as summary
    """
    def __init__(self, config):
        super().__init__()
        self.bertlayer1 = BertLayer(config)
        self.bertlayer2 = BertLayer(config)
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, sentence_output):
        bert_output = self.bertlayer1(sentence_output)
        bert_output = self.bertlayer2(bert_output[0])
        
        sent_pred_score = self.seq_relationship(bert_output[0])
        return sent_pred_score


class AlignmentPredictionHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 1)
        self.alig = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, sentence_output, doc_output):
        sent_pred_score = self.seq_relationship(sentence_output)
        alig_score = self.alig(doc_output)

        return sent_pred_score, alig_score


class ExtractiveBert(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        #self.mlm = BertOnlyMLMHead(config)
        self.sent_classifier = SentClassification(config)
        #self.alig_pred = AlignmentPredictionHead(config)
        self.alignment = AlignmentHead(config)

        self.triplet_criterion = torch.nn.TripletMarginLoss(margin=2)
        self.init_weights()

    def compute_mlm_loss(self, prediction_scores, labels):
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 ignore index
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return masked_lm_loss

    def compute_doc_sent_loss(self, doc_alig, sent_pred, labels, mask, batch_size, device):
        """Compute Sentence Prediction """
        ### Shapes
        # doc alig: [batch, 2]
        # sent pred: [batch, 512, 2]
        # labels: [batch, 512]
        # mask: [batch, 512]

        # loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        pred_loss = loss_fct(sent_pred.view(-1, 2), labels.long().view(-1))  # [batch*512]

        pred_loss = pred_loss.view(batch_size, -1)
        pred_loss *= mask
        pred_loss = pred_loss.mean()
        # print('pred loss: ', pred_loss)

        """Compute Doc - Stories Alignment """
        loss_fct = torch.nn.CrossEntropyLoss()
        is_paired = torch.ones((batch_size, 1)).to(device)
        alignment_loss = loss_fct(doc_alig.view(-1, 2), is_paired.long().view(-1))
        # print('alignmet loss: ', alignment_loss)

        return pred_loss, alignment_loss

    def compute_doc_alig_loss(self, alignment_scores, batch_size, device, match=True):
        """Compute Doc - Stories Alignment """
        loss_fct = torch.nn.CrossEntropyLoss()
        if match:
            is_paired = torch.ones((batch_size, 1)).to(device)
        else:
            is_paired = torch.zeros((batch_size, 1)).to(device)
        alignment_loss = loss_fct(alignment_scores.view(-1, 2), is_paired.long().view(-1))

        return alignment_loss

    def forward(
            self,
            stories_mlm,  # [batch, 511]
            labels_lm,  # [batch, 511]
            doc_story,  # [batch, 512, 768]
            doc_high,  # [batch, seq_len, 768]
            doc_nonhigh,  # [batch, seq_len, 768]
            story_mask,  # [batch, 511]
            doc_story_mask,
            doc_high_mask,
            doc_nonhigh_mask,
            stories,  # [batch, 511]
            story_labels,
            stories_cls_tokens_mask,  # [batch, seq_len]
            pos_high_cls_tokens_mask,  # [batch, seq_len]
            neg_high_cls_tokens_mask,  # [batch, seq_len]
            device=None

    ):
        batch_size = doc_story.shape[0]
        seq_length = doc_story.shape[1]
        hidden_dim = doc_story.shape[2]

        """1st PASS FOR LANGUAGE MODELING"""
        #         outputs = self.bert(input_ids=stories_mlm, attention_mask=story_mask, return_dict=True)
        #         sequence_output = outputs.last_hidden_state # [batch, 512, 768]

        #         # Only MLM loss
        #         mlm_scores = self.mlm(sequence_output) # mlm scores [batch, 511, 30522]
        #         mlm_loss = self.compute_mlm_loss(mlm_scores, labels_lm)

        """1st PASS FOR DOC - STORY RELATIONSHIP"""
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        outputs = self.bert(inputs_embeds=doc_story,
                            attention_mask=doc_story_mask,
                            return_dict=True)
    
        sequence_output = outputs.last_hidden_state  # [batch, 512, 768]
        doc_output = sequence_output[:, 0, :] # [batch, 768]
        #story_output = sequence_output[:, 1:, :]
        
        # Mask all tokens that are not [CLS] to calculate alignment loss
        sequence_output_ = torch.clone(sequence_output).detach()
        sequence_output_[stories_cls_tokens_mask == -100] = -100

        # Calculate sentence classification
        sent_scores  = self.sent_classifier(sequence_output_)  # [batch, 512, 1] 
        sent_class_loss = 0
        
        for b in range(batch_size):
            sent_scores_b = sent_scores[b, :, :]  # [512, 1]
            stories_cls_b = stories_cls_tokens_mask[b, :]  # [512]
            story_labels_b = story_labels[b, :]  # [num_sents]
            # Now get only CLS tokens
            only_cls_scores = sent_scores_b[stories_cls_b != -100] # [batch, 2]
            # Now do BCE loss
            labels = story_labels_b[:only_cls_scores.shape[0]].long()
            #pdb.set_trace()
            try:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss_b = loss_fct(only_cls_scores, labels)
                #pdb.set_trace()
                #loss_b = F.binary_cross_entropy_with_logits(only_cls_scores.squeeze(), labels)
            except:
                print('len labels: ', labels.shape)
                print('len scores: ', only_cls_scores.shape)
                continue
            sent_class_loss += loss_b

        sent_class_loss /= batch_size

        # Calculate alignment
        # With all tokens 
        all_tokens = sequence_output.mean(dim=1)
        alig_scores = self.alignment(all_tokens) # [batch, 1]
        #alig_scores = self.alignment(doc_output) # [batch, 1]
        doc_story_alig_labels = torch.ones_like(alig_scores).float().to(device)
        doc_story_alig_loss = F.binary_cross_entropy_with_logits(alig_scores, doc_story_alig_labels)


        """2nd PASS FOR DOC - MATCHING HIGHLIGHTS RELATIONSHIP"""
        outputs = self.bert(inputs_embeds=doc_high, attention_mask=doc_high_mask, return_dict=True)
        sequence_output_high = outputs.last_hidden_state  # [batch, 512, 768]
        doc_output = sequence_output[:, 0, :]

        all_tokens = sequence_output_high.mean(dim=1)
        doc_high_alig_scores = self.alignment(all_tokens) # [batch, 1]
        #doc_high_alig_scores = self.alignment(doc_output)
        doc_high_alig_labels = torch.ones_like(doc_high_alig_scores).float().to(device)
        doc_high_alig_loss = F.binary_cross_entropy_with_logits(doc_high_alig_scores, doc_high_alig_labels)

        """4th PASS FOR DOC - NON-MATCHING HIGHLIGHTS RELATIONSHIP"""
        outputs = self.bert(inputs_embeds=doc_nonhigh, attention_mask=doc_nonhigh_mask, return_dict=True)
        sequence_output_nonhigh = outputs.last_hidden_state  # [batch, 512, 768]
        doc_output = sequence_output[:, 0, :]

        # Predict the alignment score (whether embedding of doc + highlights match )
        all_tokens = sequence_output_nonhigh.mean(dim=1)
        doc_nonhigh_alig_scores = self.alignment(all_tokens) # [batch, 1]
        #doc_nonhigh_alig_scores = self.alignment(doc_output)
        doc_nonhigh_alig_labels = torch.zeros_like(doc_nonhigh_alig_scores).float().to(device)
        doc_nonhigh_alig_loss = F.binary_cross_entropy_with_logits(doc_nonhigh_alig_scores, doc_nonhigh_alig_labels)

        ######## Calculate triplet loss
        # Anchor is story, Positive is high, Negative is non-high
        #anchor = sequence_output[stories_cls_tokens_mask == 101]
        anchor = doc_output
        pos = sequence_output_high[pos_high_cls_tokens_mask == 101]
        neg = sequence_output_nonhigh[neg_high_cls_tokens_mask == 101]

        a_len = anchor.shape[0]
        p_len = pos.shape[0]
        n_len = neg.shape[0]

        lens = np.array([a_len, p_len, n_len])
        min_ = np.min(lens)

        anchor = anchor[:min_, :]
        pos = pos[:min_, :]
        neg = neg[:min_, :]

        triplet_loss = self.triplet_criterion(anchor, pos, neg)

        return {
            # "raw_outputs": outputs,
            # "masked_lm_loss": mlm_loss,
            "sent_class_loss": sent_class_loss,
            "doc_story_loss": doc_story_alig_loss,
            "doc_high_loss": doc_high_alig_loss,
            "doc_nonhigh_loss": doc_nonhigh_alig_loss,
            "triplet_loss": triplet_loss
        }


def train(extractive_bert, data_path, params, device):
    torch.manual_seed(0)
    #root = '../data/batches_processed/'
    print('---Loading files...')
    files = sorted(glob.glob(data_path + 'cnn_train_4_dataloader_train_batch_' + '[0-9]*.pkl'))
    datasets = list(map(lambda x: LazyDataset(x), files))
    dataset = torch.utils.data.ConcatDataset(datasets)
#     file = '/Users/manuelladron/iCloud_archive/Documents/_CMU/PHD-CD/spring2021/11747_neural_networks_for_NLP/project/data/cnn/cnn_final/test/cnn_test_4_dataloader_test_batch_0.pkl'
#     dataset = LazyDataset(file)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=new_collate,
    )
    del dataset
    del datasets
    del files
    gc.collect()

    print('loaded, len dataloader: ', len(dataloader))
    extractive_bert.to(device)
    extractive_bert.train()
    opt = transformers.Adafactor(
        extractive_bert.parameters(),
        lr=params.lr,
        beta1=params.beta1,
        weight_decay=params.weight_decay,
        clip_threshold=params.clip,
        relative_step=False,
        scale_parameter=True,
        warmup_init=False
    )

    # For saving model
    dir_name = "../finetuned"
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)

    scheduler = get_linear_schedule_with_warmup(opt, params.num_warmup_steps, params.num_epochs * len(dataloader))

    #     with torch.profiler.profile(
    #     schedule=torch.profiler.schedule(
    #         wait=2,
    #         warmup=2,
    #         active=6,
    #         repeat=1),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/extractivebert'),
    #     ) as profiler:
    for ep in range(params.num_epochs):
        #ep += 2
        print('epoch: ', ep)
        avg_losses = {"sent_class_loss": [], "doc_story_loss": [], "doc_high_loss": [],
                      "doc_nonhigh_loss": [], "triplet_loss": [], "total": []}

        i = 0
        for ids, doc_embed, story_ids, story_labels, story_att_mask, high_ids, neg_high_ids, high_str, \
            story_str in tqdm(dataloader):
            opt.zero_grad()

            #             print('\n-----Shapes---------')
            #             print('doc embed: ',doc_embed.shape)     #[batch, 768]
            #             print('story ids: ', story_ids.shape)    #[batch, 511]
            #             print('attention mask story: ', story_att_mask.shape)  #[batch, 511]
            #             print('high ids: ', high_ids.shape)      #[batch, 512]
            #             print('neg high ids: ', neg_high_ids.shape) #[ #[batch, 512]

            ## FOR LML LOSS, mask tokens and pass in to BERT model
            # mask stories tokens with prob 15%, note id 103 is the [MASK] token
            token_mask = torch.rand(story_ids.shape)
            masked_input_ids = story_ids.detach().clone()
            story_ids_mlm = story_ids.detach().clone()

            masked_input_ids[token_mask < 0.15] = 103  # [MASK] token

            story_ids_mlm[token_mask >= 0.15] = -100  # it doesn't compute these
            story_ids_mlm[story_att_mask == 0] = -100  # it doesn't compute these
            story_ids[story_att_mask == 0] = -100  # it doesn't compute these

            # Mask for highlights ids
            high_ids_mask = high_ids.detach().clone()
            high_ids_mask[high_ids[:, :] > 0] = 1

            # Mask for nonhighlights ids
            neg_high_ids_mask = neg_high_ids.detach().clone()
            neg_high_ids_mask[neg_high_ids[:, :] > 0] = 1

            # FOR DOC + STORIES ALIGNMENT
            doc_story = construct_bert_input(doc_embed, story_ids, extractive_bert, device=device)  # [batch, 512,768]
            # pad attention mask with a 1 on the left so model pays attention to the doc part
            doc_story_mask = F.pad(story_att_mask, (1, 0), value=1)

            # FOR DOC + HIGHLIGHTS ALIGNMENT
            doc_high = construct_bert_input(doc_embed, high_ids, extractive_bert, device=device)  # [batch, 512,768]
            doc_high_mask = F.pad(high_ids_mask, (1, 0), value=1)

            # FOR DOC + NON-HIGHLIGHTS ALIGNMENT
            doc_nonhigh = construct_bert_input(doc_embed, neg_high_ids, extractive_bert,
                                               device=device)  # [batch, 512,768]
            doc_nonhigh_mask = F.pad(neg_high_ids_mask, (1, 0), value=1)

            # Mask out all tokens but [CLS] from stories, highlihgts and nonhighlights to calculate alignment loss
            story_cls_tokens = story_ids.clone()
            high_cls_tokens = high_ids.clone()
            neg_high_cls_tokens = neg_high_ids.clone()

            story_cls_tokens[story_ids != 101] = -100
            high_cls_tokens[high_ids != 101] = -100
            neg_high_cls_tokens[neg_high_ids != 101] = -100

            # Pad with 0 in front because they are used with the document embedding token (construct bert input)
            story_cls_tokens = F.pad(story_cls_tokens, (1, 0), value=0)
            high_cls_tokens = F.pad(high_cls_tokens, (1, 0), value=0)
            neg_high_cls_tokens = F.pad(neg_high_cls_tokens, (1, 0), value=0)

            # pdb.set_trace()
            outputs = extractive_bert(
                stories_mlm=masked_input_ids.to(device),
                labels_lm=story_ids_mlm.to(device),
                doc_story=doc_story.to(device),
                doc_high=doc_high.to(device),
                doc_nonhigh=doc_nonhigh.to(device),
                story_mask=story_att_mask.to(device),
                doc_story_mask=doc_story_mask.to(device),
                doc_high_mask=doc_high_mask.to(device),
                doc_nonhigh_mask=doc_nonhigh_mask.to(device),
                stories=story_ids.to(device),
                story_labels=story_labels.to(device),
                stories_cls_tokens_mask=story_cls_tokens.to(device),
                pos_high_cls_tokens_mask=high_cls_tokens.to(device),
                neg_high_cls_tokens_mask=neg_high_cls_tokens.to(device),
                device=device
            )

#             loss = 2 * outputs['sent_class_loss'] \
#                    + 1 * outputs['doc_story_loss'] \
#                    + 1 * outputs['doc_high_loss'] \
#                    + 1 * outputs['doc_nonhigh_loss'] \
#                    + 1 * outputs["triplet_loss"]

            loss = adaptive_loss(outputs)
            # print('adaptive loss: ', loss.item())
            loss.backward()
            opt.step()
            scheduler.step()
            #print_memory()
#             print('memory stats')
#             print(torch.cuda.memory_stats(device=device))

            for k, v in outputs.items():
                if k in avg_losses:
                    if i % 50 == 0:
                        print(f'{k}: ', v.cpu().item())
                    avg_losses[k].append(v.cpu().item())
            avg_losses["total"].append(loss.cpu().item())
            #print('total loss: ', loss.item())
            if i % 50 == 0:
                print('loss: ', loss.item())
            #                 print('gpu processes:')
            #                 print(torch.cuda.list_gpu_processes(device))
            del outputs
            gc.collect()
            i += 1

        # Saving model
        print('Saving model...')
        model_name = f"extractivebert_cnn_epoch_{ep}"
        save_path = os.path.join(dir_name, model_name)
        print('saving in... ', save_path)
        extractive_bert.save_pretrained(save_directory=save_path)
        save_json(f"{save_path}/train_params.json", params.__dict__)
        print('Saved!')

        print("***************************")
        print(f"At epoch {ep + 1}, losses: ")
        for k, v in avg_losses.items():
            print(f"{k}: {sum(v)/len(v)}")
        print("***************************")
    return avg_losses

def train_iter_fct():
    return DataloaderMultiple(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                              shuffle=True, is_test=False)


def train_(train_iter_fct, train_steps):
    step = 0
    true_batchs = []
    print('training steps: ', train_steps)

    train_iter = train_iter_fct()
    while step <= train_steps:

        for i, batch in enumerate(train_iter):
            print('batch: ', batch)
            true_batchs.append(batch)


class TrainParams:
    lr = 2e-5
    batch_size = 12
    beta1 = 0.95
    beta2 = .999
    weight_decay = 1e-4
    num_warmup_steps = 5000
    num_epochs = 5
    clip = 1.0

def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default='ext', type=str, choices=['ext', 'abs'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../data/all_train_files/')
    parser.add_argument("-train_from", default=None)
    parser.add_argument("-results_path", default='../data/results/')
    parser.add_argument("-batch_size", default=12, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)
    parser.add_argument('-path_to_dataset', help='Absolute path to .json file',
                        default='../data/cnn_train.json')
    parser.add_argument('-disable-cuda', action='store_true',
                        help='Disable CUDA')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_arg()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print('Device: ', device)
    params = TrainParams()

    print('Initiating extractive bert....')
    if args.train_from != None:
        extractive_bert = ExtractiveBert.from_pretrained(args.train_from, return_dict=True)
    else:
        extractive_bert = ExtractiveBert.from_pretrained('bert-base-uncased', return_dict=True)

    try:
        print('training!')
        avg_losses = train(extractive_bert, args.bert_data_path, params, device)
        print('avg losses: ', avg_losses)
    except KeyboardInterrupt:
        pass
    
    dir_name = "../finetuned"
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    model_time = datetime.datetime.now().strftime("%X")
    model_name = f"extractivebert_last_{model_time}"
    save_path = os.path.join(dir_name, model_name)
    print(f"Saving trained model to directory {save_path}...")
    extractive_bert.save_pretrained(save_path)
    save_json(f"{save_path}/train_params.json", params.__dict__)
