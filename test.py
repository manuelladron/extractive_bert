import torch, torchvision
import sys
import json
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertOnlyMLMHead
from utils import construct_bert_input, save_json
from dataset import ExtractiveDataset, new_collate, load_dataset, DataloaderMultiple, LazyDataset
from tqdm import tqdm
import glob
import pdb
#torch.set_printoptions(threshold=10_000) # to print full tensor
import argparse
import datetime
import gc
import pickle
import psutil
from utils import print_memory
from others.utils import test_rouge, rouge_results_to_str
import numpy as np


class SentClassification(torch.nn.Module):
    """
    Use to calculate 1 objectivves:
        - Sequence prediction: whether the [CLS] tokens of each sent should be included as summary

    """
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, sentence_output):
        sent_pred_score = self.seq_relationship(sentence_output)
        return sent_pred_score


class AlignmentPredictionHead(torch.nn.Module):
    """
    Use to calculate 2 objectivves:
        - Sequence prediction: whether the [CLS] tokens of each sent should be included as summary
        - Alignment between doc and sentences: whether Doc and [CLS] of each sent are deemed to matched
    """
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 1)
        self.alig = torch.nn.Linear(config.hidden_size, 1)

    def forward(self, sentence_output, doc_output):
        sent_pred_score = self.seq_relationship(sentence_output)
        alig_score = self.alig(doc_output)

        return sent_pred_score, alig_score

class ExtractiveBertTest(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        #self.mlm = BertOnlyMLMHead(config)
        self.sent_classifier = SentClassification(config)
        #self.alig_pred = AlignmentPredictionHead(config) # sent classification and doc-sents alignment
        #self.init_weights()

    def forward(
            self,
            doc_story,                # [batch, 512, 768]
            doc_story_mask,           # [batch, 512]
            stories_cls_tokens_mask,  # [batch, 512]
            stories,                  # [batch, 511]
            story_labels,             # [batch, num_sents]
            story_str,                # batch
            high_str,
            step,
            device=None,

    ):
        batch_size = doc_story.shape[0]

        """PASS FOR DOC - STORY RELATIONSHIP"""
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        outputs = self.bert(inputs_embeds=doc_story,
                            attention_mask=doc_story_mask,
                            return_dict=True)

        sequence_output = outputs.last_hidden_state  # [batch, 512, 768]
        #doc_output = sequence_output[:, 0, :]

        # Mask all tokens that are not [CLS] to calculate alignment loss
        sequence_output_ = torch.clone(sequence_output).detach()
        sequence_output_[stories_cls_tokens_mask == -100] = -100

        sent_scores = self.sent_classifier(sequence_output_)  # [batch, 512, 1] and [batch, 1]

        b_candidates_ids = []
        b_labels_ids = []
        losses = 0

        pred = []
        gold = []

        for b in range(batch_size):
            sent_scores_b = sent_scores[b, :, :]           # [512, 1]
            stories_cls_b = stories_cls_tokens_mask[b, :]  # [512]
            story_labels_b = story_labels[b, :]            # [num_sents]
            # Now get only CLS tokens
            only_cls_scores = sent_scores_b[stories_cls_b != -100]
            # Now do BCE loss
            labels = story_labels_b[:only_cls_scores.shape[0]].float()
            loss_b = F.binary_cross_entropy_with_logits(only_cls_scores.squeeze(), labels)
            losses += loss_b

            # Now select ids from candidates and labels
            selected_ids = torch.argsort(only_cls_scores, dim=0, descending=True)
            labels_ids = torch.where(labels == 1)[0]
            b_candidates_ids.append(selected_ids.cpu().numpy())
            b_labels_ids.append(labels_ids.cpu().numpy())

            _pred = []
            for i, idx in enumerate(selected_ids):

                if len(story_str[b][idx]) == 0:
                    continue
                candidate = story_str[b][idx].strip()
                _pred.append(candidate)
                if len(_pred) == 3:
                    break

            #_pred = '<q>'.join(_pred)
            #pred.append(_pred)
            _gold = high_str[b][:3]
            if len(_gold) < len(_pred):
                _pred = _pred[:len(_gold)]
            elif len(_gold) > len(_pred):
                _gold = _gold[:len(_pred)]

            pred += _pred
            gold += _gold

        losses /= batch_size
        print('loss_b: ', losses)
        return pred, gold


def test(extractive_bert, test_set_path, params, device):

    # Set model in validating mode.
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    torch.manual_seed(0)

    #file = '../data/cnn_test_4_dataloader_test_batch_0.pkl'
    dataset = LazyDataset(test_set_path)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=new_collate,
    )
    del dataset

    gc.collect()
    print('loaded!')
    #print('len dataloader: ', len(dataloader))
    extractive_bert.to(device)
    extractive_bert.eval()

    PRED, GOLD = [], []
    with torch.no_grad():

        avg_losses = {"masked_lm_loss": [], "sent_pred_loss": [], "doc_story_loss": [], "doc_high_loss": [],
                      "doc_nmhigh_loss": [], "total": []}
        I = 0
        for ids, doc_embed, story_ids, story_labels, story_att_mask, high_ids, neg_high_ids, high_str, \
            story_str in tqdm(dataloader):

            token_mask = torch.rand(story_ids.shape)
            masked_input_ids = story_ids.detach().clone()
            story_ids_mlm = story_ids.detach().clone()

            masked_input_ids[token_mask < 0.15] = 103  # [MASK] token

            story_ids_mlm[token_mask >= 0.15] = -100  # it doesn't compute these
            story_ids_mlm[story_att_mask == 0] = -100  # it doesn't compute these

            story_ids[story_att_mask == 0] = -100  # it doesn't compute these

            doc_story = construct_bert_input(doc_embed, story_ids, extractive_bert, device=device) # [batch, 512,768]
            doc_story_mask  = F.pad(story_att_mask, (1, 0), value=1) # [batch, 512]

            story_cls_tokens = story_ids.clone()
            story_cls_tokens[story_ids != 101] = -100
            story_cls_tokens = F.pad(story_cls_tokens, (1, 0), value=-100)

            pred, gold = extractive_bert(
                doc_story=doc_story.to(device),
                doc_story_mask=doc_story_mask.to(device),
                stories_cls_tokens_mask=story_cls_tokens.to(device),
                stories=story_ids.to(device),
                story_labels = story_labels.to(device),
                story_str=story_str,
                high_str=high_str,
                step=I,
                device=device,
            )
            PRED += pred
            GOLD += gold

            I += 1
            #if I > 2: break

    results_path = args.results_path
    train_ep = 0
    can_path = '%s_step%d.candidate' % (results_path, train_ep)
    gold_path = '%s_step%d.gold' % (results_path, train_ep)
    with open(can_path, 'w') as save_pred:
        with open(gold_path, 'w') as save_gold:
            for i in range(len(gold)): # for each batch
                save_gold.write(gold[i].strip() + '\n')
            for i in range(len(pred)):
                save_pred.write(pred[i].strip() + '\n')

    temp_dir = '../data/temp/'
    rouges = test_rouge(temp_dir, can_path, gold_path)
    print('-----ROUGES------')
    print(rouge_results_to_str(rouges))

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
    parser.add_argument("-mode", default='test', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-bert_data_path", default='../data/cnn_test_4_dataloader_test_batch_0.pkl')
    parser.add_argument("-trained_model", default='../finetuned/extractivebert_cnn_epoch_3/')
    parser.add_argument("-results_path", default='../data/results/')

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
    print(args.trained_model)
    extractive_bert = ExtractiveBertTest.from_pretrained(args.trained_model, return_dict=True)

    try:
        print('testing...')
        test(extractive_bert, args.bert_data_path, params, device)

    except KeyboardInterrupt:
        pass

    # dir_name = "../finetuned"
    # os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    # model_time = datetime.datetime.now().strftime("%X")
    # model_name = f"extractivebert_last_{model_time}"
    # save_path = os.path.join(dir_name, model_name)
    # print(f"Saving trained model to directory {save_path}...")
    # extractive_bert.save_pretrained(save_path)
    # save_json(f"{save_path}/train_params.json", params.__dict__)
