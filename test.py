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
import psutil
from utils import print_memory


class ExtractiveBertTest(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlm = BertOnlyMLMHead(config)
        self.alig_pred = AlignmentPredictionHead(config)
        self.alignment = AlignmentHead(config)

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
            neg_highl_cls_tokens_mask,  # [batch, seq_len]
            device=None

    ):
        """
            Args:
                doc_story
                    hidden embeddings to pass to the bert model
                        batch size, seq length, hidden dim
                attention_mask
                    batch size, seq length
                labels: unmasked inputs ids
                    batch size, seq length,
                pos_highlights_ids
                    Unmasked tokenized token ids corresponding to matching highlights
                        batch size, word sequence length
                neg_highlights_ids
                    Unmasked tokenized token ids corresponding to random non-matching highlights
                        batch size, word sequence length
        """
        #         print('stories_mlm: ', stories_mlm.shape)
        #         print('doc_story: ', doc_story.shape)
        #         print('doc_high: ', doc_high.shape)
        #         print('doc_nonhigh: ', doc_nonhigh.shape)
        #         print('story_mask: ', story_mask.shape)
        #         print('story: ', stories.shape)
        #         print('labels: ', labels_lm.shape)

        batch_size = doc_story.shape[0]
        seq_length = doc_story.shape[1]
        hidden_dim = doc_story.shape[2]

        """PASS FOR DOC - STORY RELATIONSHIP"""
        # with profiler.profile(with_stack=True, profile_memory=True) as prof:
        outputs = self.bert(inputs_embeds=doc_story,
                            attention_mask=doc_story_mask,
                            return_dict=True)

        sequence_output = outputs.last_hidden_state  # [batch, 512, 768]
        doc_output = sequence_output[:, 0, :]

        # Mask all tokens that are not [CLS] to calculate alignment loss
        sequence_output_ = torch.clone(sequence_output).detach()
        sequence_output_[stories_cls_tokens_mask == 0] = -100

        seq_len = sequence_output.shape[1]

        sent_pred, doc_alig_pred = self.alig_pred(sequence_output, doc_output)  # [2, 512, 2] and [2, 512]

        sent_loss, doc_story_alig_loss = self.compute_doc_sent_loss(doc_alig_pred, sent_pred, story_labels,
                                                                    stories_cls_tokens_mask, batch_size, device)


        return {
            # "raw_outputs": outputs,
            # "masked_lm_loss": mlm_loss,
            "sent_pred_loss": sent_loss,
            "doc_story_loss": doc_story_alig_loss,
            "doc_high_loss": doc_high_alig_loss,
            "doc_nmhigh_loss": doc_nonhigh_alig_loss
        }


def test(extractivebert, test_set, params, device):

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
    root = '../data/batches_processed/'
    print('---Loading files...')
    files = sorted(glob.glob(root + 'cnn_test_4_dataloader_batch_' + '[0-9]*.pkl'))
    datasets = list(map(lambda x: LazyDataset(x), files))
    dataset = torch.utils.data.ConcatDataset(datasets)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=new_collate,
    )
    del dataset
    del datasets
    del files
    gc.collect()

    print('loaded, len dataloader: ', len(dataloader))
    extractive_bert.to(device)
    extractive_bert.eval()

    with torch.no_grad():
        for ep in range(params.num_epochs):
            print('epoch: ', ep)

            avg_losses = {"masked_lm_loss": [], "sent_pred_loss": [], "doc_story_loss": [], "doc_high_loss": [],
                          "doc_nmhigh_loss": [], "total": []}
            i = 0
            for ids, doc_embed, story_ids, story_labels, story_att_mask, high_ids, neg_high_ids, high_str in tqdm(
                    dataloader):

                token_mask = torch.rand(story_ids.shape)
                masked_input_ids = story_ids.detach().clone()
                story_ids_mlm = story_ids.detach().clone()

                masked_input_ids[token_mask < 0.15] = 103  # [MASK] token

                story_ids_mlm[token_mask >= 0.15] = -100  # it doesn't compute these
                story_ids_mlm[story_att_mask == 0] = -100  # it doesn't compute these
                story_ids[story_att_mask == 0] = -100  # it doesn't compute these

                doc_story = construct_bert_input(doc_embed, story_ids, extractive_bert, device=device) # [batch, 512,768]
                doc_story_mask  = F.pad(story_att_mask, (1, 0), value=1)

