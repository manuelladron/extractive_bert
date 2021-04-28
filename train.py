import torch, torchvision
import sys
import json
import torch.nn.functional as F
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import AdamW
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainingHeads
from utils import construct_bert_input, save_json
from transformers import get_linear_schedule_with_warmup
from dataset import ExtractiveDataset, collate

import argparse
import datetime

def adaptive_loss(outputs):
    masked_lm_loss = outputs['masked_lm_loss']
    alignment_loss = outputs['doc_story_loss']
    doc_high_alig_loss = outputs['doc_high_loss']
    doc_nmhigh_alig_loss = outputs['doc_nmhigh_loss']

    G = torch.stack([masked_lm_loss, alignment_loss, doc_high_alig_loss, doc_nmhigh_alig_loss])  #[4]
    w0 = 1.0
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    isAdaptive = True
    if isAdaptive:
        logits = torch.nn.Softmax(dim=0)(G)
        nG = logits * logits
        alpha = 1.0
        K = 4.0
        denominator = (alpha * K - nG[0]) * (alpha * K - nG[1]) + (alpha * K - nG[1]) * (alpha * K - nG[2]) + (
                    alpha * K - nG[2]) * (alpha * K - nG[3]) + (alpha * K - nG[3]) * (alpha * K - nG[0])
        w0 = (alpha * K - nG[1]) * (alpha * K - nG[2]) * (alpha * K - nG[3]) / denominator
        w1 = (alpha * K - nG[2]) * (alpha * K - nG[0]) * (alpha * K - nG[3]) / denominator
        w2 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[3]) / denominator
        w3 = (alpha * K - nG[0]) * (alpha * K - nG[1]) * (alpha * K - nG[2]) / denominator

    adaptive_loss = w0 * masked_lm_loss + w1 * alignment_loss + w2 * doc_high_alig_loss + w3 * doc_nmhigh_alig_loss

    return adaptive_loss

class ExtractiveBertHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = torch.nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class ExtractiveBert(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)

        self.im_to_embedding = torch.nn.Linear(2048, 768)
        self.im_to_embedding_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.cls = BertPreTrainingHeads(config)
        self.doc_high_alignment = ExtractiveBertHead(config)

        self.init_weights()

    def compute_doc_sent_loss(self, prediction_scores, alignment_scores, labels, batch_size, device):
        """Compute Mask Language Model Loss """
        # Compute masked language loss
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        """Compute Doc - Stories Alignment """
        loss_fct = torch.nn.CrossEntropyLoss()
        is_paired = torch.ones((batch_size, 1)).to(device)
        alignment_loss = loss_fct(alignment_scores.view(-1, 2), is_paired.long().view(-1))

        return masked_lm_loss, alignment_loss

    def compute_doc_high_loss(self, alignment_scores, batch_size, device, match=True):
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
            embeds,
            attention_mask,
            labels, 
            pos_highlights_ids=None,
            neg_highlights_ids=None,
            device=None
    ):
        """
            Args:
                embeds
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
        batch_size = embeds.shape[0]
        seq_length = embeds.shape[1]
        hidden_dim = embeds.shape[2]

        """1st PASS FOR DOC - STORY RELATIONSHIP"""
        outputs = self.bert(inputs_embeds=embeds,
                            attention_mask=attention_mask,
                            return_dict=True)

        """ Pooled output is the doc embedding """
        sequence_output = outputs.last_hidden_state # [batch, 512, 768]
        pooled_output = outputs.pooler_output       # [batch, 768]

        # hidden states corresponding to document token
        doc_output = sequence_output[:, 0, :]   # [batch, 768]
        # hidden states corresponding to the text part
        text_output = sequence_output[:, 1:, :] # [batch, 511, 768]

        # Predict the masked text tokens and alignment scores (whether embedding of doc + stories match )
        prediction_scores, alignment_scores = self.cls(text_output, pooled_output)
        """
        prediction scores [batch, 511, 30522]
        alignment scores [batch, 2]
        """
        
        mlm_loss, alig_loss = self.compute_doc_sent_loss(prediction_scores, alignment_scores, labels, batch_size, device)
        print('mlm loss: ', mlm_loss)
        print('alig loss: ', alig_loss)
        
        """2nd PASS FOR DOC - MATCHING HIGHLIGHTS RELATIONSHIP"""
        outputs = self.bert(input_ids=pos_highlights_ids, return_dict=True)

        """ Pooled output is the [CLS] token """
        pooled_output = outputs.pooler_output

        # Predict the alignment score (whether embedding of doc + highlights match )
        alignment_scores = self.doc_high_alignment(pooled_output)
        doc_high_alig_loss = self.compute_doc_high_loss(alignment_scores, batch_size, device, match=True)
        print('doc high alig loss: ', doc_high_alig_loss)
        
        """3rd PASS FOR DOC - NON-MATCHING HIGHLIGHTS RELATIONSHIP"""
        outputs = self.bert(input_ids=neg_highlights_ids, return_dict=True)

        """ Pooled output is the [CLS] token """
        pooled_output = outputs.pooler_output

        # Predict the alignment score (whether embedding of doc + highlights match )
        alignment_scores = self.doc_high_alignment(pooled_output)
        doc_nmhigh_alig_loss = self.compute_doc_high_loss(alignment_scores, batch_size, device, match=False)
        print('doc nmhigh alig loss: ', doc_nmhigh_alig_loss)
        
        return {
            #"raw_outputs": outputs,
            "masked_lm_loss": mlm_loss,
            "doc_story_loss": alig_loss,
            "doc_high_loss": doc_high_alig_loss,
            "doc_nmhigh_loss": doc_nmhigh_alig_loss
        }


def train(extractive_bert, train_set, params, device):
    torch.manual_seed(0)

    dataset = ExtractiveDataset(train_set, device)
    print('loading data loader..')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
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

    scheduler = get_linear_schedule_with_warmup(opt, params.num_warmup_steps, params.num_epochs * len(dataloader))

    for ep in range(params.num_epochs):
        print('epoch: ', ep)
        avg_losses = {"masked_lm_loss": [], "masked_patch_loss": [], "alignment_loss": [], "total": []}
        for doc_embed, story_ids, story_att_mask, high_ids, neg_high_ids in dataloader:
            opt.zero_grad()

#             print('\n-----Shapes---------')
#             print('doc embed: ',doc_embed.shape)     #[batch, 768]
#             print('story ids: ', story_ids.shape)    #[batch, 511]
#             print('attention mask story: ', story_att_mask.shape)  #[batch, 511]
#             print('high ids: ', high_ids.shape)      #[batch, 512]
#             print('neg high ids: ', neg_high_ids.shape) #[ #[batch, 512]

            # mask stories tokens with prob 15%, note id 103 is the [MASK] token
            token_mask = torch.rand(story_ids.shape)
            masked_input_ids = story_ids.detach().clone()
            masked_input_ids[token_mask < 0.15] = 103 # [MASK] token

            story_ids[token_mask >= 0.15] = -100   # it doesn't compute these
            story_ids[story_att_mask == 0] = -100  # it doesn't compute these

            print('\nCalling construct bert input...')
            embeds = construct_bert_input(doc_embed, masked_input_ids, extractive_bert, device=device) # [batch, 512,768]
            # pad attention mask with a 1 on the left so model pays attention to the doc part
            attention_mask  = F.pad(story_att_mask, (1, 0), value=1)

            outputs = extractive_bert(
                embeds=embeds.to(device),
                attention_mask=attention_mask.to(device),
                labels = story_ids.to(device),
                pos_highlights_ids=high_ids.to(device),
                neg_highlights_ids=neg_high_ids.to(device),
                device=device
            )

            loss = adaptive_loss(outputs)
            print('adaptive loss: ', loss)
            loss.backward()
            opt.step()
            scheduler.step()
            
            for k, v in outputs.items():
                if k in avg_losses:
                    avg_losses[k].append(v.cpu().item())
            avg_losses["total"].append(loss.cpu().item())
            print('loop complete')
            
        if (ep % 5 == 0 and ep != 0) or ep == params.num_epochs - 1:
            print('Saving!')
            model_name = f'extractivebert_cnn_epoch_{ep}'
            ExtractiveBert.save_pretrained(model_name)
            save_json(f"{model_name}/train_params.json", params.__dict__)

        print("***************************")
        print(f"At epoch {ep + 1}, losses: ")
        for k, v in avg_losses.items():
            print(f"{k}: {sum(v) / len(v)}")
        print("***************************")


class TrainParams:
    lr = 2e-5
    batch_size = 2
    beta1 = 0.95
    beta2 = .999
    weight_decay = 1e-4
    num_warmup_steps = 5000
    num_epochs = 10
    clip = 1.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ExtractiveBERT.')
    #parser.add_argument('--path_to_images', help='Absolute path to image directory', default='/Users/alexschneidman/Documents/CMU/CMU/F20/777/ADARI/v2/full')
    #parser.add_argument('--path_to_data_dict', help='Absolute path to json containing img name, sentence pair dict', default='/Users/alexschneidman/Documents/CMU/CMU/F20/777/ADARI/ADARI_furniture_pairs.json')
    parser.add_argument('--path_to_dataset', help='Absolute path to .json file',
                        default='../data/cnn_train.json')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)
    args = parser.parse_args()
    params = TrainParams()

    print('Initiating extractive bert....')
    extractive_bert = ExtractiveBert.from_pretrained('bert-base-uncased', return_dict=True)
    #dataset = MultiModalBertDataset(
    #    args.path_to_images,
    #    args.path_to_data_dict,
    #    device=device,
    #    )
    #dataset = ExtractiveDataset(args.path_to_dataset, device)

    try:
        print('training!')
        train(extractive_bert, args.path_to_dataset, params, device)
    except KeyboardInterrupt:
        pass
    model_time = datetime.datetime.now().strftime("%X")
    model_name = f"fashionbert_{model_time}"
    print(f"Saving trained model to directory {model_name}...")
    fashion_bert.save_pretrained(model_name)
    save_json(f"{model_name}/train_params.json", params.__dict__)
