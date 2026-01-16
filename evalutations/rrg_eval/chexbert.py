from typing import List
import os
from collections import OrderedDict
import re
import json
import random

import torch
import numpy as np
from transformers import BertModel, AutoModel, BertTokenizer
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from rrg_eval.factuality_utils import CheXbert_CONDITIONS, CONDITIONS_5, map_to_binary
from rrg_eval.factuality_eval import (
    generate_classification_report,
    test
) 

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class UnlabeledDataset(torch.utils.data.Dataset):
    """The dataset to contain report impressions without any labels."""
        
    def __init__(self, reports: List[str], tokenizer: BertTokenizer):
        self.encoded_imp = self.tokenize(reports, tokenizer)

    def tokenize(self, reports, tokenizer):
        rets = []
        for r in reports:
            r = r.strip().replace('\n', ' ')
            r = re.sub(r'\s+', ' ', r)
            tokenized_r = tokenizer.tokenize(r)
            if tokenized_r:  # not an empty report
                res = tokenizer.encode_plus(tokenized_r)['input_ids']
                if len(res) > 512:  # length exceeds maximum size
                    # print("report length bigger than 512")
                    res = res[:511] + [tokenizer.sep_token_id]
                rets.append(res)
            else:  # an empty report
                rets.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
        return rets

    def __len__(self):
        """Compute the length of the dataset

        @return (int): size of the dataframe
        """
        return len(self.encoded_imp)

    def __getitem__(self, idx):
        """ Functionality to index into the dataset
        @param idx (int): Integer index into the dataset

        @return (dictionary): Has keys 'imp', 'label' and 'len'. The value of 'imp' is
                              a LongTensor of an encoded impression. The value of 'label'
                              is a LongTensor containing the labels and 'the value of
                              'len' is an integer representing the length of imp's value
        """
        if torch.is_tensor(idx):
                idx = idx.tolist()
        imp = self.encoded_imp[idx]
        imp = torch.LongTensor(imp)
        return {"imp": imp, "len": imp.shape[0]}


def collate_fn_no_labels(sample_list):
    """Custom collate function to pad reports in each batch to the max len,
       where the reports have no associated labels
    @param sample_list (List): A list of samples. Each sample is a dictionary with
                               keys 'imp', 'len' as returned by the __getitem__
                               function of ImpressionsDataset

    @returns batch (dictionary): A dictionary with keys 'imp' and 'len' but now
                                 'imp' is a tensor with padding and batch size as the
                                 first dimension. 'len' is a list of the length of 
                                 each sequence in batch
    """
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=0
    )
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch


def load_unlabeled_data(reports, tokenizer, batch_size=18, num_workers=4, shuffle=False):
    """ Create UnlabeledDataset object for the input reports
    @param reports (list[string]): radiology reports
    @param batch_size (int): the batch size. As per the BERT repository, the max batch size
                             that can fit on a TITAN XP is 6 if the max sequence length
                             is 512, which is our case. We have 3 TITAN XP's
    @param num_workers (int): how many worker processes to use to load data
    @param shuffle (bool): whether to shuffle the data or not  
    
    @returns loader (dataloader): dataloader object for the reports
    """
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(reports, tokenizer)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn
    )
    return loader


class bert_labeler(torch.nn.Module):
    def __init__(self, p=0.1, clinical=False, freeze_embeddings=False, pretrain_path=None):
        """ Init the labeler module
        @param p (float): p to use for dropout in the linear heads, 0.1 by default is consistant with 
                          transformers.BertForSequenceClassification
        @param clinical (boolean): True if Bio_Clinical BERT desired, False otherwise. Ignored if
                                   pretrain_path is not None
        @param freeze_embeddings (boolean): true to freeze bert embeddings during training
        @param pretrain_path (string): path to load checkpoint from
        """
        super(bert_labeler, self).__init__()

        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
                
        self.dropout = torch.nn.Dropout(p)
        #size of the output of transformer's last layer
        hidden_size = self.bert.pooler.dense.in_features
        #classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = torch.nn.ModuleList([torch.nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        #classes: yes, no for the 'no finding' observation
        self.linear_heads.append(torch.nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded, attention_mask):
        """ Forward pass of the labeler
        @param source_padded (torch.LongTensor): Tensor of word indices with padding, shape (batch_size, max_len)
        @param attention_mask (torch.Tensor): Mask to avoid attention on padding tokens, shape (batch_size, max_len)
        @returns out (List[torch.Tensor])): A list of size 14 containing tensors. The first 13 have shape 
                                            (batch_size, 4) and the last has shape (batch_size, 2)  
        """
        #shape (batch_size, max_len, hidden_size)
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]
        #shape (batch_size, hidden_size)
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden = self.dropout(cls_hidden)
        out = []
        for i in range(14):
            out.append(self.linear_heads[i](cls_hidden))
        return out


class CheXbert(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load model
        model = bert_labeler()
        checkpoint = torch.load(hf_hub_download(repo_id='StanfordAIMI/RRG_scorers', filename="chexbert.pth"), map_location=torch.device('cpu'))
        # Data parallel init introduces over head.
        if False: # torch.cuda.device_count() > 0: #works even if only 1 GPU available
            print("Using", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model) #to utilize multiple GPU's
            model = model.to(self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_state_dict'].items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        self.model = model.to(self.device).eval()

        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, reports: List[str]):
        dataloader  = load_unlabeled_data(reports, self.tokenizer, batch_size=128, num_workers=4)
        y_pred = [[] for _ in range(len(CheXbert_CONDITIONS))]

        for i, data in enumerate(tqdm(dataloader, disable=True)):
            batch = data['imp'] #(batch_size, max_len)
            batch = batch.to(self.device)
            src_len = data['len']
            attn_mask = self.generate_attention_masks(batch, src_len)

            out = self.model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1) #shape is (batch_size)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)

        y_pred = [t.tolist() for t in y_pred]
        return y_pred

    def generate_attention_masks(self, batch, source_lengths):
        """Generate masks for padded batches to avoid self-attention over pad tokens
        @param batch (Tensor): tensor of token indices of shape (batch_size, max_len)
                               where max_len is length of longest sequence in the batch
        @param source_lengths (List[Int]): List of actual lengths for each of the
                               sequences in the batch
        @param device (torch.device): device on which data should be

        @returns masks (Tensor): Tensor of masks of shape (batch_size, max_len)
        """
        masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
        for idx, src_len in enumerate(source_lengths):
            masks[idx, src_len:] = 0
        return masks.to(self.device)


from scipy.stats import bootstrap

pbar = None

def compute_statistic(preds, refs, key):
    def _compute(indices):
        global pbar
        p = [preds[i] for i in indices]
        r = [refs[i] for i in indices]
        _, _, avg_f1, _ = precision_recall_fscore_support(
            y_pred=p,
            y_true=r,
            average=key
        )
        pbar.update()
        return avg_f1

    return _compute


def bootstrap_confidence_interval(
        preds, refs, key, n_resamples: int = 500, method: str = "percentile",
    ):
    global pbar
    pbar = tqdm(
        total=len(preds) + n_resamples + 1 if method == "BCa" else n_resamples,
        desc=f"bootstrap {key} f1 95% CI"
    )
    bs = bootstrap(
        data=(list(range(len(preds))),),
        statistic=compute_statistic(preds, refs, key),
        method=method,
        paired=False,
        vectorized=False,
        confidence_level=0.95,
        random_state=3,
        n_resamples=n_resamples
    )
    return bs


def bootstrap_breakdown_scores(pred, gold, names, n_resamples: int = 500):
    support = precision_recall_fscore_support(y_pred=pred, y_true=gold)[-1].tolist()
    indices = list(range(len(pred)))
    precisions, recalls, f1s = [], [], []
    for _ in tqdm(range(n_resamples)):
        _indices = random.choices(indices, k=len(indices))
        _p, _g = [pred[i] for i in _indices], [gold[i] for i in _indices]
        prec, recall, f1, _ = precision_recall_fscore_support(y_pred=_p, y_true=_g, zero_division=0)
        precisions.append(prec)
        recalls.append(recall)
        f1s.append(f1)
    all_samples = np.stack([np.stack(precisions), np.stack(recalls), np.stack(f1s)])
    # input: [3, 500, 14]
    # output: [3, 14]
    ci_l, m, ci_h = np.percentile(all_samples, [2.5, 50, 97.5], axis=1)
    ci_l, m, ci_h = ci_l.tolist(), m.tolist(), ci_h.tolist()
    ret = {}
    for i, name in sorted(enumerate(names), key=lambda x: x[1]):
        metrics = {}
        for j, metric in enumerate(["precision", "recall", "f1"]):
            metrics[metric] = m[j][i]
            metrics[f"{metric}_ci"] = [ci_l[j][i], ci_h[j][i]] 
        metrics["support"] = support[i]
        ret[name] = metrics
    return ret


def evaluate(
        preds: List[str],
        refs: List[str],
        include_original: bool = True,
        bootstrap_ci: bool = False,
        save_breakdown: bool = False
    ):
    model = CheXbert()
    # [condition_size, num_samples]
    outputs = model(preds + refs)
    rets = [list(map(map_to_binary, i)) for i in outputs]

    # Vilmedic CheXbert 14
    # [num_samples, condition_size]
    binary_rets = list(map(list, zip(*rets)))
    vilmedic_cr = generate_classification_report(
        y_pred=binary_rets[:len(preds)], y_true=binary_rets[len(preds):], target_names=CheXbert_CONDITIONS
    )

    if bootstrap_ci:
        for key in ("micro", "macro"):
            # _, _, avg_f1, _ = precision_recall_fscore_support(
            #     y_pred=binary_rets[:len(preds)],
            #     y_true=binary_rets[len(preds):],
            #     average=key
            # )
            # assert vilmedic_cr[f"{key} avg"]["f1-score"] == avg_f1, breakpoint()
            bs = bootstrap_confidence_interval(
                binary_rets[:len(preds)], binary_rets[len(preds):], key
            )
            key = key + " avg"
            vilmedic_cr[key]["median"] = np.median(bs.bootstrap_distribution)
            vilmedic_cr[key]["ci_l"] = bs.confidence_interval.low
            vilmedic_cr[key]["ci_h"] = bs.confidence_interval.high


    # Vilmedic CheXbert 5
    conditions_5_index = np.where(np.isin(CheXbert_CONDITIONS, CONDITIONS_5))[0]
    rets_5 = [rets[i] for i in conditions_5_index]
    # [num_samples, condition_size]
    binary_rets_5 = list(map(list, zip(*rets_5)))
    vilmedic_cr5 = generate_classification_report(
        y_pred=binary_rets_5[:len(preds)], y_true=binary_rets_5[len(preds):], target_names=CONDITIONS_5
    )

    if bootstrap_ci:
        for key in ("micro", "macro"):
            # _, _, avg_f1, _ = precision_recall_fscore_support(
            #     y_pred=binary_rets_5[:len(preds)],
            #     y_true=binary_rets_5[len(preds):],
            #     average=key
            # )
            # assert vilmedic_cr5[f"{key} avg"]["f1-score"] == avg_f1, breakpoint()
            bs = bootstrap_confidence_interval(
                binary_rets_5[:len(preds)], binary_rets_5[len(preds):], key
            )
            key = key + " avg"
            vilmedic_cr5[key]["median"] = np.median(bs.bootstrap_distribution)
            vilmedic_cr5[key]["ci_l"] = bs.confidence_interval.low
            vilmedic_cr5[key]["ci_h"] = bs.confidence_interval.high


    rets_p = [list(map(lambda x: map_to_binary(x, "rrg+"), i)) for i in outputs]

    # Vilmedic CheXbert 14+
    # [num_samples, condition_size]
    binary_rets_p = list(map(list, zip(*rets_p)))
    vilmedic_cr_p = generate_classification_report(
        y_pred=binary_rets_p[:len(preds)], y_true=binary_rets_p[len(preds):], target_names=CheXbert_CONDITIONS
    )

    if bootstrap_ci:
        for key in ("micro", "macro"):
            bs = bootstrap_confidence_interval(
                binary_rets_p[:len(preds)], binary_rets_p[len(preds):], key
            )
            key = key + " avg"
            vilmedic_cr_p[key]["median"] = np.median(bs.bootstrap_distribution)
            vilmedic_cr_p[key]["ci_l"] = bs.confidence_interval.low
            vilmedic_cr_p[key]["ci_h"] = bs.confidence_interval.high

    # Vilmedic CheXbert 5
    conditions_5_index = np.where(np.isin(CheXbert_CONDITIONS, CONDITIONS_5))[0]
    rets_5_p = [rets_p[i] for i in conditions_5_index]
    # [num_samples, condition_size]
    binary_rets_5_p = list(map(list, zip(*rets_5_p)))
    vilmedic_cr5_p = generate_classification_report(
        y_pred=binary_rets_5_p[:len(preds)], y_true=binary_rets_5_p[len(preds):], target_names=CONDITIONS_5
    )

    if bootstrap_ci:
        for key in ("micro", "macro"):
            bs = bootstrap_confidence_interval(
                binary_rets_5_p[:len(preds)], binary_rets_5_p[len(preds):], key
            )
            key = key + " avg"
            vilmedic_cr5_p[key]["median"] = np.median(bs.bootstrap_distribution)
            vilmedic_cr5_p[key]["ci_l"] = bs.confidence_interval.low
            vilmedic_cr5_p[key]["ci_h"] = bs.confidence_interval.high


    cr = {}
    if include_original:
        # The original metrics used in CheXbert
        y_pred = [y_cond[:len(preds)] for y_cond in outputs]
        y_true = [y_cond[len(preds):] for y_cond in outputs]
        cr = test(y_true, y_pred)


    if save_breakdown:
        breakdown = {}
        breakdown["CheXbert_F1_uncertain_as_negative"] = bootstrap_breakdown_scores(
            binary_rets[:len(preds)], binary_rets[len(preds):], CheXbert_CONDITIONS
        )
        breakdown["CheXbert_F1_uncertain_as_positive"] = bootstrap_breakdown_scores(
            binary_rets_p[:len(preds)], binary_rets_p[len(preds):], CheXbert_CONDITIONS
        )
        with open(save_breakdown, "w") as f:
            json.dump(breakdown, f, indent=2)

    return vilmedic_cr, vilmedic_cr5, vilmedic_cr_p, vilmedic_cr5_p, cr


def evaluate2(hyps, refs):
    return evaluate(preds=hyps, refs=refs)


if __name__ == '__main__':
    import json

    hyps= [
        'No pleural effusion. Normal heart size.',
        'Normal heart size.',
        'Increased mild pulmonary edema and left basal atelectasis.',
        'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
        'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
    ]
    refs= [
        'No pleural effusions.',
        'Enlarged heart.',
        'No evidence of pneumonia. Stable cardiomegaly.',
        'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
        'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.'
    ]
    cr, cr5 = evaluate(hyps, refs)
    
    print(json.dumps(cr, indent=4))
    print(f"Micro_avg_f1_score_14classes: {cr['micro avg']['f1-score']}")