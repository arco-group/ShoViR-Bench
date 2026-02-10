# run_eval_chexbert_class.py
"""
Radiology report generation evaluation script (CheXbert single-class only).

It loads a JSON file containing model predictions and references, computes ONLY the
CheXbert F1 for a SINGLE specified condition, and saves the results.

Two output modes (selectable via CLI flag):
1) Per-file CSV (legacy mode):
   - If the input path contains 'outputs', replace it with 'results' and keep the subpath
     after 'outputs', changing extension to .csv.
   - Example:
       .../outputs/baseline/mimic-cxr-jpg/foo.json
       -> results/baseline/mimic-cxr-jpg/foo.csv

2) Per-experiment+dataset CSV (aggregated mode):
   - Create/update one CSV file per (experiment, dataset), where:
       experiment = folder right after 'outputs'
       dataset    = folder right after experiment
   - Example:
       .../outputs/baseline/mimic-cxr-jpg/foo.json
       -> results/baseline/mimic-cxr-jpg/results.csv
     Each model/run becomes a row. The row index is the JSON filename stem (e.g., "foo").

Bootstrap:
- If --bootstrap-ci is enabled, the script adds median/ci_l/ci_h for the F1 (bootstrap percentile 95%).
- If --bootstrap-ci is disabled, it saves a single scalar score.

Expected JSON format:
[
  {"prediction": "...", "reference": "..."},
  ...
]

Dependencies:
- rrg_eval must be importable (for CheXbert_CONDITIONS and map_to_binary).
- HF download for StanfordAIMI/RRG_scorers chexbert.pth (unless cached).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from pathlib import Path
import argparse
import json
import random
import re
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertModel, AutoModel, BertTokenizer
from huggingface_hub import hf_hub_download

from rrg_eval.factuality_utils import CheXbert_CONDITIONS, map_to_binary

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(3)
np.random.seed(3)
torch.manual_seed(3)


# -----------------------------
# CheXbert core (minimal inference)
# -----------------------------
class UnlabeledDataset(torch.utils.data.Dataset):
    """Dataset containing report strings without labels."""

    def __init__(self, reports: List[str], tokenizer: BertTokenizer):
        self.encoded_imp = self._tokenize(reports, tokenizer)

    @staticmethod
    def _tokenize(reports: List[str], tokenizer: BertTokenizer) -> List[List[int]]:
        rets: List[List[int]] = []
        for r in reports:
            r = (r or "").strip().replace("\n", " ")
            r = re.sub(r"\s+", " ", r)
            tokenized_r = tokenizer.tokenize(r)
            if tokenized_r:
                res = tokenizer.encode_plus(tokenized_r)["input_ids"]
                if len(res) > 512:
                    res = res[:511] + [tokenizer.sep_token_id]
                rets.append(res)
            else:
                rets.append([tokenizer.cls_token_id, tokenizer.sep_token_id])
        return rets

    def __len__(self) -> int:
        return len(self.encoded_imp)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        imp = torch.LongTensor(self.encoded_imp[idx])
        return {"imp": imp, "len": int(imp.shape[0])}


def collate_fn_no_labels(sample_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_list = [s["imp"] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=0
    )
    len_list = [s["len"] for s in sample_list]
    return {"imp": batched_imp, "len": len_list}


def load_unlabeled_data(
    reports: List[str],
    tokenizer: BertTokenizer,
    batch_size: int = 128,
    num_workers: int = 4,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    dset = UnlabeledDataset(reports, tokenizer)
    return torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_no_labels,
    )


class bert_labeler(torch.nn.Module):
    """BERT backbone + 14 heads (CheXbert)."""

    def __init__(
        self,
        p: float = 0.1,
        clinical: bool = False,
        freeze_embeddings: bool = False,
        pretrain_path: Optional[str] = None,
    ):
        super().__init__()
        if pretrain_path is not None:
            self.bert = BertModel.from_pretrained(pretrain_path)
        elif clinical:
            self.bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        self.dropout = torch.nn.Dropout(p)
        hidden_size = self.bert.pooler.dense.in_features

        self.linear_heads = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, 4, bias=True) for _ in range(13)]
        )
        self.linear_heads.append(torch.nn.Linear(hidden_size, 2, bias=True))

    def forward(self, source_padded: torch.LongTensor, attention_mask: torch.Tensor):
        final_hidden = self.bert(source_padded, attention_mask=attention_mask)[0]  # [B,T,H]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)  # [B,H]
        cls_hidden = self.dropout(cls_hidden)
        return [head(cls_hidden) for head in self.linear_heads]


class CheXbert(torch.nn.Module):
    """Frozen CheXbert inference wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = bert_labeler()

        ckpt_path = hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
        )
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

        new_state_dict = OrderedDict()
        for k, v in checkpoint["model_state_dict"].items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

        self.model = model.to(self.device).eval()
        for _, p in self.model.named_parameters():
            p.requires_grad = False

    def generate_attention_masks(self, batch: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        masks = torch.ones(batch.size(0), batch.size(1), dtype=torch.float)
        for idx, src_len in enumerate(source_lengths):
            masks[idx, src_len:] = 0
        return masks.to(self.device)

    @torch.no_grad()
    def forward(self, reports: List[str]) -> List[List[int]]:
        """
        Returns:
          outputs: list length 14; each element is list[int] (argmax class) for each report
        """
        dataloader = load_unlabeled_data(
            reports, self.tokenizer, batch_size=128, num_workers=4
        )
        y_pred: List[List[torch.Tensor]] = [[] for _ in range(len(CheXbert_CONDITIONS))]

        for data in tqdm(dataloader, disable=True):
            batch = data["imp"].to(self.device)
            src_len = data["len"]
            attn_mask = self.generate_attention_masks(batch, src_len)
            out = self.model(batch, attn_mask)

            for j in range(len(out)):
                y_pred[j].append(out[j].argmax(dim=1))

        merged = [torch.cat(chunks, dim=0).tolist() for chunks in y_pred]
        return merged


# -----------------------------
# Single-condition metric
# -----------------------------
def compute_chexbert_f1_single_condition(
    preds: List[str],
    refs: List[str],
    condition: str,
    uncertain_mode: str = "rrg-",
    bootstrap_ci: bool = False,
    n_resamples: int = 500,
    seed: int = 3,
) -> Dict[str, Any]:
    """
    Compute ONLY binary F1 for a single CheXbert condition.
    Returns:
      - bootstrap_ci=False: {"CheXbert_F1_<cond>": f1}
      - bootstrap_ci=True : {"CheXbert_F1_<cond>": {"median":..., "ci_l":..., "ci_h":...}}
    """
    if len(preds) != len(refs):
        raise ValueError(f"Length mismatch: preds={len(preds)} vs refs={len(refs)}")

    if condition not in CheXbert_CONDITIONS:
        raise ValueError(
            f"Condition '{condition}' not found. Available:\n{list(CheXbert_CONDITIONS)}"
        )

    cond_idx = list(CheXbert_CONDITIONS).index(condition)

    model = CheXbert()
    outputs = model(preds + refs)  # list length 14, each list length Ntot

    cond_outputs = outputs[cond_idx]  # multiclass predictions (0..3) for this condition

    # map multiclass -> binary presence/absence, with uncertain policy
    if uncertain_mode == "rrg+":
        binary_all = [map_to_binary(x, "rrg+") for x in cond_outputs]
    elif uncertain_mode == "rrg-":
        binary_all = [map_to_binary(x) for x in cond_outputs]
    else:
        raise ValueError("uncertain_mode must be 'rrg-' or 'rrg+'")

    y_pred = binary_all[: len(preds)]
    y_true = binary_all[len(preds) :]

    def f1_of(indices: List[int]) -> float:
        _yt = [y_true[i] for i in indices]
        _yp = [y_pred[i] for i in indices]
        return float(
            precision_recall_fscore_support(
                _yt, _yp, average="binary", zero_division=0
            )[2]
        )

    metric_name = f"CheXbert_F1_{condition.replace(' ', '_')}"

    if not bootstrap_ci:
        return {metric_name: f1_of(list(range(len(y_true))))}

    rng = random.Random(seed)
    idxs = list(range(len(y_true)))
    samples: List[float] = []
    for _ in tqdm(range(n_resamples), desc=f"bootstrap {metric_name} 95% CI", disable=False):
        sample = [rng.choice(idxs) for _ in idxs]
        samples.append(f1_of(sample))

    ci_l, med, ci_h = np.percentile(samples, [2.5, 50, 97.5])
    return {
        metric_name: {"median": float(med), "ci_l": float(ci_l), "ci_h": float(ci_h)}
    }


# -----------------------------
# Results table builders
# -----------------------------
def build_main_results_table(results: Dict[str, Any], bootstrap_ci: bool) -> pd.DataFrame:
    """
    Format as:
    - bootstrap_ci=True  -> index ['median','ci_l','ci_h'], columns [metric]
    - bootstrap_ci=False -> index ['score'], columns [metric]
    """
    if bootstrap_ci:
        # results[metric] is dict median/ci_l/ci_h
        metric = next(iter(results.keys()))
        df = pd.DataFrame.from_dict({metric: results[metric]})
        return df[[metric]]

    metric = next(iter(results.keys()))
    df = pd.DataFrame([{metric: float(results[metric])}], index=["score"])
    return df


def main_results_to_single_row(main_results: pd.DataFrame, bootstrap_ci: bool) -> Dict[str, float]:
    """
    Flatten:
    - bootstrap_ci=False -> {metric: value}
    - bootstrap_ci=True  -> {metric_median:..., metric_ci_l:..., metric_ci_h:...}
    """
    row: Dict[str, float] = {}
    if not bootstrap_ci:
        idx = main_results.index[0]
        for col in main_results.columns:
            row[col] = float(main_results.loc[idx, col])
        return row

    for stat in main_results.index:
        for col in main_results.columns:
            row[f"{col}_{stat}"] = float(main_results.loc[stat, col])
    return row


# -----------------------------
# Output path derivation (same logic as your script)
# -----------------------------
def derive_per_file_output_paths(input_path: str) -> Dict[str, Path]:
    p = Path(input_path).resolve()
    parts = list(p.parts)

    if "outputs" in parts:
        idx = parts.index("outputs")
        rel_after_outputs = Path(*parts[idx + 1 :])
        base_no_ext = rel_after_outputs.with_suffix("")
        main_csv = Path("results") / base_no_ext
    else:
        main_csv = Path("results") / p.with_suffix("").name

    main_csv = main_csv.with_suffix(".csv")
    return {"main": main_csv}


def derive_experiment_dataset_results_csv(input_path: str) -> Path:
    p = Path(input_path).resolve()
    parts = list(p.parts)

    experiment = "unknown_experiment"
    dataset = "unknown_dataset"

    if "outputs" in parts:
        idx = parts.index("outputs")
        if idx + 1 < len(parts):
            experiment = parts[idx + 1]
        if idx + 2 < len(parts):
            dataset = parts[idx + 2]

    return Path("results") / experiment / dataset / "results.csv"


def upsert_aggregated_csv(csv_path: Path, model_name: str, row_dict: Dict[str, float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame()

    for k in row_dict.keys():
        if k not in df.columns:
            df[k] = np.nan

    df.loc[model_name, list(row_dict.keys())] = list(row_dict.values())

    df = df.sort_index()
    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(csv_path)


# -----------------------------
# Main runner
# -----------------------------
def run(
    filepath: str,
    condition: str,
    uncertain_mode: str = "rrg-",
    bootstrap_ci: bool = False,
    n_resamples: int = 500,
    output_mode: str = "per-file",  # "per-file" or "per-experiment"
    seed: int = 3,
) -> None:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    preds, refs = [], []
    for item in data:
        preds.append(item.get("prediction", "") or "")
        refs.append(item.get("reference", "") or "")

    results = compute_chexbert_f1_single_condition(
        preds=preds,
        refs=refs,
        condition=condition,
        uncertain_mode=uncertain_mode,
        bootstrap_ci=bootstrap_ci,
        n_resamples=n_resamples,
        seed=seed,
    )

    print("\n")
    print(f"Total reports: {len(preds)}\n")
    print("========== CheXbert Single-Condition Result ==========")
    main_results = build_main_results_table(results, bootstrap_ci=bootstrap_ci)
    print(main_results)
    print("")

    model_name = Path(filepath).with_suffix("").name

    if output_mode == "per-file":
        out_paths = derive_per_file_output_paths(filepath)
        out_paths["main"].parent.mkdir(parents=True, exist_ok=True)
        main_results.to_csv(out_paths["main"], index=True)
        print(f"Saved per-file results to: {out_paths['main']}")

    elif output_mode == "per-experiment":
        csv_path = derive_experiment_dataset_results_csv(filepath)
        row_dict = main_results_to_single_row(main_results, bootstrap_ci=bootstrap_ci)
        upsert_aggregated_csv(csv_path, model_name=model_name, row_dict=row_dict)
        print(f"Updated aggregated CSV: {csv_path} (row='{model_name}')")

    else:
        raise ValueError("Invalid output_mode. Use 'per-file' or 'per-experiment'.")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate report generation outputs with CheXbert single-class F1 only."
    )
    parser.add_argument("--filepath", type=str, 
    required=True, help="Path to the JSON predictions file.")
    parser.add_argument(
        "--condition",
        type=str,
        default='Atelectasis',
        #required=True,
        help="CheXbert condition name (exact string from CheXbert_CONDITIONS).",
    )
    parser.add_argument(
        "--uncertain-mode",
        type=str,
        default="rrg-",
        choices=["rrg-", "rrg+"],
        help="How to treat uncertain labels when mapping to binary.",
    )
    parser.add_argument("--bootstrap-ci", action="store_true", help="Enable bootstrap confidence intervals for F1.")
    parser.add_argument("--n-resamples", type=int, default=500, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=3, help="Random seed for bootstrap.")
    parser.add_argument(
        "--output-mode",
        type=str,
        default="per-file",
        choices=["per-file", "per-experiment"],
        help="Output mode: 'per-file' saves one CSV per JSON; 'per-experiment' appends a row to results/<experiment>/<dataset>/results.csv.",
    )
    parser.add_argument(
        "--print-conditions",
        action="store_true",
        help="Print available CheXbert_CONDITIONS and exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.print_conditions:
        print("Available CheXbert_CONDITIONS:")
        for c in CheXbert_CONDITIONS:
            print(f"- {c}")
        raise SystemExit(0)

    run(
        filepath=args.filepath,
        condition=args.condition,
        uncertain_mode=args.uncertain_mode,
        bootstrap_ci=args.bootstrap_ci,
        n_resamples=args.n_resamples,
        output_mode=args.output_mode,
        seed=args.seed,
    )
