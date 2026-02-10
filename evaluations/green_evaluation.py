import os
import sys
import json
sys.path.insert(0, "GREEN")
from green_score import GREEN


# Specifica il path del file JSON
file_path = "/mimer/NOBACKUP/groups/naiss2023-6-336/msalme/Shortcut-Learning-RRG/outputs/baseline/mimic-cxr-jpg/google__medgemma-1.5-4b-it_medgemma_default.json"

# Leggi il file JSON
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Crea le liste refs e hyps
refs = [item["reference"] for item in data]
hyps = [item["prediction"] for item in data]


model_name = "StanfordAIMI/GREEN-radllama2-7b"

green_scorer = GREEN(model_name, output_dir=".")
mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
print(green_score_list)
print(summary)