from green_score import GREEN
import os
import json

# Specifica il path del file JSON
file_path = "/Users/marcosalme/Desktop/ShortCutRRG/Shortcut-Learning-RRG/evaluations/predictions_rrg_eval.json"

# Leggi il file JSON
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Crea le liste refs e hyps
refs = [item["references"] for item in data]
hyps = [item["predictions"] for item in data]


model_name = "StanfordAIMI/GREEN-radllama2-7b"

green_scorer = GREEN(model_name, output_dir=".")
mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
print(green_score_list)
print(summary)
# for index, row in result_df.iterrows():
#     print(f"Row {index}:\n")
#     for col_name in result_df.columns:
#         print(f"{col_name}: {row[col_name]}\n")
#     print('-' * 80)