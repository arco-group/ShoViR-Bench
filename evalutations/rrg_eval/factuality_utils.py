import pandas as pd
import numpy as np

#CONDITIONS is a list of all 14 medical observations 
CONDITIONS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]

CHEXBERT_CONDITIONS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices", "No Finding"
]

CONDITIONS_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]

CheXbert_CONDITIONS = [
    'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices', 'No Finding'
]

CLASS_MAPPING = {0: "Blank", 1: "Positive", 2: "Negative", 3: "Uncertain"}


POSITIVE = 1
NEGATIVE = 2
UNCERTAIN = 3
BLANK = 0

name2class = {"Blank": 0, "Positive": 1, "Negative": 2, "Uncertain": 3}


def map_to_binary(l, mode: str = "rrg"):
    if mode == "rrg":
        return {BLANK: 0, NEGATIVE: 0, UNCERTAIN: 0, POSITIVE: 1}[l]
    elif mode == "rrg+":
        return {BLANK: 0, NEGATIVE: 0, UNCERTAIN: 1, POSITIVE: 1}[l]
    elif mode == "classification":
        return {BLANK: 0, NEGATIVE: 0, UNCERTAIN: 0, POSITIVE: 1}[l]


def load_mimic_cxr_expert_annotations(annotations: str = "data/mimic_cxr_test_set_annotated.xlsx") -> pd.DataFrame:
    df = pd.read_excel(annotations, index_col=0)
    df.rename(columns={
        "Airspace Opacity_ann1": "Lung Opacity_ann1",
        "Airspace Opacity_ann2": "Lung Opacity_ann2"
    }, inplace=True)
    df.replace(0, NEGATIVE, inplace=True)
    df.replace(-1, UNCERTAIN, inplace=True)
    df.fillna(BLANK, inplace=True)
    return df