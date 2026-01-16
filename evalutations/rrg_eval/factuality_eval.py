import copy
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from statsmodels.stats.inter_rater import cohens_kappa

from rrg_eval.factuality_utils import (
    load_mimic_cxr_expert_annotations,
    map_to_binary,
    CONDITIONS, NEGATIVE, UNCERTAIN, POSITIVE
)


def get_weighted_f1_weights(
        path_or_df="data/mimic_cxr_test_set_annotated.xlsx",
        annotators=["ann1", "ann2"],
        mask=None,
    ):
    """Compute weights used to obtain the weighted average of
       mention, negation and uncertain f1 scores. 
    @param path_or_df: A path to the excel file or a dataframe
    @annotators: List of annotator ids
    @mask: List of 14 binary tensor each of shape (dev_set_size) 

    @return weight_dict (dictionary): maps conditions to a list of weights, the order
                                      in the lists is negation, uncertain, positive 
    """
    if isinstance(path_or_df, str):
        df = load_mimic_cxr_expert_annotations(path_or_df)
    else:
        df = path_or_df

    weight_dict = {}
    for i, cond in enumerate(CONDITIONS):
        weights = [0, 0, 0]
        for ann in annotators:
            col = df[f"{cond}_{ann}"]

            col_mask = (col == NEGATIVE).to_numpy()
            col_mask = col_mask * mask[i] if mask is not None else col_mask
            weights[0] += col_mask.sum()

            col_mask = (col == UNCERTAIN).to_numpy()
            col_mask = col_mask * mask[i] if mask is not None else col_mask
            weights[1] += col_mask.sum()

            col_mask = (col == POSITIVE).to_numpy()
            col_mask = col_mask * mask[i] if mask is not None else col_mask
            weights[2] += col_mask.sum()

        if np.sum(weights) > 0:
            weights = np.array(weights)/np.sum(weights)
        weight_dict[cond] = weights
    return weight_dict


def weighted_avg(scores, weights):
    """Compute weighted average of scores
    @param scores(List): the task scores
    @param weights (List): corresponding normalized weights

    @return (float): the weighted average of task scores
    """
    return np.sum(np.array(scores) * np.array(weights))


def compute_mention_f1(y_true, y_pred, mask=None):
    """Compute the mention F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
    @mask: List of 14 binary tensor each of shape (dev_set_size) 

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == NEGATIVE] = 1
        y_true[j][y_true[j] == UNCERTAIN] = 1
        y_pred[j][y_pred[j] == NEGATIVE] = 1
        y_pred[j][y_pred[j] == UNCERTAIN] = 1

    res = []
    for j in range(len(y_true)): 
        if mask is not None:
            y_true[j] = [y_t for y_t, m in zip(y_true[j], mask[j]) if m == 1]
            y_pred[j] = [y_p for y_p, m in zip(y_pred[j], mask[j]) if m == 1]

        res.append(f1_score(y_true[j], y_pred[j], pos_label=1))
        
    return res


def compute_blank_f1(y_true, y_pred, mask=None):
    """Compute the blank F1 score 
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
    @mask: List of 14 binary tensor each of shape (dev_set_size) 
                                                                         
    @returns res (list): List of 14 scalars                           
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == NEGATIVE] = 1
        y_true[j][y_true[j] == UNCERTAIN] = 1
        y_pred[j][y_pred[j] == NEGATIVE] = 1
        y_pred[j][y_pred[j] == UNCERTAIN] = 1

    res = []
    for j in range(len(y_true)):
        if mask is not None:
            y_true[j] = [y_t for y_t, m in zip(y_true[j], mask[j]) if m == 1]
            y_pred[j] = [y_p for y_p, m in zip(y_pred[j], mask[j]) if m == 1]

        res.append(f1_score(y_true[j], y_pred[j], pos_label=0))

    return res

        
def compute_negation_f1(y_true, y_pred, mask=None):
    """Compute the negation F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions   
    @mask: List of 14 binary tensor each of shape (dev_set_size) 

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == UNCERTAIN] = 0
        y_true[j][y_true[j] == POSITIVE] = 0
        y_pred[j][y_pred[j] == UNCERTAIN] = 0
        y_pred[j][y_pred[j] == POSITIVE] = 0

    res = []
    for j in range(len(y_true)-1):
        if mask is not None:
            y_true[j] = [y_t for y_t, m in zip(y_true[j], mask[j]) if m == 1]
            y_pred[j] = [y_p for y_p, m in zip(y_pred[j], mask[j]) if m == 1]

        res.append(f1_score(y_true[j], y_pred[j], pos_label=2))

    res.append(0) #No Finding gets score of zero
    return res


def compute_positive_f1(y_true, y_pred, mask=None):
    """Compute the positive F1 score
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions 
    @mask: List of 14 binary tensor each of shape (dev_set_size) 

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == UNCERTAIN] = 0
        y_true[j][y_true[j] == NEGATIVE] = 0
        y_pred[j][y_pred[j] == UNCERTAIN] = 0
        y_pred[j][y_pred[j] == NEGATIVE] = 0

    res = []
    for j in range(len(y_true)):
        if mask is not None:
            y_true[j] = [y_t for y_t, m in zip(y_true[j], mask[j]) if m == 1]
            y_pred[j] = [y_p for y_p, m in zip(y_pred[j], mask[j]) if m == 1]

        res.append(f1_score(y_true[j], y_pred[j], pos_label=1))

    return res

        
def compute_uncertain_f1(y_true, y_pred, mask=None):
    """Compute the negation F1 score as in CheXpert paper
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
    @mask: List of 14 binary tensor each of shape (dev_set_size) 

    @returns res (list): List of 14 scalars
    """
    for j in range(len(y_true)):
        y_true[j][y_true[j] == NEGATIVE] = 0
        y_true[j][y_true[j] == POSITIVE] = 0
        y_pred[j][y_pred[j] == NEGATIVE] = 0
        y_pred[j][y_pred[j] == POSITIVE] = 0

    res = []
    for j in range(len(y_true)-1):
        if mask is not None:
            y_true[j] = [y_t for y_t, m in zip(y_true[j], mask[j]) if m == 1]
            y_pred[j] = [y_p for y_p, m in zip(y_pred[j], mask[j]) if m == 1]

        res.append(f1_score(y_true[j], y_pred[j], pos_label=3))

    res.append(0) #No Finding gets a score of zero
    return res


def evaluate(y_true, y_pred, f1_weights, mask=None):
    """ Function to evaluate the current model weights
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
    @param f1_weights (dictionary): dictionary mapping conditions to f1
                                    task weights
    @param mask (list): List of 14 tensors each of shape (dev_set_size)

    @returns res_dict (dictionary): dictionary with keys 'blank', 'mention', 'negation',
                           'uncertain', 'positive' and 'weighted', with values 
                            being lists of length 14 with each element in the 
                            lists as a scalar. If return_pred is true then a 
                            tuple is returned with the aforementioned dictionary 
                            as the first item, a list of predictions as the 
                            second item, and a list of ground truth as the 
                            third item
    """
    mention_f1 = compute_mention_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred), mask)
    negation_f1 = compute_negation_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred), mask)
    uncertain_f1 = compute_uncertain_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred), mask)
    positive_f1 = compute_positive_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred), mask)
    blank_f1 = compute_blank_f1(copy.deepcopy(y_true), copy.deepcopy(y_pred), mask)
    
    weighted = []
    kappas = []
    for j in range(len(y_pred)):
        cond = CONDITIONS[j]
        avg = weighted_avg([negation_f1[j], uncertain_f1[j], positive_f1[j]], f1_weights[cond])
        weighted.append(avg)

        if mask is not None:
            y_true[j] = [y_t for y_t, m in zip(y_true[j], mask[j]) if m == 1]
            y_pred[j] = [y_p for y_p, m in zip(y_pred[j], mask[j]) if m == 1]

        mat = confusion_matrix(y_true[j], y_pred[j])
        kappas.append(cohens_kappa(mat, return_results=False))

    res_dict = {'mention': mention_f1,
                'blank': blank_f1,
                'negation': negation_f1,
                'uncertain': uncertain_f1,
                'positive': positive_f1,
                'weighted': weighted,
                'kappa': kappas}
    
    return res_dict


def test(y_true, y_pred, f1_weights=None, mask=None):
    """Evaluate model on test set. 
    @param y_true (list): List of 14 tensors each of shape (dev_set_size)
    @param y_pred (list): Same as y_true but for model predictions
    @param f1_weights (dictionary): maps conditions to f1 task weights
    @param mask (list): List of 14 binary tensors each of shape (dev_set_size)
    """

    y_true = [np.array(y_i) for y_i in y_true]
    y_pred = [np.array(y_i) for y_i in y_pred]
    if f1_weights is None:
        f1_weights = get_weighted_f1_weights(mask=mask)
    ret = {cond: {} for cond in CONDITIONS + ["avg"]}

    metrics = evaluate(y_true=y_true, y_pred=y_pred, f1_weights=f1_weights, mask=mask)
    weighted = metrics['weighted']
    kappas = metrics['kappa']

    for j in range(len(CONDITIONS)):
        ret[CONDITIONS[j]]["kappas"] = kappas[j]
    ret["avg"]["kappas"] = np.mean(kappas)

    for j in range(len(CONDITIONS)):
        ret[CONDITIONS[j]]["weighted f1"] = weighted[j]
    ret["avg"]["weighted f1"] = np.mean(weighted)
    
    for j in range(len(CONDITIONS)):
        ret[CONDITIONS[j]]["blank f1"] = metrics["blank"][j]
        ret[CONDITIONS[j]]["negation f1"] = metrics["negation"][j]
        ret[CONDITIONS[j]]["uncertain f1"] = metrics["uncertain"][j]
        ret[CONDITIONS[j]]["positive f1"] = metrics["positive"][j]
        ret[CONDITIONS[j]]["mention f1"] = metrics["mention"][j]

    ret["avg"]["blank f1"] = np.mean(metrics['blank'])
    ret["avg"]["negation f1"] = np.mean(metrics['negation'][:-1]) #No Finding has no negations
    ret["avg"]["uncertain f1"] = np.mean(metrics['uncertain'][:-2]) #No Finding, Support Devices have no uncertain labels in test set
    ret["avg"]["positive f1"] = np.mean(metrics['positive'])
    ret["avg"]["mention f1"]= np.mean(metrics['mention'])

    return ret


def generate_classification_report(y_true, y_pred, target_names):
    """
    @param y_true (list): A list of lists each containing true binary labelss for target names
    @param y_pred (list): Same as y_true but for model predictions
    """
    cr = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)

    return cr