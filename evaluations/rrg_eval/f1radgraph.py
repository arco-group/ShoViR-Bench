"""
Custom f1 radgraph that can output precision and recall
"""

from radgraph import F1RadGraph
import numpy as np
from scipy.stats import bootstrap

import os
from radgraph.radgraph import CACHE_DIR
from huggingface_hub import hf_hub_download


class F1RadGraphv2(F1RadGraph):
    def __init__(
            self,
            reward_level,
            **kwargs
    ):

        self._download_radgraph()
        super().__init__(reward_level, **kwargs)
        assert reward_level in ["simple", "partial", "complete", "all"]

    def _download_radgraph(self):
        if not os.path.exists(os.path.join(CACHE_DIR, "radgraph.tar.gz")):
            os.makedirs(CACHE_DIR, exist_ok=True)
            hf_hub_download(
                repo_id="StanfordAIMI/RRG_scorers",
                filename="radgraph.tar.gz",
                revision="d97745aa136e5beb927da7e768e99de6ae807902",
                local_dir=CACHE_DIR,
            )

    def forward(self, refs, hyps):
        # Checks
        assert isinstance(hyps, str) or isinstance(hyps, list)
        assert isinstance(refs, str) or isinstance(refs, list)

        if isinstance(hyps, str):
            hyps = [hyps]
        if isinstance(hyps, str):
            refs = [refs]

        assert len(refs) == len(hyps)

        # getting empty report list
        number_of_reports = len(hyps)
        empty_report_index_list = [i for i in range(number_of_reports) if (len(hyps[i]) == 0) or (len(refs[i]) == 0)]
        number_of_non_empty_reports = number_of_reports - len(empty_report_index_list)

        # stacking all reports (hyps and refs)
        report_list = [
                          hypothesis_report
                          for i, hypothesis_report in enumerate(hyps)
                          if i not in empty_report_index_list
                      ] + [
                          reference_report
                          for i, reference_report in enumerate(refs)
                          if i not in empty_report_index_list
                      ]

        assert len(report_list) == 2 * number_of_non_empty_reports

        # getting annotations
        inference_dict = self.radgraph(report_list)

        # Compute reward
        reward_list = []
        hypothesis_annotation_lists = []
        reference_annotation_lists = []
        non_empty_report_index = 0
        for report_index in range(number_of_reports):
            if report_index in empty_report_index_list:
                reward_list.append((0., 0., 0.))
                
                continue

            hypothesis_annotation_list = inference_dict[str(non_empty_report_index)]
            reference_annotation_list = inference_dict[
                str(non_empty_report_index + number_of_non_empty_reports)
            ]

            reward_list.append(
                compute_reward(
                    hypothesis_annotation_list,
                    reference_annotation_list,
                    self.reward_level,
                )
            )
            reference_annotation_lists.append(reference_annotation_list)
            hypothesis_annotation_lists.append(hypothesis_annotation_list)
            non_empty_report_index += 1

        assert non_empty_report_index == number_of_non_empty_reports
        
        if self.reward_level == "all":
            reward_list = ([r[0] for r in reward_list], [r[1] for r in reward_list], [r[2] for r in reward_list])
            mean_reward = (np.mean(reward_list[0]), np.mean(reward_list[1]), np.mean(reward_list[2]))
        else:
            mean_reward =np.mean(reward_list, axis=0)
            mean_reward = {'f1-radgraph': mean_reward[0], 'precision-radgraph': mean_reward[1], 'recall-radgraph': mean_reward[2]}

        return (
            mean_reward,
            reward_list,
            hypothesis_annotation_lists,
            reference_annotation_lists,
        )


def compute_statistic(reward_list):
    def _compute(indices):
        r = [reward_list[i] for i in indices]
        return np.mean(r, axis=0)[0]

    return _compute


def bootstrap_confidence_interval(
        reward_list, n_resamples: int = 500, method: str = "percentile",
    ):
    bs = bootstrap(
        data=(list(range(len(reward_list))),),
        statistic=compute_statistic(reward_list),
        method=method,
        paired=False,
        vectorized=False,
        confidence_level=0.95,
        random_state=3,
        n_resamples=n_resamples
    )
    return bs


def exact_entity_token_if_all_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.extend([(entity["tokens"].lower(),
                                   entity["label"],
                                   r[0],
                                   annotation_list["entities"][r[1]]["tokens"].lower())
                                  for r in entity["relations"]]
                                 )

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates
    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall


def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            if not entity["relations"]:
                candidate.append((entity["tokens"], entity["label"]))
            if entity["relations"]:
                candidate.append((entity["tokens"], entity["label"], True))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )
    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall


def exact_entity_token_match_reward(
        hypothesis_annotation_list, reference_annotation_list
):
    candidates = []
    for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
        candidate = []
        for entity in annotation_list["entities"].values():
            candidate.append((entity["tokens"], entity["label"]))

        candidate = set(candidate)
        candidates.append(candidate)

    hypothesis_relation_token_list, reference_relation_token_list = candidates

    precision = (
        sum(
            [
                1
                for x in hypothesis_relation_token_list
                if (x in reference_relation_token_list)
            ]
        )
        / len(hypothesis_relation_token_list)
        if len(hypothesis_relation_token_list) > 0
        else 0.0
    )
    recall = (
        sum(
            [
                1
                for x in reference_relation_token_list
                if (x in hypothesis_relation_token_list)
            ]
        )
        / len(reference_relation_token_list)
        if len(reference_relation_token_list) > 0
        else 0.0
    )

    f1_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return f1_score, precision, recall


def compute_reward(
        hypothesis_annotation_list,
        reference_annotation_list,
        reward_level,
):
    assert reward_level in ["simple", "partial", "complete", "all"]
    if (
            len(hypothesis_annotation_list["entities"].keys()) == 0
            or len(reference_annotation_list["entities"].keys()) == 0
    ):
        return (0., 0., 0.)
    simple = exact_entity_token_match_reward(hypothesis_annotation_list, reference_annotation_list)
    partial = exact_entity_token_if_rel_exists_reward(hypothesis_annotation_list, reference_annotation_list)
    complete = exact_entity_token_if_all_match_reward(hypothesis_annotation_list, reference_annotation_list)
    all = (simple, partial, complete)
    
    return eval(reward_level)

if __name__ == '__main__':
    import json
    import time

    m = F1RadGraphv2(reward_level='partial')
    m_og = F1RadGraph(reward_level='partial')

    t = time.time()
    test_hyps = ['No pleural effusion. Normal heart size.',
              'Normal heart size.',
              'Increased mild pulmonary edema and left basal atelectasis.',
              'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
              'Elevated left hemidiaphragm and blunting of the left costophrenic angle although no definite evidence of pleural effusion seen on the lateral view.',
              ]
    test_refs = ['No pleural effusions.',
              'Enlarged heart.',
              'No evidence of pneumonia. Stable cardiomegaly.',
              'Bilateral lower lobe bronchiectasis with improved right lower medial lung peribronchial consolidation.',
              'No acute cardiopulmonary process. No significant interval change. Please note that peribronchovascular ground-glass opacities at the left greater than right lung bases seen on the prior chest CT of ___ were not appreciated on prior chest radiography on the same date and may still be present. Additionally, several pulmonary nodules measuring up to 3 mm are not not well appreciated on the current study-CT is more sensitive.'
              ]
    f1_radgraph = m(hyps=test_hyps, refs=test_refs)
    f1_radgraph_og = m_og(hyps=test_hyps, refs=test_refs)
    
    
    assert f1_radgraph[0]['f1-radgraph'] == f1_radgraph_og[0]