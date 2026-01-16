""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt
from rouge_score import rouge_scorer, scoring


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return self.tokenizer_func(text)


def compute(
    predictions,
    references,
    rouge_types=None,
    use_aggregator=True,
    use_stemmer=False,
    tokenizer=None,
    confidence_interval=0.95,
    n_samples=500
):
    if rouge_types is None:
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    multi_ref = isinstance(references[0], list)

    if tokenizer is not None:
        tokenizer = Tokenizer(tokenizer)

    scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, use_stemmer=use_stemmer)
    if use_aggregator:
        aggregator = scoring.BootstrapAggregator(
            confidence_interval=confidence_interval, n_samples=n_samples
        )
    else:
        scores = []

    for ref, pred in zip(references, predictions):
        if multi_ref:
            score = scorer.score_multi(ref, pred)
        else:
            score = scorer.score(ref, pred)
        if use_aggregator:
            aggregator.add_scores(score)
        else:
            scores.append(score)

    if use_aggregator:
        result = aggregator.aggregate()
        for key in result:
            result[key] = {
                'median': result[key].mid.fmeasure,
                'ci_l': result[key].low.fmeasure,
                'ci_h': result[key].high.fmeasure,
            }

    else:
        result = {}
        for key in scores[0]:
            result[key] = list(score[key].fmeasure for score in scores)

    return result
