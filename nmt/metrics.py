import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score


nltk.download('wordnet')


class Metric(object):
    """Base class for all metrics.
    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """
    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class MeteorScore(Metric):
    def __init__(self):
        self.counter = 0
        self._agg_meteor = 0

    def reset(self):
        self.counter = 0
        self._agg_meteor = 0

    def add(self, s1, s2):
        assert len(s1) == len(s2)
        for s1_, s2_ in zip(s1, s2):
            self._agg_meteor += single_meteor_score(s1_, s2_)
            self.counter += 1

    def value(self):
        return self._agg_meteor / self.counter


class BLEUScore(Metric):
    def __init__(self):
        self.counter = 0
        self._agg_bleu = 0

    def reset(self):
        self.counter = 0
        self._agg_bleu = 0

    def add(self, s1, s2):
        assert len(s1) == len(s2)
        for s1_, s2_ in zip(s1, s2):
            self._agg_bleu += sentence_bleu([s1_.split(' ')], [s2_.split(' ')])
            self.counter += 1

    def value(self):
        return self._agg_bleu / self.counter
