import itertools
from nltk.classify import ClassifierI
from nltk.probability import FreqDist
from naive_bayes import nb_classifier, test_feats, train_feats
from precision_recall import precision_recall
from maxent import me_classifier
from decision_tree import dt_classifier
from nltk.classify.util import accuracy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression


class MaxVoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        self._labels = sorted(
            set(itertools.chain(*[c.labels() for c in classifiers])))

    def labels(self):
        return self._labels

    def classify(self, feats):
        counts = FreqDist()

        for classifier in self._classifiers:
            counts[classifier.classify(feats)] += 1

        return counts.max()


sk_classifier = SklearnClassifier(LogisticRegression())
sk_classifier.train(train_feats)

mv_classifier = MaxVoteClassifier(
    nb_classifier, dt_classifier, me_classifier, sk_classifier)
print("MaxVoteClassifier")
print(mv_classifier.labels())
print(accuracy(mv_classifier, test_feats))  # 0.736
mv_precisions, mv_recalls = precision_recall(mv_classifier, test_feats)
print(mv_precisions['pos'])  # 0.66
print(mv_precisions['neg'])  # 0.96
print(mv_recalls['pos'])  # 0.98
print(mv_recalls['neg'])  # 0.492
