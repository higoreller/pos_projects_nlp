from nltk.corpus import reuters
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from nltk.metrics import BigramAssocMeasures
from high_information_words import high_information_words, bag_of_words_in_set
from naive_bayes import bag_of_words
import collections
from nltk.metrics import precision, recall, masi_distance
from nltk.classify import MultiClassifierI

print(len(reuters.categories()))


def reuters_high_info_words(score_fn=BigramAssocMeasures.chi_sq):
    labeled_words = []

    for label in reuters.categories():
        labeled_words.append((label, reuters.words(categories=[label])))

    return high_information_words(labeled_words, score_fn=score_fn)


def reuters_train_test_feats(feature_detector=bag_of_words):
    train_feats = []
    test_feats = []
    for fileid in reuters.fileids():
        if fileid.startswith('training'):
            featlist = train_feats
        else:  # fileid.startswith('test')
            featlist = test_feats
        feats = feature_detector(reuters.words(fileid))
        labels = reuters.categories(fileid)
        featlist.append((feats, labels))
    return train_feats, test_feats


rwords = reuters_high_info_words()
def featdet(words): return bag_of_words_in_set(words, rwords)


multi_train_feats, multi_test_feats = reuters_train_test_feats(featdet)


def train_binary_classifiers(trainf, labelled_feats, labelset):
    pos_feats = collections.defaultdict(list)
    neg_feats = collections.defaultdict(list)
    classifiers = {}

    for feat, labels in labelled_feats:
        for label in labels:
            pos_feats[label].append(feat)

        for label in labelset - set(labels):
            neg_feats[label].append(feat)

    for label in labelset:
        postrain = [(feat, label) for feat in pos_feats[label]]
        negtrain = [(feat, '!%s' % label) for feat in neg_feats[label]]
        classifiers[label] = trainf(postrain + negtrain)

    return classifiers


def trainf(train_feats): return SklearnClassifier(
    LogisticRegression()).train(train_feats)


labelset = set(reuters.categories())
classifiers = train_binary_classifiers(trainf, multi_train_feats, labelset)
len(classifiers)


class MultiBinaryClassifier(MultiClassifierI):

    def __init__(self, *label_classifiers):
        self._label_classifiers = dict(label_classifiers)
        self._labels = sorted(self._label_classifiers.keys())

    def labels(self):
        return self._labels

    def classify(self, feats):
        lbls = set()
        for label, classifier in self._label_classifiers.items():
            if classifier.classify(feats) == label:
                lbls.add(label)

        return lbls


multi_classifier = MultiBinaryClassifier(*classifiers.items())


def multi_metrics(multi_classifier, test_feats):
    mds = []
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feat, labels) in enumerate(test_feats):
        for label in labels:
            refsets[label].add(i)

        guessed = multi_classifier.classify(feat)

        for label in guessed:
            testsets[label].add(i)

        mds.append(masi_distance(set(labels), guessed))

    avg_md = sum(mds) / float(len(mds))
    precisions = {}
    recalls = {}

    for label in multi_classifier.labels():
        precisions[label] = precision(refsets[label], testsets[label])
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls, avg_md


multi_precisions, multi_recalls, avg_md = multi_metrics(
    multi_classifier, multi_test_feats)
print(avg_md)

print(multi_precisions['soybean'])
print(multi_recalls['soybean'])
print(multi_precisions['sunseed'])
print(multi_recalls['sunseed'])
print(len(reuters.fileids(categories=['crude'])))
