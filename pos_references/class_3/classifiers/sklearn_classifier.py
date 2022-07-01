from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from naive_bayes import train_feats, test_feats
from nltk.classify.util import accuracy
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.preprocessing import LabelEncoder
# from nltk import ClassifierI, DictionaryProbDist
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC

# Demonstração de como a classe SklearnClassifier funciona. A entrada é o
# algoritmo a ser utilizado, um encoder é criado e o processo de vetorização
# é realizado.

"""class SklearnClassifier(ClassifierI):
    def __init__(self, estimator, dtype=float, sparse=True):
        self._clf = estimator
        self._encoder = LabelEncoder()
        self._vectorizer = DictVectorizer(dtype=dtype, sparse=sparse)

    def batch_classify(self, featuresets):
        X = self._vectorizer.transform(featuresets)
        classes = self._encoder.classes_
        return [classes[i] for i in self._clf.predict(X)]

    def batch_prob_classify(self, featuresets):
        X = self._vectorizer.transform(featuresets)
        y_proba_list = self._clf.predict_proba(X)
        return [self._make_probdist(y_proba) for y_proba in y_proba_list]

    def labels(self):
        return list(self._encoder.classes_)

    def train(self, labeled_featuresets):
        X, y = list(compat.izip(*labeled_featuresets))
        X = self._vectorizer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._clf.fit(X, y)
        return self

    def _make_probdist(self, y_proba):
        classes = self._encoder.classes_
        return DictionaryProbDist(dict((classes[i], p) for i, p in
        enumerate(y_proba)))"""


# Utilizando o algoritmo MultinomialNB

sk_classifier = SklearnClassifier(MultinomialNB())
sk_classifier.train(train_feats)
accuracy(sk_classifier, test_feats)  # 0.83

# Utilizando o algoritmo BernoulliNB

sk_classifier = SklearnClassifier(BernoulliNB())
sk_classifier.train(train_feats)
accuracy(sk_classifier, test_feats)  # 0.812

# Utilizando um algoritmo de regressão logística

sk_classifier = SklearnClassifier(LogisticRegression())
"""<SklearnClassifier(LogisticRegression(C=1.0, class_weight=None, dual=False,
fit_intercept=True, intercept_scaling=1, penalty='l2', random_state=None,
tol=0.0001))>"""
sk_classifier.train(train_feats)
accuracy(sk_classifier, test_feats)  # 0.896

# Este algoritmo é similar ao algoritmo de máxima entropia do NLTK, entretanto
# utiliza-se de uma otimização em seu processamento, tendo uma velocidade de
# treino muito inferior, além de demonstrar uma acurácia bem maior.


sk_classifier = SklearnClassifier(SVC())
sk_classifier.train(train_feats)
"""<SklearnClassifier(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
degree=3, gamma=0.0, kernel='rbf', max_iter=-1, probability=False,
random_state=None, shrinking=True, tol=0.001, verbose=False))>"""
accuracy(sk_classifier,
         test_feats)  # Deu 0.864 também, mas supostamente deveria ser 0.69

sk_classifier = SklearnClassifier(LinearSVC())
sk_classifier.train(train_feats)
"""<SklearnClassifier(LinearSVC(C=1.0, class_weight=None, dual=True,
fit_intercept=True, intercept_scaling=1, loss='l2', multi_class='ovr',
penalty='l2', random_state=None, tol=0.0001, verbose=0))>"""
accuracy(sk_classifier, test_feats)  # 0.864

sk_classifier = SklearnClassifier(NuSVC())
sk_classifier.train(train_feats)
"""<SklearnClassifier(NuSVC(cache_size=200, coef0=0.0, degree=3, gamma=0.0,
kernel='rbf', max_iter=-1, nu=0.5, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False))>"""
accuracy(sk_classifier, test_feats)  # 0.882
