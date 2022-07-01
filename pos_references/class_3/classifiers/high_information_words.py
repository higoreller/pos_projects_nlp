from nltk.metrics import BigramAssocMeasures
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
from nltk.probability import FreqDist, ConditionalFreqDist
import collections
from naive_bayes import bag_of_words, label_feats_from_corpus
from naive_bayes import split_label_feats
from precision_recall import precision_recall
from nltk.corpus import movie_reviews


def high_information_words(labelled_words, score_fn=BigramAssocMeasures.chi_sq,
                           min_score=5):
    word_fd = FreqDist()  # Calcula a frequência de cada palavra
    label_word_fd = ConditionalFreqDist()  # Calcula a frequência condicional

    for label, words in labelled_words:
        for word in words:
            word_fd[word] += 1
            label_word_fd[label][word] += 1

    n_xx = label_word_fd.N()
    high_info_words = set()

    for label in label_word_fd.conditions():
        n_xi = label_word_fd[label].N()
        word_scores = collections.defaultdict(int)

        for word, n_ii in label_word_fd[label].items():
            n_ix = word_fd[word]
            score = score_fn(n_ii, (n_ix, n_xi), n_xx)
            word_scores[word] = score

    bestwords = [word for word, score in word_scores.items()
                 if score >= min_score]
    high_info_words |= set(bestwords)
    return high_info_words


def bag_of_words_in_set(words, goodwords):
    return bag_of_words(set(words) & set(goodwords))


labels = movie_reviews.categories()
labeled_words = [(lab, movie_reviews.words(categories=[lab]))
                 for lab in labels]
high_info_words = set(high_information_words(labeled_words))
def feat_det(words): return bag_of_words_in_set(words, high_info_words)


lfeats = label_feats_from_corpus(movie_reviews, feature_detector=feat_det)
train_feats, test_feats = split_label_feats(lfeats)

nb_classifier = NaiveBayesClassifier.train(train_feats)
print(accuracy(nb_classifier, test_feats))  # 0.9

nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)
print(nb_precisions['pos'])  # Antes: 0,65 / Depois: 0,88
print(nb_precisions['neg'])  # Antes: 0,95 / Depois: 0,92
print(nb_recalls['pos'])  # Antes: 0,98 / Depois: 0,924
print(nb_recalls['neg'])  # Antes: 0,476 / Depois: 0,876

me_classifier = MaxentClassifier.train(
    train_feats, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
print(accuracy(me_classifier, test_feats))  # 0.91

me_precisions, me_recalls = precision_recall(me_classifier, test_feats)
print(me_precisions['pos'])  # 0.65 / 0.89
print(me_precisions['neg'])  # 0.97 / 0.93
print(me_recalls['pos'])  # 0.98 / 0.93
print(me_recalls['neg'])  # 0.46 / 0.89

dt_classifier = DecisionTreeClassifier.train(
    train_feats, binary=True, depth_cutoff=20, support_cutoff=20,
    entropy_cutoff=0.01)
accuracy(dt_classifier, test_feats)

dt_precisions, dt_recalls = precision_recall(
    dt_classifier, test_feats)  # Manteve os valores de precisão e recall
dt_precisions['pos']
dt_precisions['neg']
dt_recalls['pos']
dt_recalls['neg']

"""A árvore provavelmente já coloca os valores mais significantes em primeiro
lugar. A precisão só aumentaria se aumentasse muito a profundidade da árvore,
o que geraria um tempo de processamento absurdamente mais alto."""

# ATENÇÃO:
# Para o classificador sklearn temos o seguinte:

"""Sua precisão antes era de 86,4%, então tivemos uma pequena diminuição.
Em geral, a máquina de vetor de suporte e os algoritmos baseados em regressão
logística se beneficiarão menos, ou talvez até sejam prejudicados, pela
pré-filtragem dos recursos de treinamento. Isso ocorre porque esses algoritmos
são capazes de aprender pesos de recursos que correspondem à significância de
cada recurso, enquanto os algoritmos Naive Bayes não."""
