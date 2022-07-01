from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import collections
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier


# Label => Output
# Feature => Input. É uma propriedade dos dados que estão sendo treinados, ou
# também uma coluna dos dados de entrada quando representados por tabela.

words = ['cachorro', 'gato', 'pássaro', 'cobra', 'zebra']


def bag_of_words(words):
    return dict([(word, True) for word in words])


def bag_of_words_not_in_set(words, badwords):
    return bag_of_words(set(words) - set(badwords))


def bag_of_non_stopwords(words, stopfile='english'):
    badwords = stopwords.words(stopfile)
    return bag_of_words_not_in_set(words, badwords)

# A ocorrência de um bigrama é mais rara em um texto, sendo assim pode ajudar
# o classificador a tomar decisões mais precisas.


def bag_of_bigrams_words(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return bag_of_words(words + bigrams)


# Exemplo: reviews de filmes, com comentários de usuários sendo classificados
# como positivos e negativos.
# Estrutura do dicionário: {label: [featureset]}

def label_feats_from_corpus(corp, feature_detector=bag_of_words):
    # Cria um defaultdict como valor padrão para a lista
    label_feats = collections.defaultdict(list)

    for label in corp.categories():  # Retorna as categorias/rótulos do corpus
        # (pos,neg)

        for fileid in corp.fileids(categories=[label]):

            feats = feature_detector(corp.words(fileids=[fileid]))

            label_feats[label].append(feats)

    return label_feats


def split_label_feats(lfeats, split=0.75):
    train_feats = []
    test_feats = []
    for label, feats in lfeats.items():
        cutoff = int(len(feats) * split)
        train_feats.extend([(feat, label) for feat in feats[:cutoff]])
        test_feats.extend([(feat, label) for feat in feats[cutoff:]])
    return train_feats, test_feats


lfeats = label_feats_from_corpus(movie_reviews, feature_detector=bag_of_words)

lfeats.keys()

split_label_feats(lfeats)

train_feats, test_feats = split_label_feats(lfeats, split=0.75)

len(train_feats)

len(test_feats)

nb_classifier = NaiveBayesClassifier.train(train_feats)
nb_classifier.labels()

comment = 'Crazy. The story involved.'

nb_classifier.classify(bag_of_words(comment))

# Informa as palavras/features mais significantes para a classificação e suas
# probabilidades.
nb_classifier.show_most_informative_features(n=10)
