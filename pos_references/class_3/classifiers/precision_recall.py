import collections
from nltk.metrics import precision, recall
from naive_bayes import nb_classifier
from naive_bayes import test_feats
from maxent import me_classifier
from sklearn_classifier import sk_classifier


def precision_recall(classifier, testfeats):
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    precisions = {}
    recalls = {}

    for label in classifier.labels():
        # Chamar a função diretamente, metrics.precision causa erro
        precisions[label] = precision(refsets[label], testsets[label])
        # Chamar a função diretamente, metrics.recall causa erro
        recalls[label] = recall(refsets[label], testsets[label])

    return precisions, recalls


nb_precisions, nb_recalls = precision_recall(nb_classifier, test_feats)

nb_precisions['pos']  # 0,65
nb_precisions['neg']  # 0,95
nb_recalls['pos']  # 0,98
nb_recalls['neg']  # 0,476

"""Isso nos diz que, embora a classe NaiveBayesClassifier possa identificar
corretamente a maioria dos conjuntos de recursos "pos" (revocação alta), ela
também classifica muitos dos conjuntos de recursos negativos como 'pos'
(baixa precisão). Esse comportamento contribui para alta precisão, mas baixa
rechamada para o rótulo 'neg' — como o rótulo 'neg' não é fornecido
com frequência (revocação baixa), quando é, é muito provável que esteja
correto (alta precisão). A conclusão pode ser que existem certas palavras
comuns que são tendenciosas para o rótulo pos, mas ocorrem com frequência
suficiente nos conjuntos de recursos 'neg' para causar classificações
incorretas. Para corrigir esse comportamento, usaremos apenas as palavras
mais informativas na próxima seção."""

me_precisions, me_recalls = precision_recall(me_classifier, test_feats)
me_precisions['pos']  # 0,65
me_precisions['neg']  # 0,97
me_recalls['pos']  # 0,98
me_recalls['neg']  # 0,46

sk_precisions, sk_recalls = precision_recall(sk_classifier, test_feats)
sk_precisions['pos']  # 0,90
sk_precisions['neg']  # 0,85
sk_recalls['pos']  # 0,85
sk_recalls['neg']  # 0,90

# nltk.metrics.f_measure() => calcula a média harmônica ponderada da precisão
# e do recall
