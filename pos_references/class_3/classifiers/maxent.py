from nltk.classify import MaxentClassifier
from nltk.classify.util import accuracy
from naive_bayes import train_feats, test_feats

me_classifier = MaxentClassifier.train(
    train_feats, trace=0, max_iter=1, min_lldelta=0.5)
accuracy(me_classifier, test_feats)
# Vai retornar 0.5

me_classifier = MaxentClassifier.train(
    train_feats, algorithm='gis', trace=0, max_iter=10, min_lldelta=0.5)
accuracy(me_classifier, test_feats)

# Vai retornar 0.722 devido a mudança do algoritmo de "iis" (default) para
# "gis"

# Posteriormente tentar instalar o algoritmo "megam"
