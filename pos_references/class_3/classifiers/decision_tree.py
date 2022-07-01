from nltk.classify import DecisionTreeClassifier
from naive_bayes import train_feats, test_feats
from nltk.classify.util import accuracy

dt_classifier = DecisionTreeClassifier.train(
    train_feats, binary=True, entropy_cutoff=0.8, depth_cutoff=5,
    support_cutoff=30)
print(accuracy(dt_classifier, test_feats))

# binary =>
# entropy_cutoff =>
# depth_cutoff =>
# suport_cutoff =>
