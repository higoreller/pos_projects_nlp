# from pattern.text.en import tag
from nltk.corpus import treebank
import nltk

# nltk.download('treebank')
"""
Parts of speech (POS) tagging

Shallow parsing

Dependency-based parsing

Constituency-based parsing
"""

"""
The nltk library, preferably version 3.1 or 3.2.1

The spacy library

The pattern library

The Stanford parser

Graphviz and necessary libraries for the same
"""


sentence = 'The brown fox is quick and he is jumping over the lazy dog'

tokens = nltk.word_tokenize(sentence)
tagged_sent = nltk.pos_tag(tokens, tagset='universal')
tagged_sent
"""[('The', u'DET'), ('brown', u'ADJ'), ('fox', u'NOUN'), ('is', u'VERB'),
('quick', u'ADJ'), ('and', u'CONJ'), ('he', u'PRON'), ('is', u'VERB'),
('jumping', u'VERB'), ('over', u'ADP'), ('the', u'DET'), ('lazy', u'ADJ'),
('dog', u'NOUN')]
"""

# tagged_sent = tag(sentence)
# print(tagged_sent)
"""[(u'The', u'DT'), (u'brown', u'JJ'), (u'fox', u'NN'), (u'is', u'VBZ'),
(u'quick', u'JJ'), (u'and', u'CC'), (u'he',u'PRP'), (u'is', u'VBZ'),
(u'jumping', u'VBG'), (u'over', u'IN'), (u'the', u'DT'), (u'lazy', u'JJ'),
(u'dog', u'NN')]
"""

data = treebank.tagged_sents()
train_data = data[:3500]
test_data = data[3500:]

# get a look at what each data point looks like
print(train_data[0])
"""[(u'Pierre', u'NNP'), (u'Vinken', u'NNP'), (u',', u','), (u'61', u'CD'),
(u'years', u'NNS'), (u'old', u'JJ'), (u',', u','), (u'will', u'MD'),
(u'join', u'VB'), (u'the', u'DT'), (u'board', u'NN'), (u'as', u'IN'),
(u'a', u'DT'), (u'nonexecutive', u'JJ'), (u'director', u'NN'),
(u'Nov.', u'NNP'), (u'29', u'CD'),
(u'.', u'.')]"""

# remember tokens is obtained after tokenizing our sentence
tokens = nltk.word_tokenize(sentence)
print(tokens)
"""['The', 'brown', 'fox', 'is', 'quick', 'and', 'he', 'is',
'jumping', 'over', 'the', 'lazy', 'dog']"""
