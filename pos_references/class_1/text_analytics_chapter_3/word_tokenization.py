import nltk

"""
Ferramentas para tokenização em palavras:
1. word_tokenize
2. TreebankWordTokenizer
3. RegexpTokenizer
4. Inherited tokenizers from RegexpTokenizer
"""

# 1ª Ferramenta: utilizando o word_tokenize

sentence = "The brown fox wasn't that quick and he couldn't win the race"

default_wt = nltk.word_tokenize
words = default_wt(sentence)
print(words)
"""['The', 'brown', 'fox', 'was', "n't", 'that', 'quick',
    'and', 'he', 'could', "n't", 'win', 'the', 'race']"""

# 2ª Ferramenta: utilizando o TreebankWordTokenizer

treebank_wt = nltk.TreebankWordTokenizer()
words = treebank_wt.tokenize(sentence)
print(words)
"""['The', 'brown', 'fox', 'was', "n't", 'that', 'quick',
    'and', 'he', 'could', "n't", 'win', 'the', 'race']"""

# 3ª Ferramenta: utilizando o RegexpTokenizer

# pattern para identificar as palavras

TOKEN_PATTERN = r'\w+'
regex_wt = nltk.RegexpTokenizer(pattern=TOKEN_PATTERN, gaps=False)
words = regex_wt.tokenize(sentence)
print(words)
"""
['The', 'brown', 'fox', 'wasn', 't', 'that', 'quick', 'and', 'he', 'couldn',
't', 'win', 'the', 'race']
"""

# pattern para separar por espaços (evita a separação das contrações)

GAP_PATTERN = r'\s+'
regex_wt = nltk.RegexpTokenizer(pattern=GAP_PATTERN, gaps=True)
words = regex_wt.tokenize(sentence)
print(words)

# pega o índice inicial e final de cada token e imprime

word_indices = list(regex_wt.span_tokenize(sentence))
print(word_indices)
print([sentence[start:end] for start, end in word_indices])
"""
[(0, 3), (4, 9), (10, 13), (14, 20), (21, 25), (26, 31),
 (32, 35), (36, 38), (39, 47), (48, 51), (52, 55), (56, 60)]
['The', 'brown', 'fox', "wasn't", 'that', 'quick',
 'and', 'he', "couldn't", 'win', 'the', 'race']
 """

# 4ª Ferramenta: WordPunktTokenizer e WhitespaceTokenizer

# o WordPunktTokenizer separa os tokens por meio dos itens alfabéticos
# e não alfabéticos
wordpunkt_wt = nltk.WordPunctTokenizer()
words = wordpunkt_wt.tokenize(sentence)
print(words)
"""['The', 'brown', 'fox', 'wasn', "'", 't', 'that', 'quick',
    'and', 'he', 'couldn', "'", 't', 'win', 'the', 'race']"""

# o WhitespaceTokenizer separa pelos espaços em branco
whitespace_wt = nltk.WhitespaceTokenizer()
words = whitespace_wt.tokenize(sentence)
print(words)
"""['The', 'brown', 'fox', "wasn't", 'that', 'quick',
    'and', 'he', "couldn't", 'win', 'the', 'race']"""
