"""
Ferramentas de extração de sentenças
1. sent_tokenize
2. PunktSentenceTokenizer
3. RegexpTokenizer
4. Pre-trained sentence tokenization models
"""

import nltk
from nltk.corpus import gutenberg
from pprint import pprint

# nltk.download('gutenberg')

alice = gutenberg.raw(fileids='carroll-alice.txt')
sample_text = """We will discuss briefly about the basic syntax,
structure and design philosophies. There is a defined hierarchical
syntax for Python code which you should remember when writing code!
Python is a really powerful programming language!"""

# 1ª Ferramenta: utilizando o sent_tokenize

default_st = nltk.sent_tokenize
alice_sentences = default_st(text=alice)
sample_sentences = default_st(text=sample_text)

"""
print('Total sentences in sample_text:', len(sample_sentences))
print('Sample text sentences :-')
pprint(sample_sentences)
print('\nTotal sentences in alice:', len(alice_sentences))
print('First 5 sentences in alice:-')
pprint(alice_sentences[0:5])
"""

# 2ª Ferramenta: utilizando o PunktSentenceTokenizer

punkt_st = nltk.tokenize.PunktSentenceTokenizer()
sample_sentences = punkt_st.tokenize(sample_text)
# pprint(sample_sentences)

# 3ª Ferramenta: utilizando o RegexpTokenizer

SENTENCE_TOKENS_PATTERN = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!)\s'
regex_st = nltk.tokenize.RegexpTokenizer(
    pattern=SENTENCE_TOKENS_PATTERN,
    gaps=True)
sample_sentences = regex_st.tokenize(sample_text)
pprint(sample_sentences)
