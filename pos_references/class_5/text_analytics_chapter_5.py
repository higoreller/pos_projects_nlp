from pyparsing import unicodeString
from scipy.sparse.linalg import svds
import nltk
import re
import unicodedata
from HTMLParser import HTMLParser


""" O processo de normalização pode ser divido em 6 etapas:

1. Sentence extraction
2. Unescape HTML escape sequences
3. Expand contractions
4. Lemmatize text
5. Remove special characters
6. Remove stopwords

OBS.: outras ferramentas que devem/podem ser usadas para a normalização:

Utilização de expressões regulares
.lower() => converte todo o texto para minúsculo
.strip() => remove espaços em branco no início e fim do texto
.replace() => substitui uma string por outra
.split() => divide o texto em uma lista de strings
.join() => junta uma lista de strings em um texto
.encode() => converte um texto para bytes
.decode() => converte bytes para texto

"""

# Passo 1: extração das sentenças do documento


def parse_document(document):
    document = re.sub('\n', ' ', document)
    if isinstance(document, bytes):
        document = document
    elif isinstance(document, str):
        # normaliza o documento antes de aplicar o encode
        return unicodedata.normalize('NFKD', document).encode('ascii', 'ignore')
    else:
        raise ValueError('Document is not string or unicode!')
    document = document.strip()
    sentences = nltk.sent_tokenize(document)
    sentences = [sentence.strip() for sentence in sentences]
    return sentences

# Segundo passo: lida com caracteres HTML especiais sem escape que são
# escapados ou codificados. A segunda parte é uma lemmatização, caso
# queira que seja feita.


html_parser = HTMLParser()


def unescape_html(parser, text):
    return parser.unescape(text)


def normalize_corpus(corpus, lemmatize=True, tokenize=False):

    normalized_corpus = []
    for text in corpus:
        text = html_parser.unescape(text)
        text = expand_contractions(text, CONTRACTION_MAP)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)

    return normalized_corpus

# Singular Value Decomposition (SVD) utilizando low rank matrix


def low_rank_svd(matrix, singular_count=2):

    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt
