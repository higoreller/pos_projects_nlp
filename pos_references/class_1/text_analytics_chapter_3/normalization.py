from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.stem import RegexpStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import PorterStemmer
import collections
from nltk.corpus import wordnet
from contractions import CONTRACTION_MAP
from nltk import clean_html, clean_url
import nltk
from nltk.corpus import gutenberg
import re
import string


# Obtendo dados e pacotes

# nltk.download('gutenberg')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

alice = gutenberg.raw(fileids='carroll-alice.txt')
sample_text = """We will discuss briefly about the basic syntax, structure
and design philosophies. There is a defined hierarchical syntax for Python
code which you should remember when writing code! Python is a really powerful
programming language!"""

"""
A normalização envolve vários passos:
1. Limpeza do texto
2. Tokenização
3. Remoção de caracteres especiais
"""

# 1º Passo: limpeza de alguns elementos do texto como html, url, etc.


def cleaning_text(text):
    cleaned_text = clean_html(text)
    cleaned_text = clean_url(cleaned_text)
    return cleaned_text

# 2º Passo: tokenização


def tokenize_text(text):
    sentences = nltk.sent_tokenize(text)
    word_tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return word_tokens


corpus = [alice, "Olá me chamo Higor Souza Eller.", sample_text]
token_list = [tokenize_text(text) for text in corpus]
# pprint(token_list)

# 3º Passo: remoção de caracteres especiais


def remove_characters_after_tokenization(tokens):
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(
        None, [pattern.sub('', token) for token in tokens])
    return filtered_tokens


filtered_list_1 = [filter(None, [remove_characters_after_tokenization(
    tokens) for tokens in sentence_tokens]) for sentence_tokens in token_list]
# print(filtered_list_1)


def remove_characters_before_tokenization(sentence,
                                          keep_apostrophes=False):
    sentence = sentence.strip()
    if keep_apostrophes:
        # add other characters here to remove them
        PATTERN = r'[?|!|$|&|*|%|@|(|)|~|.|,|;|:|+|=|_|/|\|]'
        filtered_sentence = re.sub(PATTERN, r'', sentence)
        filtered_sentence = filtered_sentence.replace("--", "")
        filtered_sentence = filtered_sentence.replace("''", "")
        filtered_sentence = filtered_sentence.replace("'", "")
        filtered_sentence = filtered_sentence.replace("`", "")
        filtered_sentence = filtered_sentence.replace("``", "")

    else:
        PATTERN = r'[^a-zA-Z0-9 ]'  # only extract alpha-numeric characters
        filtered_sentence = re.sub(PATTERN, r'', sentence)
    return filtered_sentence


filtered_list_2 = [remove_characters_before_tokenization(sentence)
                   for sentence in corpus]
# print(filtered_list_2)

cleaned_corpus = [remove_characters_before_tokenization(
    sentence, keep_apostrophes=True) for sentence in corpus]
# print(cleaned_corpus)

# 4º Passo: expandindo contrações


def expand_contractions(sentence, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(
        contraction_mapping.keys())),
        flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
            if contraction_mapping.get(match)\
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence


expanded_corpus = [expand_contractions(sentence, CONTRACTION_MAP)
                   for sentence in cleaned_corpus]
# print(expanded_corpus)

# 5º Passo: lowercase ou uppercase


def to_lower_case(sentence):
    return sentence.lower()


# 6º Passo: remoção de stopwords

def remove_stopwords(tokens):
    stopword_list = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


expanded_corpus_tokens = [tokenize_text(text)
                          for text in expanded_corpus]
filtered_list_3 = [[remove_stopwords(tokens)
                    for tokens in sentence_tokens]
                   for sentence_tokens in expanded_corpus_tokens]
# print(filtered_list_3)

# 7º Passo: correção de palavras incorretas (com letras duplicadas)


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens


sample_sentence = 'My schooool is realllllyyy amaaazingggg'
sample_sentence_tokens = tokenize_text(sample_sentence)[0]
# print(sample_sentence_tokens)
"""['My', 'schooool', 'is', 'realllllyyy', 'amaaazingggg']"""

# print(remove_repeated_characters(sample_sentence_tokens))
"""['My', 'school', 'is', 'really', 'amazing']"""

# 8º Passo: correção de palavras que foram digitadas/escritas erradas


def tokens(text):
    """
    Get all words from the corpus
    """
    return re.findall('[a-z]+', text.lower())


WORDS = tokens(
    open('src/pos_references/class_1/text_analytics_chapter_3/big.txt', 'r')
    .read())
WORD_COUNTS = collections.Counter(WORDS)

# top 10 words in the corpus
print(WORD_COUNTS.most_common(10))
"""[('the', 80030), ('of', 40025), ('and', 38313), ('to', 28766),
('in', 22050), ('a', 21155), ('that', 12512), ('he', 12401),
('was', 11410), ('it', 10681)]"""

# retorno de strings que foram editadas nenhuma, uma ou duas vezes a partir
# do input


def edits0(word):
    """
    Return all strings that are zero edits away
    from the input word (i.e., the word itself).
    """
    return {word}


def edits1(word):
    """
    Return all strings that are one edit away
    from the input word.
    """
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    def splits(word):
        """
        Return a list of all possible (first, rest) pairs
        that the input word is made of.
        """
        return [(word[:i], word[i:]) for i in range(len(word)+1)]

    pairs = splits(word)
    deletes = [a+b[1:] for (a, b) in pairs if b]
    transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
    replaces = [a+c+b[1:] for (a, b) in pairs for c in alphabet if b]
    inserts = [a+c+b for (a, b) in pairs for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    """Return all strings that are two edits away
    from the input word.
    """
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def known(words):
    """
    Return the subset of words that are actually
    in our WORD_COUNTS dictionary.
    """
    return {w for w in words if w in WORD_COUNTS}


# Teste do algoritmo de correção de palavras
# input word
word = 'fianlly'

# zero edit distance from input word
edits0(word)
"{'fianlly'}"
# returns null set since it is not a valid word
known(edits0(word))
"set()"

# one edit distance from input word
edits1(word)
"""{'afianlly', 'aianlly', ..'yianlly', 'zfianlly', 'zianlly'}"""
# get correct words from above set
known(edits1(word))
"{'finally'}"

# two edit distances from input word
edits2(word)
"""{'fchnlly', 'fianjlys', ..'fiapgnlly', 'finanlqly'}"""
# get correct words from above set
known(edits2(word))
"""{'faintly', 'finally', 'finely', 'frankly'}"""

# priorizando as palavras que precisam de menos edições

candidates = (known(edits0(word)) or
              known(edits1(word)) or
              known(edits2(word)) or
              [word])

candidates
"{'finally'}"

# No caso de empate de palavras, este algoritmo prioriza as palavras
# de maior ocorrência do nosso vocabulário


def correct(word):
    """
    Get the best correct spelling for the input word
    """
    # Priority is for edit distance 0, then 1, then 2
    # else defaults to the input word itself.
    candidates = (known(edits0(word)) or
                  known(edits1(word)) or
                  known(edits2(word)) or
                  [word])
    return max(candidates, key=WORD_COUNTS.get)


# É case sensitive e falha ao corrigir palavras com letras maiúsculas
correct('fianlly')
'finally'

correct('FIANLLY')
'FIANLLY'

# Função para correção de palavras sem case sensitive


def correct_match(match):
    """
    Spell-correct word in match,
    and preserve proper upper/lower/title case.
    """

    word = match.group()

    def case_of(text):
        """
        Return the case-function appropriate
        for text: upper, lower, title, or just str.:
            """
        return (str.upper if text.isupper() else
                str.lower if text.islower() else
                str.title if text.istitle() else
                str)
    return case_of(word)(correct(word.lower()))


def correct_text_generic(text):
    """
    Correct all the words within a text,
    returning the corrected text.
    """
    return re.sub('[a-zA-Z]+', correct_match, text)


# Pode-se utilizar uma biblioteca externa para correção de textos
# from pattern.en import suggest (deu problema na instalação)
"""print(suggest('fianlly'))
[('finally', 1.0)]

print(suggest('flaot'))
[('flat', 0.85), ('float', 0.15)]"""

# 9º Passo: stemming

# Porter Stemming
ps = PorterStemmer()

print(ps.stem('jumping'), ps.stem('jumps'), ps.stem('jumped'))
"jump jump jump"

print(ps.stem('lying'))
"lie"

print(ps.stem('strange'))
"strang"

# Lancaster Stemming

ls = LancasterStemmer()

print(ls.stem('jumping'), ls.stem('jumps'), ls.stem('jumped'))
"jump jump jump"

print(ls.stem('lying'))
"lying"

print(ls.stem('strange'))
"strange"

# Regex based stemmer

rs = RegexpStemmer('ing$|s$|ed$', min=4)

print(rs.stem('jumping'), rs.stem('jumps'), rs.stem('jumped'))
"jump jump jump"

print(rs.stem('lying'))
"ly"

print(rs.stem('strange'))
"strange"


# Snowball Stemmer: tem suporte para outras línguas
ss = SnowballStemmer("german")

print('Supported Languages:', SnowballStemmer.languages)
"""Supported Languages: (u'danish', u'dutch', u'english', u'finnish',
u'french', u'german', u'hungarian', u'italian', u'norwegian', u'porter',
u'portuguese', u'romanian', u'russian', u'spanish', u'swedish')"""

# stemming on German words
# autobahnen -> cars
# autobahn -> car
print(ss.stem('autobahnen'))
u'autobahn'

# springen -> jumping
# spring -> jump
print(ss.stem('springen'))
u'spring'


# 10º Passo: Lemmatization

wnl = WordNetLemmatizer()

# lemmatize nouns
wnl.lemmatize('cars', 'n')
wnl.lemmatize('men', 'n')
"car"
"men"

# lemmatize verbs
wnl.lemmatize('running', 'v')
wnl.lemmatize('ate', 'v')
'run'
'eat'

# lemmatize adjectives
wnl.lemmatize('saddest', 'a')
wnl.lemmatize('fancier', 'a')
'sad'
'fancy'
