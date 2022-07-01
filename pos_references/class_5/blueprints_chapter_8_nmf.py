from sklearn.decomposition import NMF
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
# import matplotlib.pyplot as plt

# Assim como o método de Singular Value Decomposition (SVD), o método de
# Non-Negative Matrix Factorization (NMF) é um método de decomposição baseado
# em álgebra linear divergindo, dentre outras coisas, na forma como se
# decompõe as matrizes.

# V(documento x palavras) = W(documento x tópicos) * H(palavras x tópicos)
# É uma aproximação (assim como a SVD)

debates = pd.read_csv("src/pos_references/class_5/un-general-debates.csv")
debates.info()
df = debates

repr(df.iloc[2666]["text"][0:200])
repr(df.iloc[4729]["text"][0:200])

df["paragraphs"] = df["text"].map(lambda text: re.split('[.?!]\s*\n', text))
df["number_of_paragraphs"] = df["paragraphs"].map(len)

# %matplotlib inline
debates.groupby('year').agg({'number_of_paragraphs': 'mean'}).plot.bar()

# Será calculada a matriz TF-IDF para os discursos
# (linhas: discurso/documento x colunas: palavras) e
# para os parágrafos (linhas: parágrafo x colunas: palavras)

# Para os discursos temos:
tfidf_text = TfidfVectorizer(stop_words=stopwords, min_df=5, max_df=0.7)
vectors_text = tfidf_text.fit_transform(debates['text'])
print(vectors_text.shape)
"(7507, 24611)"

# Para os parágrafos temos:
# flatten the paragraphs keeping the years
paragraph_df = pd.DataFrame([{"text": paragraph, "year": year}
                             for paragraphs, year in
                             zip(df["paragraphs"], df["year"])
                             for paragraph in paragraphs if paragraph])

tfidf_para_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5,
                                        max_df=0.7)
tfidf_para_vectors = tfidf_para_vectorizer.fit_transform(paragraph_df["text"])
print(tfidf_para_vectors.shape)
"(282210, 25165)"
"""As linhas aumentaram porque obviamente cada discurso possui vários
parágrafos. As colunas também mudaram porque min_df e max_df têm um
efeito na seleção de recursos, pois o número de documentos mudou"""

# Realizando a decomposição da matriz TF-IDF pelo método NMF

# tfidf_text_vectors = tfidf_para_vectors or vectors_text

tfidf_text_vectors = tfidf_para_vectors

nmf_text_model = NMF(n_components=10, random_state=42)
W_text_matrix = nmf_text_model.fit_transform(tfidf_text_vectors)
H_text_matrix = nmf_text_model.components_

# Mostrando os tópicos


def display_topics(model, features, no_top_words=5):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]  # invert sort order
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print("  %s (%2.2f)" % (features[largest[i]],
                  word_vector[largest[i]]*100.0/total))


# tfidf_text_vectorizer = tfidf_text_vectorizer ou tfidf_para_vectorizer

tfidf_text_vectorizer = tfidf_para_vectorizer

display_topics(nmf_text_model, tfidf_text_vectorizer.get_feature_names_out())

# para mostrar o quão grande um tópico é:
W_text_matrix.sum(axis=0)/W_text_matrix.sum()*100.0
