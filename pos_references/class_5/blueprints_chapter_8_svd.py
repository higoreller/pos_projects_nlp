# A modelagem de tópicos também pode ser realizada
# com o método de Singular Value Decomposition (SVD).

# V(m x n matriz) = U(m x k matriz unitária) * S(k x k matriz diagonal)
# * Vt(k x n matriz)

from sklearn.decomposition import TruncatedSVD
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords


debates = pd.read_csv("src/pos_references/class_5/un-general-debates.csv")
df = debates

repr(df.iloc[2666]["text"][0:200])
repr(df.iloc[4729]["text"][0:200])

# Para retirar as quebras de linha "\n" e outros caracteres
# que aparecem nos parágrafos
df["paragraphs"] = df["text"].map(lambda text: re.split('[.?!]\s*\n', text))
df["number_of_paragraphs"] = df["paragraphs"].map(len)


# Será calculada a matriz TF-IDF para os parágrafos x palavras, mas também
# pode ser calculada a matriz documentos x palavras.

# Para os parágrafos temos:
# flatten the paragraphs keeping the years
paragraph_df = pd.DataFrame([{"text": paragraph, "year": year}
                             for paragraphs, year in
                             zip(df["paragraphs"], df["year"])
                             for paragraph in paragraphs if paragraph])

tfidf_para_vectorizer = TfidfVectorizer(stop_words=stopwords, min_df=5,
                                        max_df=0.7)
tfidf_para_vectors = tfidf_para_vectorizer.fit_transform(paragraph_df["text"])

# Aplicando a decomposição SVD
svd_para_model = TruncatedSVD(n_components=10, random_state=42)
W_svd_para_matrix = svd_para_model.fit_transform(tfidf_para_vectors)
H_svd_para_matrix = svd_para_model.components_

# Função para mostrar os tópicos


def display_topics(model, features, no_top_words=5):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]  # invert sort order
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print("  %s (%2.2f)" % (features[largest[i]],
                  word_vector[largest[i]]*100.0/total))


display_topics(svd_para_model, tfidf_para_vectorizer.get_feature_names_out())

# Para vizualizar o tamanho dos tópicos:
svd_para_model.singular_values_
