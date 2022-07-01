# Método baseado em distribuições de probabilidade


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
import pandas as pd
import re
from blueprints_chapter_8_nmf import tfidf_para_vectorizer

debates = pd.read_csv("src/pos_references/class_5/un-general-debates.csv")
debates.info()
df = debates

df["paragraphs"] = df["text"].map(lambda text: re.split('[.?!]\s*\n', text))

# flatten the paragraphs keeping the years
paragraph_df = pd.DataFrame([{"text": paragraph, "year": year}
                             for paragraphs, year in
                             zip(df["paragraphs"], df["year"])
                             for paragraph in paragraphs if paragraph])

# A vetorização utilizada foi a baseada na frequência de
# ocorrência das palavras
count_para_vectorizer = CountVectorizer(stop_words=stopwords, min_df=5,
                                        max_df=0.7)
count_para_vectors = count_para_vectorizer.fit_transform(paragraph_df["text"])

# Utilizando o LDA para gerar a distribuição probabilística
# Este processo é bem mais demorado do que o NMF e o SVD
lda_para_model = LatentDirichletAllocation(n_components=10, random_state=42)
W_lda_para_matrix = lda_para_model.fit_transform(count_para_vectors)
H_lda_para_matrix = lda_para_model.components_

# Mostrando os tópicos:


def display_topics(model, features, no_top_words=5):
    for topic, word_vector in enumerate(model.components_):
        total = word_vector.sum()
        largest = word_vector.argsort()[::-1]  # invert sort order
        print("\nTopic %02d" % topic)
        for i in range(0, no_top_words):
            print("  %s (%2.2f)" % (features[largest[i]],
                  word_vector[largest[i]]*100.0/total))


# Não o porque da utilização do tfidf_para_vectorizer
display_topics(lda_para_model, tfidf_para_vectorizer.get_feature_names_out())


"""
Para visualizar com o pyLDAvis:

import pyLDAvis.sklearn

lda_display = pyLDAvis.sklearn.prepare(lda_para_model, count_para_vectors,
                                       count_para_vectorizer,
                                        sort_topics=False)
pyLDAvis.display(lda_display)

"""
