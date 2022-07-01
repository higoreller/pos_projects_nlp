import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as stopwords
import pandas as pd
import re

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

print(count_para_vectors)
# Utilizando o LDA para gerar a distribuição probabilística
# Este processo é bem mais demorado do que o NMF e o SVD
lda_para_model = LatentDirichletAllocation(n_components=2, random_state=42)
W_lda_para_matrix = lda_para_model.fit_transform(count_para_vectors)
H_lda_para_matrix = lda_para_model.components_

print(W_lda_para_matrix)
print(H_lda_para_matrix)


def wordcloud_topics(model, features, no_top_words=40):
    for topic, words in enumerate(model.components_):
        size = {}
        largest = words.argsort()[::-1]  # invert sort order
        for i in range(0, no_top_words):
            size[features[largest[i]]] = abs(words[largest[i]])
        wc = WordCloud(background_color="white", max_words=100,
                       width=960, height=540)
        wc.generate_from_frequencies(size)
        plt.figure(figsize=(12, 12))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        # if you don't want to save the topic model, comment the next line
        plt.savefig(f'topic{topic}.png')


# wordcloud_topics(nmf_para_model, tfidf_para_vectorizer.get_feature_names())
wordcloud_topics(lda_para_model, count_para_vectorizer.get_feature_names_out())
