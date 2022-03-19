import pandas as pd
import pickle

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import gensim

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


def get_average_number_of_words(corpus):
    """
    Creates average number of words in a list. 
    Parameters:
    corpus - list of lists for each document
    
    Result
    average length of documents
    """
    lengths = [len((' '.join(doc).split())) for doc in corpus]
    avg_len = float(sum(lengths)) / len(lengths)
    
    return round(avg_len)


def create_vectors_for_recipe_id(word2vec_model, corpus, ids):
    """
    Creates dictionary of doc vectors for ids
    Parameters:
    corpus - list of documents
    ids - list of ids (position in list of ids corresponds to position in corpus)
    
    Result
    tfidf_vectorizer: TfidfEmbeddingVectorizer object
    doc_vec_dict - dictionary of ids and document vectors
    """
    
    if type(corpus[0]) is list:
        corpus_str = [' '.join(doc) for doc in corpus]
    else:
        corpus_str = corpus
        
    tfidf_vectorizer = TfidfEmbeddingVectorizer(word2vec_model)
    tfidf_vectorizer.fit(corpus_str)
    doc_vec = tfidf_vectorizer.create_doc_vectors(corpus)
     
    ids = ids.tolist()
    doc_vec_dict = defaultdict()
    for i in range(0, len(ids)):
        doc_vec_dict[ids[i]] = doc_vec[i]
        
    return tfidf_vectorizer, doc_vec_dict



def create_doc2vec_embeddings(df, col_name, ids_col_name='RecipeId', vector_size=100):
    """
    Creates doc2vec model
    Parameters:
    df - dataframe with ids and documents
    col_name - name of the columns with documents
    ids_col_names - name of the column with ids
    vector_size - size of vectors in the model
    
    Result
    doc2vec model
    """
    ids = df[ids_col_name]
    window_size = get_average_number_of_words(df[col_name])
    corpus = [TaggedDocument(row[1], [row[0]]) for index, row in df.iterrows()]
    doc2vec_model = Doc2Vec(corpus, vector_size=vector_size, window=2, min_count=1, workers=4)
    return doc2vec_model


def tsne_plot(word_labels, categories_dict, model=None, vectors_dict=None):
    labels = []
    tokens = []
    categories = []

    if not model:
        for key, value in word_labels.items():
            tokens.append(vectors_dict[key][0])
            labels.append(word_labels[key])
            categories.append(categories_dict[key])
    else:        
        for key, value in word_labels.items():
            tokens.append(model.dv[key])
            labels.append(value)
            categories.append(categories_dict[key])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23, verbose=False)
    new_values = tsne_model.fit_transform(tokens)
    
    
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    return pd.DataFrame({"X Value": x, "Y Value": y, "Category": categories, "Name": labels})
    
#     plt.figure(figsize=(16, 16)) 
#     for i in range(len(indexes)):
#         plt.scatter(x[i],y[i])
#         plt.annotate(labels[indexes[i]],
#                      xy=(x[i], y[i]),
#                      xytext=(5, 2),
#                      textcoords='offset points',
#                      ha='right',
#                      va='bottom')
#     plt.show()


def draw_scatter_per_category(values):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="X Value", y="Y Value",
        hue="Category",
        data=values,
        legend="full",
        palette=sns.color_palette("tab10"),
        alpha=0.3
    )
    
    
def draw_scatter_with_labels(values):
    list(mcolors.TABLEAU_COLORS)
    sample_for_label_values = values.sample(50)
    x = list(sample_for_label_values['X Value'])
    y = list(sample_for_label_values['Y Value'])
    labels = list(sample_for_label_values['Name'])
    categories = list(sample_for_label_values['Category'])

    colors = list(mcolors.TABLEAU_COLORS)
    categories_set = list(set(sample_for_label_values.Category))
    colors_for_categories = dict(zip(categories_set, colors))

    plt.figure(figsize=(20, 20)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c=colors_for_categories[categories[i]])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    
class TfidfEmbeddingVectorizer():
    """
    A class to represent a TFidf embedding vectorizer.

    ...

    Attributes
    ----------
    word2vec_model : Word2Vec gensim model
    word_idf_weight : float
        maximum idf value for documents in the corpus
    vector_size : int
        size of Word2Vec vectors
    tfidf: TfidfVectorizer class
    
    Methods
    -------
    fit(corpus):
        trains tfidf model and assigns idf value to each token
    create_doc_vectors(corpus):
        creates document vectors
    doc_average(doc):
        calculates weighted average of all tokens in document, weighted by idf value
    """
    
    def __init__(self, word2vec_model):
        self.word2vec_model = word2vec_model
        self.word_idf_weight = None
        self.vector_size = word2vec_model.wv.vector_size
        
    def fit(self, corpus):
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(corpus)
        
        max_idf = max(self.tfidf.idf_)
        self.word_idf_weight = defaultdict(
        lambda: max_idf, 
                [(word, self.tfidf.idf_[i]) for word, i in self.tfidf.vocabulary_.items()])
        return self
    
    def create_doc_vectors(self, corpus):
        doc_word_vector = np.vstack([self.doc_average(doc) for doc in corpus])
        doc_word_vector = [doc.reshape(1, -1) for doc in doc_word_vector]
        return doc_word_vector
    
    def doc_average(self, doc):
        mean = []
        for word in doc:
            if word in self.word2vec_model.wv.index_to_key:
                mean.append(self.word2vec_model.wv.get_vector(word) * self.word_idf_weight[word])
        if not mean:
            return np.zeros(self.vector_size)
        else:
            mean = np.array(mean).mean(axis=0)
            return mean