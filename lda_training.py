from sklearn.feature_extraction.text import CountVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.models.ldamodel import LdaModel
from gensim import models, similarities
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.coherencemodel import CoherenceModel


def train_lda_model(list_of_docs, num_of_topics, num_of_passes):
    dictionary = Dictionary([ing for ing in list(list_of_docs)])
    corpus = [dictionary.doc2bow(text) for text in list(list_of_docs)]
    lda_model = LdaModel(corpus, num_topics = num_of_topics, passes = num_of_passes, id2word = dictionary)
    return dictionary, corpus, lda_model


def get_topics_vector_for_doc(corpus, lda_model, doc_number):
    vector = lda_model.get_document_topics(corpus[doc_number], minimum_probability=0.0)
    return vector


def get_top_topics_for_doc(corpus, lda_model, doc_number, n=1):
    vector = get_topics_vector_for_doc(corpus, lda_model, doc_number)
    topics = sorted(vector,key=lambda x:x[1],reverse=True)
    return topics[:n]

def get_topic_words_dict(lda_model, num_topics, num_words):
    x=lda_model.show_topics(num_topics=num_topics, num_words=num_words,formatted=False)
    topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]
    topics_dict = {}
    for topic, words in topics_words:
        topics_dict[topic] = words
    return topics_dict


def create_topic_per_doc_dict(lda_model, corpus, ids, n=1):
    topic_per_doc_dict = {}
    ids = list(ids)
    for i in range(0, len(ids)):
        topics = get_top_topics_for_doc(corpus=corpus, lda_model=lda_model, doc_number=i, n=n)
        topic_per_doc_dict[ids[i]] = []
        for topic in topics:
            topic_per_doc_dict[ids[i]].append(topic[0])
    return topic_per_doc_dict


def create_doc_per_topic_dict(topic_per_doc_dict, ids, n=1):
    doc_per_topics_dict = {}
    ids = list(ids)
    for i in range(0, len(ids)):
        topics = topic_per_doc_dict[ids[i]]
        for topic in topics:
            if topic in doc_per_topics_dict.keys():
                doc_per_topics_dict[topic].append(ids[i])
            else:
                doc_per_topics_dict[topic] = [ids[i]]
    return doc_per_topics_dict


def compute_coherence_values(list_of_docs, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        dictionary, corpus, lda_model = train_lda_model(list_of_docs, num_topics, 50)
        model_list.append(lda_model)
        coherence_model = CoherenceModel(model=lda_model, texts=list_of_docs, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())

    return model_list, coherence_values