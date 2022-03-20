import pickle
from gensim.models import Word2Vec
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import pandas as pd
from sentence_transformers import util
import torch

from collections import defaultdict
import heapq


def create_top_k_similar_vectors(vectors, items_to_check, top_k=1, verbose=False):

    #create dictionary of positions in list
    list_pos_to_recipe_id = {}
    recipe_id_to_list_pos = {}
    i = 0
    for key in vectors.keys():
        list_pos_to_recipe_id[i] = key
        recipe_id_to_list_pos[key] = i
        i += 1
    
    # create list of vectors from dictionary
    vectors = list(vectors.values())
    vectors = [np.array(x).ravel() for x in vectors]

    tensors = torch.tensor(vectors, dtype=torch.float)

    if (top_k+1) > len(tensors):
        top_k = len(tensors)
    else:
        top_k += 1
    
    top_scores = defaultdict()
    i = 0
    for key in items_to_check:
        if(i % 1000 == 0 and verbose):
            print("Iteration: ", list_pos)
        cos_scores = util.pytorch_cos_sim(tensors[recipe_id_to_list_pos[key]], 
                                          tensors)
        top_scores[key] = torch.topk(cos_scores, k=top_k)
        i += 1
        
    top = defaultdict(list)
    for key in items_to_check:
        top[key] = [(list_pos_to_recipe_id[k[0].item()],k[1].item()) for k 
                       in list(tuple(zip(top_scores[key][1][0], top_scores[key][0][0])))
                       if list_pos_to_recipe_id[k[0].item()] != key]
        
    return top