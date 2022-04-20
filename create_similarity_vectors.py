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


def create_top_k_similar_vectors(vectors_dict, items_to_check_list, top_k=1, verbose=True):
    '''
    Function creates dictionary of top k similar recipes for each recipe in specified in items_to_check_list from embeddings in vectors_dict
    Params:
        vectors_dict - key: recipeId, value: vector embedding
        items_to_check_list - list of recipe ids for which function creates top k list
        top_k - size of top k list
        verbose - should function print status messages
        
    Result:
        top - dictionary of top-k items per recipe id
    '''
    #create dictionary of positions in list
    list_pos_to_recipe_id = {}
    recipe_id_to_list_pos = {}
    i = 0
    for key in vectors.keys():
        list_pos_to_recipe_id[i] = key
        recipe_id_to_list_pos[key] = i
        i += 1
    
    # create list of vectors from dictionary
    vectors = list(vectors_dict.values())
    vectors = [np.array(x).ravel() for x in vectors]

    tensors = torch.tensor(vectors, dtype=torch.float)

    if (top_k+1) > len(tensors):
        top_k = len(tensors)
    else:
        top_k += 1
    
    i = 0
    top_scores = defaultdict()
    for key in items_to_check_list:
        if(i % 1000 == 0 and verbose):
            print("Iteration: ", i)
        cos_scores = util.pytorch_cos_sim(tensors[recipe_id_to_list_pos[key]], 
                                          tensors)
        top_scores[key] = torch.topk(cos_scores, k=top_k)
        i += 1
        
        
    top = defaultdict(list)
    if(verbose):
        print("Creating dictionary...")
    for key in items_to_check_list:
        top[key] = [(list_pos_to_recipe_id[k[0].item()],k[1].item()) for k 
                       in list(tuple(zip(top_scores[key][1][0], top_scores[key][0][0])))
                       if list_pos_to_recipe_id[k[0].item()] != key]
        
    return top


def create_similarity_matrix(vectors_dict, verbose=False):
    '''
    Creates nxn simialarity matrix for items in vectors_dict
    Params:
        vectors_dict - key: recipeId, value: vector embedding
        
    Result:
        pos_to_recipe_id - dictionary: position in matrix to recipe id
        recipe_id_to_pos - dictionary: recipe id to position in matrix
        similarities - similarity matrix
    '''
    pos_to_recipe_id = {}
    recipe_id_to_pos = {}
    i = 0
    for key in vectors.keys():
        pos_to_recipe_id[i] = key
        recipe_id_to_pos[key] = i
        i += 1
    
    # create list of vectors from dictionary
    vectors = list(vectors.values())
    vectors = [np.array(x).ravel() for x in vectors]

    tensors = torch.tensor(np.array(vectors), dtype=torch.float)
    
    size = len(vectors)
    similarities = np.zeros((size, size))
    for this_item in range(size):
        if (this_item % 500 == 0 and verbose==True):
            print(this_item, " of ", size)
        for other_item in range(this_item+1, size):
            tensor1 = torch.tensor(vectors[this_item], dtype=torch.float)
            tensor2 = torch.tensor(vectors[other_item], dtype=torch.float)

            sim = util.pytorch_cos_sim(tensor1, tensor2)[0][0].item()
            similarities[this_item, other_item] = sim
            similarities[other_item, this_item] = sim
            
    return pos_to_recipe_id, recipe_id_to_pos, similarities