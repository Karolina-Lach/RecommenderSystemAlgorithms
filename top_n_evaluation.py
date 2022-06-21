import pandas as pd
import pickle
from collections import defaultdict

from surprise import NormalPredictor
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise import AlgoBase
from surprise.model_selection import train_test_split as train_test_split_sup

import metrics
import sampling

import numpy as np

from sklearn.model_selection import train_test_split as train_test_split
from surprise.model_selection import LeaveOneOut
import random

from create_similarity_vectors import create_top_k_similar_vectors
from sentence_transformers import util
import torch
from surprise.prediction_algorithms.predictions import PredictionImpossible
import heapq




def create_anti_testset_for_user_tfrs(user_id, 
                                 user_items_in_trainset,
                                 ratings_df, 
                                 sample_size=500, 
                                 user_sample_size=0,
                                 knn=False):
    # get all items rated by user
    user_items = ratings_df[ratings_df.AuthorId == user_id][['RecipeId', 'Rating']]
    # get all items rated by user that are NOT in the trainset
    candidates = user_items[~user_items.RecipeId.isin(user_items_in_trainset)].RecipeId.unique()
    
    # all items in ratings that are NOT rated by user
    all_items = ratings_df[~ratings_df.RecipeId.isin(user_items_in_trainset)]
    all_items = all_items[~all_items.RecipeId.isin(candidates)].RecipeId.unique()

    if user_sample_size > 0:
        candidates = np.random.choice(candidates, size=min(user_sample_size, len(candidates)))
    else:
        if (len(candidates) >= sample_size):
    #         print(len(user_items_in_trainset))
            candidates = np.random.choice(candidates, size=min(len(candidates), int(sample_size*0.5)))
        
#     number_of_rated_items = len(set(candidates).intersection(user_items.index))
    sample_size = sample_size - len(candidates)

    items_sample = np.random.choice(all_items, size=sample_size)
    fill = ratings_df.Rating.mean()
    
    anti_testset = []
    relevant_items = []
    for item_id in items_sample:
        if knn:
            anti_testset.append(item_id)
        else:
            anti_testset.append((user_id, item_id, fill))

    for item_id in candidates:
        if knn:
            anti_testset.append(item_id)
        else:
            anti_testset.append((user_id, item_id, fill))
        relevant_items.append(item_id)
    
    random.shuffle(anti_testset)

    return anti_testset, relevant_items




def create_anti_testset_for_user_all(user_id, user_items_in_trainset, ratings_df, knn=False):
    
    # get all user items
    user_items = ratings_df[ratings_df.AuthorId == user_id][['RecipeId', 'Rating']].set_index("RecipeId")
    # all items used in ratings
    all_items = ratings_df.RecipeId.unique()
    # get mean rating
    fill = ratings_df.Rating.mean()

    # print(f"All items {len(all_items)}")
    # print(f"All items in trainset {len(user_items_in_trainset)}")
    anti_testset = []
    relevant_items = []
    for item_id in all_items:
        if item_id not in user_items_in_trainset:
            if knn:
                if item_id in user_items.index:
                    # print('x')
                    anti_testset.append(item_id)
                    relevant_items.append(item_id)
                else:
                    anti_testset.append(item_id)
            else:    
                if item_id in user_items.index:
                    # print('x')
                    anti_testset.append((user_id, item_id, user_items.loc[item_id, 'Rating']))
                else:
                    anti_testset.append((user_id, item_id, fill))
        

    return anti_testset, relevant_items


def create_anti_testset_for_user(user_id, 
                                 user_items_in_trainset,
                                 ratings_df, 
                                 sample_size=300, 
                                 user_sample_size=20,
                                 knn=False):
    # get all items rated by user
    user_items = ratings_df[ratings_df.AuthorId == user_id][['RecipeId', 'Rating']]
    # get all items rated by user that are NOT in the trainset
    candidates = user_items[~user_items.RecipeId.isin(user_items_in_trainset)].RecipeId.unique()
    
    # all items in ratings that are NOT rated by user
    all_items = ratings_df[~ratings_df.RecipeId.isin(user_items_in_trainset)]
    all_items = all_items[~all_items.RecipeId.isin(candidates)].RecipeId.unique()

    if user_sample_size > 0:
        candidates = np.random.choice(candidates, size=min(user_sample_size, len(candidates)))
    else:
        if (len(candidates) >= sample_size):
    #         print(len(user_items_in_trainset))
            candidates = np.random.choice(candidates, size=min(len(candidates), int(sample_size*0.5)))
        
#     number_of_rated_items = len(set(candidates).intersection(user_items.index))
    sample_size = sample_size - len(candidates)

    items_sample = np.random.choice(all_items, size=sample_size)
    fill = ratings_df.Rating.mean()
    
    anti_testset = []
    relevant_items = []
    for item_id in items_sample:
        if knn:
            anti_testset.append(item_id)
        else:
            anti_testset.append((user_id, item_id, fill))

    for item_id in candidates:
        if knn:
            anti_testset.append(item_id)
        else:
            anti_testset.append((user_id, item_id, fill))
        relevant_items.append(item_id)
    
    random.shuffle(anti_testset)

    return anti_testset, relevant_items



def create_recommendation_top_n_evaluation(train_df, 
                                           ratings_df, 
                                           algorithm=None, 
                                           word2vec_vectors=None,
                                           sample_size=300, 
                                           user_sample_size=20,
                                           k=100,
                                           knn=False,
                                           verbose=False):
    recommendations = defaultdict(list)
    relevant_items = defaultdict(list)
    i = 0
    
    for user_id in train_df.AuthorId.unique():
        if i % 250 == 0 and verbose:
            print(i)
        user_items_in_trainset = train_df[train_df.AuthorId == user_id]['RecipeId'].unique()
        if sample_size > 0:
            anti_testset, relevant_items[user_id] = create_anti_testset_for_user(user_id=user_id, 
                                                        user_items_in_trainset=user_items_in_trainset, 
                                                        ratings_df=ratings_df, 
                                                        sample_size=sample_size,
                                                        user_sample_size=user_sample_size,
                                                        knn=knn)
        else:
            anti_testset, relevant_items[user_id] = create_anti_testset_for_user_all(user_id, 
                                                                                     user_items_in_trainset, 
                                                                                     ratings_df, 
                                                                                     knn=knn)
        
        if knn:
            list_pos_to_recipe_id = {}
            recipe_id_to_list_pos = {}
            
            j = 0
            vectors = []
            for item in anti_testset:
                list_pos_to_recipe_id[j] = item
                recipe_id_to_list_pos[item] = j
                vectors.append(word2vec_vectors[item])
                j += 1
            
            vectors = [np.array(x).ravel() for x in vectors]
            tensors = torch.tensor(vectors, dtype=torch.float)
#             print(len(tensors))
            top_scores = {}
            top = []
#             print("User id: ", user_id)
#             print("User items: ", len(user_items_in_trainset))
            
            for item in user_items_in_trainset:
                
                cos_scores = util.pytorch_cos_sim(torch.tensor(word2vec_vectors[item]), 
                                                  tensors)
                top_scores[item] = (torch.topk(cos_scores, k))
            
                top += [(list_pos_to_recipe_id[k[0].item()],k[1].item()) for k 
                           in list(tuple(zip(top_scores[item][1][0], top_scores[item][0][0])))
                           if list_pos_to_recipe_id[k[0].item()] != item]
            
            top.sort(key=lambda x: x[1], reverse=True)
            recommendations[user_id] = [x[0] for x in top[:k]]
            
        else:
            predictions = algorithm.test(anti_testset)
            predictions.sort(key=lambda x: x[3], reverse=True)
            predictions = predictions[:k]
            predictions_list = [iid for uid, iid, r_ui, est, _ in predictions]

            recommendations[user_id] = predictions_list
        
        i += 1
        
    return recommendations, relevant_items




def create_anti_testset_for_user_all_tfrs(user_id, user_items_in_trainset, ratings_df, knn=False):
    
    # get all user items
    user_items = ratings_df[ratings_df.AuthorId == user_id][['RecipeId', 'Rating']].set_index("RecipeId")
    # all items used in ratings
    all_items = ratings_df.RecipeId.unique()
    # get mean rating
    fill = ratings_df.Rating.mean()

    # print(f"All items {len(all_items)}")
    # print(f"All items in trainset {len(user_items_in_trainset)}")
    anti_testset = []
    relevant_items = []
    for item_id in all_items:
        if item_id not in user_items_in_trainset:
            if knn:
                if item_id in user_items.index:
                    # print('x')
                    anti_testset.append(item_id)
                    relevant_items.append(item_id)
                else:
                    anti_testset.append(item_id)
            else:    
                if item_id in user_items.index:
                    # print('x')
                    anti_testset.append((user_id, item_id, user_items.loc[item_id, 'Rating']))
                else:
                    anti_testset.append((user_id, item_id, fill))
        

    return anti_testset, relevant_items