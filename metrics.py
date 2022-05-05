import pandas as pd
from surprise import Dataset
from surprise import accuracy
from collections import defaultdict
import math
import numpy as np
import itertools


def predictions_to_dataframe(predictions):
    '''
    Creates dataframe from predictions with columns:("AuthorId", "RecipeId", "Ratings", "PredictedRating", "WasImpossible")
    '''
    if (len(predictions) > 0):
        if (type(predictions[0]) != tuple):
             predictions_tuple = predictions_to_tuple_list(predictions)
    pred_df = pd.DataFrame(predictions_tuple, columns=("AuthorId", "RecipeId", "Ratings", "PredictedRating", "WasImpossible"))
    pred_df['WasImpossible'] = pred_df['WasImpossible'].apply(lambda x: x['was_impossible'])
    return pred_df


def predictions_to_tuple_list(predictions):
    '''
    Creates list of tuples (uid, ii, r_ui, est, was_impossible)
    '''
    return [tuple(x) for x in predictions]


def get_top_n(predictions, n=10, min_rating=3.5):
    """
    Creates top n lists for each user. 
    Parameters:
    predictions (list of tuples (uid, iid, real_rating, est_rating)) - predictions made by recommendation algorithm
    n (int) - size of list
    min_rating - minimal predicted rating for item to be used in top_n
    
    Result
    top_n (dictionary) - top n list for each user
    """
    top_n = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        if (est >= min_rating):
            top_n[uid].append((iid, r_ui, est))
    for uid, ratings in top_n.items():
        ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = ratings[:n]
    return top_n


def get_relevant_items_for_user(uid, ratings_df, min_rating=3.5):
    '''
    Creates list of items user alreardy liked
    Parameters:
    uid - user id
    ratings_df - dataframe with user-item rankings
    min_rating - minimal rating to be included 
    
    '''
    return ratings_df[(ratings_df['AuthorId']==uid) & (ratings_df['Rating']>=min_rating)]["RecipeId"].tolist()
   

def get_relevant_items(predictions, ratings_df, min_ratings=3.5):
    '''
    Creates list of relevant items for each user
    Parameterers:
    predictions (list of tuples (uid, iid, real_rating, est_rating)) - predictions made by recommendation algorithm
    ratings_df - dataframe with user-item rankings
    min_rating - minimal rating to be included
    
    Result
    Dictionary with list of relevant items for each user
    '''
    relevant_items = defaultdict(list)
    unique_users = set([uid for uid, iid, r_ui, est, _ in predictions])
    for uid in unique_users:
        relevant_items[uid] = get_relevant_items_for_user(uid, ratings_df, min_rating=min_ratings)
    return relevant_items


def map_predictions_to_user(predictions):
    '''
    Creates list of predicted items for each user
    '''
    predictions_per_user = defaultdict(list)
    for uid, iid, r_ui, est, _ in predictions:
        predictions_per_user[uid].append((iid, r_ui, est))
    return predictions_per_user


def precision_per_user(recommendations_per_user: list, relevant_items_per_user: list):
    '''
    Returns precision for one user
    
    Parameters
    recommendations_per_user (list) - list of predictions for one user
    relevant_items_per_user - list of items from user's history
    
    Result
    precision - Precision for one user
    '''
    recommended_and_relevant_items = [item[0] for item in recommendations_per_user if item[0] in relevant_items_per_user]
    precision = len(recommended_and_relevant_items) / len(recommendations_per_user) if len(recommendations_per_user) != 0 else 0
    return precision


def recommender_precision_at_k(recommendations: dict, relevant_items: dict, k=10):
    '''
    Calculates precisions across all users
    
    Parameters:
    recommendations (dict) - dictionary of recommendations for all users
    relevant_items (dict) - dictionary of relevant items for all users
    
    Result:
    precisions (dict) - precision for all users
    '''
    precisions = dict()
    for uid in recommendations.keys():
        if len(recommendations[uid]) > k:
            recommended_items_for_user = recommendations[uid][:k]
        else:
            recommended_items_for_user = recommendations[uid]
        
        precisions[uid] = precision_per_user(recommended_items_for_user, relevant_items[uid])
    return precisions


def recall_per_user(recommendations_per_user: list, relevant_items_per_user: list):
    '''
    Returns recall for one user
    
    Parameters
    recommendations_per_user (list) - list of predictions for one user
    relevant_items_per_user - list of items from user's history
    
    Result
    recall - recall for one user
    '''
    recommended_and_relevant_items = [item[0] for item in recommendations_per_user if item[0] in relevant_items_per_user]
    recall = len(recommended_and_relevant_items) / len(relevant_items_per_user) if len(relevant_items_per_user) != 0 else 0
    return recall


def recommender_recall_at_k(recommendations: dict, relevant_items: dict, k=10):
    '''
    Calculates recalls across all users
    
    Parameters:
    recommendations (dict) - dictionary of recommendations for all users
    relevant_items (dict) - dictionary of relevant items for all users
    
    Result:
    recalls (dict) - precision for all users
    '''
    recalls = dict()
    for uid in recommendations.keys():
        if len(recommendations[uid]) > k:
            recommended_items_for_user = recommendations[uid][:k]
        else:
            recommended_items_for_user = recommendations[uid]

        recalls[uid] = recall_per_user(recommended_items_for_user, relevant_items[uid])
    return recalls


def average_precision_at_k(recommendations_per_user: list, relevant_items_per_user: list, k=10):
    '''
    Calculates avarage precision@k for one user
    Parameters:
    recommendations_per_user (list) - list of predictions for one user
    relevant_items_per_user - list of items from user's history

    Result
    apk - average precision at k for user
    '''
    if len(relevant_items_per_user) == 0:
        return 0.0
    
    if len(recommendations_per_user) > k:
            recommendations_per_user = recommendations_per_user[:k]
    hits = 0.0
    precision_sum = 0.0
    for i, item in enumerate(recommendations_per_user):
        if item[0] in relevant_items_per_user:
            hits += 1.0
            precision_sum += hits / (i + 1.0)
    apk = precision_sum / min(len(recommendations_per_user), k)
    return apk


def recommender_map(recommendations: dict, relevant_items: dict, k: int):
    '''
    Calculates mean average precision for recommender system at k
    
    Parameters:
    recommendations (dict) - dictionary of recommendations for all users
    relevant_items (dict) - dictionary of relevant items for all users
    k - length of top k list
    Result:
    MAP@k
    '''
    apks = []
    for uid in recommendations.keys():
        apks.append(average_precision_at_k(recommendations[uid], relevant_items[uid], k=k))
    return np.mean(apks)


def hit_rate(recommendations: dict, left_out_predictions: list):
    '''
    Calculates hit rate for all users - number of users with at least one good prediction in top_n divided by number of all users 
    
    Parameters:
    recommendations - dictionary of predicted items for each user
    left_out_predictions -   list of all ratings left in the test set
    
    Result:
    hit rate
    '''
    hits = 0
    total = 0
    for (left_out_user, left_out_item, rating) in left_out_predictions:
        hit = False
        for item, _, est_rating in recommendations['uid']:
            if (left_out_item == item):
                hit = True
                break
        if(hit):
            hits += 1
    return hits / len(left_out_predictions)


def ARHR(recommendations: dict, left_out_predictions: list):
    '''
    Calculates average reciprocal hit rank - similar to hit rate, but takes into consideration rank in the recommended list
    
    Parameters:
    recommendations - dictionary of predicted items for each user
    left_out_predictions -   list of all ratings left in the test set
    
    Result:
    arhr
    '''
    hits = 0
    for (left_out_user, left_out_item, rating) in left_out_predictions:
        rank = 0
        hit_rank = 0
        for item, _, _ in recommendations['uid']:
            rank += 1
            if (left_out_item == item):
                hit_rank = rank
                break
        if(hit_rank > 0):
            hits += 1.0 / hit_rank
    return hits / len(left_out_predictions)


def user_coverage(top_n: dict, number_of_users: int, min_rating=3.5):
    '''
    What percent of users have 1 good prediction
    
    Parameters
    top_n - dictionary of predicted items for each user
    number_of_users - number of all users in the system
    min_rating - minimum rating for item to be considered
    
    '''
    hits = 0
    for uid in top_n.keys():
        hit = False
        for item, _, est  in top_n[uid]:
            if (est > min_rating):
                hit = True
                break
        if(hit):
            hits += 1
    return hits / number_of_users


def item_coverage(top_n: dict, number_of_items: int, min_rating=3.5):
    '''
    Calculates what percantage of items' catalog can be recommended by algorithm
    
    Parameters:
    top_n - dictionary of predicted items for each user
    number_of_users - number of all users in the system
    min_rating - minimum rating for item to be considered
    '''
    items = []
    for value in top_n.values():
        for pairs in value:
            items.append(pairs[0])
    unique_items = set(items)
    return len(unique_items) / number_of_items


def novelty(top_n, ranking):
    '''
    Calculates average popularity of recommended items in top_n lists across all users
    
    Parameters:
    top_n (dict) - dictionary of predicted items for each user
    ranking (dict) - popularity rank for each item
    '''
    n = 0
    total = 0
    for uid in top_n.keys():
        for rating in top_n[uid]:
            item_id = rating[0]
            rank = ranking[item_id]
            total += rank
            n += 1
    return total / n


def diversity(top_n, item_vectors):
    '''
    Calculates diversity of recommended items in top_n lists
    
    Parameters:
    top_n (dict) - dictionary of predicted items for each user
    item_vectors - 
    '''
    n = 0
    total = 0
    for uid in top_n.keys():
        pairs = itertools.combinations(top_n[uid], 2)
        for pair in pairs:
            item1 = pair[0][0]
            item2 = pair[1][0]
            # compute similarity between item1 & item2
            # similarity = 
            # total += similarity
            # n += 1
    total_similarity = total / n
    diversity = 1 - total_similarity
    return diversity


def DCG(top_n: dict, relevant_items: dict):
    ''' 
    Calculates discounted cumulative gain
    
    Parameters:
    top_n (dict) - dictionary of predicted items for each user
    relevant_items (dict) - dictionary of relevant items for all users
    '''
    def relevance_function(is_relevant):
        return 2**int(is_relevant) - 1
    score = 0
    for user, items in top_n.items():
        rank = 0
        for item in items:
            is_relevant = is_item_relevant_for_user(item, user, relevant_items)
            score += relevance_function(is_relevant) / math.log2(rank + 1)
            rank += 1
    return score / len(top_n.keys())


def get_popularity_ranks(reviews_per_recipe):
    rankings = defaultdict(int)
    rank = 1
    for item_id, count in sorted(reviews_per_recipe.items(), key=lambda x: x[1], reverse=True):
        rankings[item_id] = rank
        rank += 1
    return rankings


def is_item_relevant_for_user(item, user, relevant_items: dict):
    return item in relevant_items['uid']