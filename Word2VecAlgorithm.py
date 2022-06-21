from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from sentence_transformers import util
import torch
from surprise.prediction_algorithms.predictions import PredictionImpossible
import heapq



class Word2VecAlgorithm(AlgoBase):
    def __init__(self, vectors, k=40, verbose=True):
        self.k = k
        self.verbose = True
        self.vectors = vectors
        try:
            self.type = sim_options['type']
        except:
            self.type = None
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_item(i)):
#             if(self.verbose):
#                 print('raise for ', u, ' ', i)
            raise PredictionImpossible('Item is unknown')
        if not (self.trainset.knows_user(u)):
            raise PredictionImpossible('User is unknown')
        
        neighbours = []
        for rating in self.trainset.ur[u]:
            tensor1 = torch.tensor(self.vectors[trainset.to_raw_iid(rating[0])], dtype=torch.float)
            tensor2 = torch.tensor(self.vectors[trainset.to_raw_iid(i)], dtype=torch.float)
                                       
            sim = util.pytorch_cos_sim(tensor1, tensor2)[0][0].item()
            neighbours.append((sim, rating[1]))
        
        # Top k most similar ratings:
        k_neighbors = heapq.nlargest(self.k, neighbours, key=lambda t: t[0])
        
        # Avg score of k neigbors weighted by user ratings
        #print('Total similarity')
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating

        
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal

        return predictedRating
    
    
def top_k_recommendations_word2vec(top_similar_dict: dict, user_history: list, k: int):
    top_similar_recipes = []
    for history_recipe in user_history:
        top_similar_recipes.append(top_similar_dict[history_recipe])
        
    k_neighbors = heapq.nlargest(k, top_similar_recipes, key=lambda t: t[1])
    return k_neighbors