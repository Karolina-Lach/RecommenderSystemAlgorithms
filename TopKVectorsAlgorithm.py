from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from sentence_transformers import util
import torch
from surprise.prediction_algorithms.predictions import PredictionImpossible
import heapq


class TopKVectorsAlgorithm(AlgoBase):
    def __init__(self, top_similar_dict, vectors_dict, k=40, verbose=True):
        self.k = k
        self.top_vectors_dict = top_similar_dict
        slef.vectors_dict = vectors_dict
        
        
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
        
        i_recipe_id = self.trainset.to_raw_iid(i)
        if i_recipe_id not in self.top_similar_dict.keys():
            raise PredictionImpossible('No similar recipes for item')
            
        neighbours = []
        for rating in self.trainset.ur[u]:
            other_recipe_id = self.trainset.to_raw_iid(rating[0])
            
            found=False
            for recipe, sim in self.top_similar_dict[i_recipe_id]:
                if recipe == other_recipe_id:
                    neighbours.append((sim, rating[1]) )
                    found = True
                    break
            if not found:
                    tensor1 = torch.tensor(self.vectors_dict[trainset.to_raw_iid(rating[0])], dtype=torch.float)
                    tensor2 = torch.tensor(self.vectors_dict[i_recipe_id], dtype=torch.float)
                                       
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