from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from sentence_transformers import util
import torch
from surprise.prediction_algorithms.predictions import PredictionImpossible
import heapq



class KnnMatrixAglorithm(AlgoBase):
    def __init__(self, matrix, recipe_id_to_pos, vectors, k=40, verbose=True):
        self.k = k
        self.verbose = verbose
        self.matrix = matrix
        self.recipe_id_to_pos = recipe_id_to_pos
        self.vectors = vectors
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.items_not_in_matrix = {}
        for item in trainset.all_items():
            raw_item = trainset.to_raw_iid(item)
            self.items_not_in_matrix[raw_item] = {}
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_item(i)):
#             if(self.verbose):
#                 print('raise for ', u, ' ', i)
            raise PredictionImpossible('Item is unknown')
        if not (self.trainset.knows_user(u)):
            raise PredictionImpossible('User is unknown')
        
        item_recipe_id = self.trainset.to_raw_iid(i)
        is_in_matrix = True
        neighbours = []
        calculated = 0
        from_matrix = 0
        from_dict = 0
        for rating in self.trainset.ur[u]:
            recipe_id = self.trainset.to_raw_iid(rating[0])
            
            try:
                item_pos = self.recipe_id_to_pos[item_recipe_id]
            except:
#                 print(f"{item_recipe_id} error")
                is_in_matrix = False
            
            try:
                recipe_pos = self.recipe_id_to_pos[recipe_id]
            except:
#                 print(f"{recipe_id} error")
                is_in_matrix = False
                
            if is_in_matrix:
                sim = self.matrix[item_pos, recipe_pos]
                from_matrix += 1
            else:
#                 print(item_recipe_id, " ", recipe_id)
                try:
                    if item_recipe_id > recipe_id:
                        sim = self.items_not_in_matrix[item_recipe_id][recipe_id]
                    else:
                        sim = self.items_not_in_matrix[recipe_id][item_recipe_id]
                    from_dict += 1
                except:
                    tensor1 = torch.tensor(self.vectors[item_recipe_id], dtype=torch.float)
                    tensor2 = torch.tensor(self.vectors[recipe_id], dtype=torch.float)

                    sim = util.pytorch_cos_sim(tensor1, tensor2)[0][0].item()
                    
                    if item_recipe_id > recipe_id:
                        self.items_not_in_matrix[item_recipe_id][recipe_id] = sim
                    else:
                        self.items_not_in_matrix[recipe_id][item_recipe_id] = sim
                    calculated += 1   
#                     print(f'Calculating similarity for {recipe_id} and {item_recipe_id}')
                
            neighbours.append((sim, rating[1]))
        k_neighbours = heapq.nlargest(self.k, neighbours, key=lambda t: t[0])
        
        sim_total = 0
        weighted_sum = 0
        for sim_score, rating in k_neighbours:
            if sim_score > 0:
                sim_total += sim_score
                weighted_sum += sim_score * rating
                
        if sim_total == 0:
            raise PredictionImpossible('No neighbours')
            
        predicted_rating= weighted_sum / sim_total
        
#         print(f"Calculated {calculated}, from matrix: {from_matrix}, from dict: {from_dict}")
        return predicted_rating