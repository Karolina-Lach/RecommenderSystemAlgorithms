from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from sentence_transformers import util
import torch
from surprise.prediction_algorithms.predictions import PredictionImpossible
import heapq



class Word2VecMatrixAlgorithm(AlgoBase):
    '''
    1) Calculate matrix during train:
        index - inner item ids
    2) Use pre-calculated matrix:
        index - recipe_id_to_pos (first change inner id to raw id, and get position from recipe id)
    3) Use submatrix from pre-calculated matrix:
        index - inner item ids
    '''
    def __init__(self, vectors, matrix=None, recipe_id_to_pos=None, create_subset=False, k=40, verbose=True):
        self.k = k
        self.verbose = True
        self.vectors = vectors
        self.matrix = matrix
        self.recipe_id_to_pos = recipe_id_to_pos
        self.create_subset = create_subset
        if matrix == None:
            self.type = "calculate matrix"
        else:
            if create_subset:
                self.type = "subset"
            else:
                self.type = "precalculated matrix"

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        if matrix != None:
            if self.create_subset:
                self.similarites = create_similarity_submatrix(self.matrix, 
                                                               self.recipe_id_to_pos_in_matrix, 
                                                               trainset)
            else:
                self.similarites = self.matrix
            del(self.matrix)
        else:
            calculate_sim_matrix(trainset)
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
            if self.type == "precalculated matrix":
                recipe_id1 = self.trainset.to_raw_iid(i)
                recipe_id2 = self.trainset.to_raw_iid(rating[0])
                
                sim = self.similarities[self.recipe_id_to_pos[recipe_id1],
                                       self.recipe_id_to_pos[recipe_id2]]
            else:
                sim = self.similarites[i, rating[0]]
            neighbours.append( (sim, rating[1]) )
        
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
    
    def calculate_sim_matrix(trainset):
        if(self.verbose):
            print("Computing content-based similarity matrix...")
    
            #self.similarities = mat.Matrix(self.trainset.n_items)
            print('Number of items in trainset: ', trainset.n_items)   
            self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items))
            for thisRating in range(trainset.n_items):
                if (thisRating % 1000 == 0):
                        print(thisRating, " of ", self.trainset.n_items)
                for otherRating in range(thisRating+1, self.trainset.n_items):
                    thisRecipeID = int(self.trainset.to_raw_iid(thisRating))
                    otherRecipeID = int(self.trainset.to_raw_iid(otherRating))
                    #if(self.verbose):
                    #    print(thisRecipeID, "  ", otherRecipeID)
                    tensor1 = torch.tensor(self.vectors[thisRecipeID], dtype=torch.float)
                    tensor2 = torch.tensor(self.vectors[otherRecipeID], dtype=torch.float)

                    sim = util.pytorch_cos_sim(tensor1, tensor2)[0][0].item()
                    self.similarities[thisRating, otherRating] = sim
                    self.similarities[otherRating, thisRating] = sim
        
        
        if(self.verbose):
            print("...done.")
            
    def create_similarity_submatrix(matrix, recipe_id_to_pos_in_matrix, trainset):
        similarities = np.zeros((trainset.n_items, trainset.n_items))
        for thisRating in range(trainset.n_items):
            if (thisRating % 1000 == 0):
                print(thisRating, " of ", trainset.n_items)
            for otherRating in range(thisRating+1, trainset.n_items):
                thisRecipeID = int(trainset.to_raw_iid(thisRating))
                otherRecipeID = int(trainset.to_raw_iid(otherRating))
                pos1 = recipe_id_to_pos_in_matrix[thisRecipeID]
                pos2 = recipe_id_to_pos_in_matrix[otherRecipeID]

                sim = matrix[pos1, pos2]
                similarities[thisRating, otherRating] = sim
                similarities[otherRating, thisRating] = sim
        return similarities