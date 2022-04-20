import tensorflow as tf
import tensorflow_recommenders as tfrs

from tensorflow.keras.layers import Flatten   # to flatten the input data
from tensorflow.keras.layers import Dense     # for the hidden layer


    
    
class UserModel(tfrs.models.Model):
    
    def __init__(self, 
                 unique_user_ids,
#                  timestamp_buckets,
                 verbose=False):
        
        super().__init__()
        self._verbose = verbose
        if(self._verbose):
            print("USER MODEL INIT")
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32)
        ])
        
#         self.timestamp_embedding = tf.keras.Sequential([
#             tf.keras.layers.Discretization(timestamp_buckets.tolist()),
#             tf.keras.layers.Embedding(len(timestamp_buckets)+1, 32),
#         ])
        
#         self.normalized_timestamp = tf.keras.layers.Normalization(axis=None)
#         self.normalized_timestamp.adapt(timestamp_buckets)
        
    def call(self, inputs):
        if(self._verbose):
            print("User model call")
            print("INPUTS: ", inputs)
        return tf.concat([
            self.user_embedding(inputs["AuthorId"]),
#             self.timestamp_embedding(inputs["Timestamp"]),
#             tf.reshape(self.normalized_timestamp(inputs["Timestamp"]), (-1,1)),
        ], axis=1)
        
        
class RecipeModel(tfrs.models.Model):
    
    def __init__(self, 
                 unique_recipe_ids,
                 recipes_dataset,
                 verbose=False):
        super().__init__()
        max_tokens = 10_000
        embedding_dim=32
        
        self._verbose = verbose
        if(verbose):
            print("RECIPE MODEL INIT")
        self.recipe_id_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_recipe_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_recipe_ids)+1, 32)
        ])
        
        self.ingredients_vectorizer = tf.keras.layers.TextVectorization(max_tokens = max_tokens)
        
        self.ingredients_text_embedding = tf.keras.Sequential([
            self.ingredients_vectorizer,
            tf.keras.layers.Embedding(input_dim=max_tokens, output_dim=embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D()
        ])
        
        self.ingredients_vectorizer.adapt(recipes_dataset.map(lambda x: x['Ingredients']))
        
    def call(self, inputs):
        if(self._verbose):
            print("Recipe model call")
            print("INPUTS: ", inputs)
        return tf.concat([
            self.recipe_id_embedding(inputs["RecipeId"]),
            self.ingredients_text_embedding(inputs["Ingredients"])
        ], axis=1)
    
    
    
class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""
    def __init__(self, 
                 layer_sizes,
                 unique_user_ids,
#                  timestamp_buckets,
                 verbose=False):
        """Model for encoding user queries.
        Args:
            layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
        """
        
        super().__init__()

        if(verbose):
            print("Query model init")
            
        self._verbose = verbose
        # We first use the user model for generating embeddings.
#         self.embedding_model = UserModel(unique_user_ids, timestamp_buckets, verbose)
   
        self.embedding_model = UserModel(unique_user_ids, verbose)

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
            
    def call(self, inputs):
        if(self._verbose):
            print("Query model call")
            print("Input: ", inputs)
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
    
    
class CandidateModel(tf.keras.Model):
    """Model for encoding recipes."""
    
    def __init__(self, 
                 layer_sizes, 
                 unique_recipe_ids,
                 recipes_dataset,
                 verbose=False):

        super().__init__()
        if(verbose):
            print("Candidate model init")
        self.embedding_model = RecipeModel(unique_recipe_ids,
                                           recipes_dataset,
                                           verbose)

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))
            
        self._verbose = verbose
    
    def call(self, inputs):
        if(self._verbose):
            print("Candidate model call")
            print("Inputs: ", inputs)
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
    
    
class CombinedModel(tfrs.models.Model):
    
    def __init__(self, layer_sizes, 
                 unique_recipe_ids,
                 recipes_dataset,
                 unique_user_ids,
#                  timestamp_buckets,
                 verbose=False):
        super().__init__()
        if(verbose):
            print("Init combined model")
#         self.query_model = QueryModel(layer_sizes, unique_user_ids, timestamp_buckets)
        self.query_model = QueryModel(layer_sizes, unique_user_ids, verbose)
        self.candidate_model = CandidateModel(layer_sizes, unique_recipe_ids, recipes_dataset, verbose)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=recipes_dataset.batch(1024).map(self.candidate_model),
            ),
        )
        self._verbose = verbose
        
        
    def compute_loss(self, features, training=False):
        if(self._verbose):
            print("Combined model compute loss")
            print("Features: ", features)
        query_embeddings = self.query_model({
            "AuthorId": features["AuthorId"]
        })
        
        recipe_embeddings = self.candidate_model({
            "RecipeId": features["RecipeId"],
            "Ingredients": features["Ingredients"]
        })
        
        return self.task(
            query_embeddings, recipe_embeddings, compute_metrics=not training)