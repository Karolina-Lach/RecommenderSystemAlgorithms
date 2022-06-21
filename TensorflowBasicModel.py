import os
import pprint
import tempfile
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

import pandas as pd
import sampling
import pickle


class RecipeRetrievalModel(tfrs.Model):
    def __init__(self, unique_recipe_ids, unique_user_ids, embedding_dimension, recipes):
        super().__init__()
        self.recipe_model: tf.keras.Model = tf.keras.Sequential([
          tf.keras.layers.StringLookup(
              vocabulary=unique_recipe_ids, mask_token=None),
          tf.keras.layers.Embedding(len(unique_recipe_ids) + 1, embedding_dimension)
        ])
        self.user_model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        
        
        metrics = tfrs.metrics.FactorizedTopK(candidates=recipes.batch(128).map(self.recipe_model))
        task = tfrs.tasks.Retrieval(metrics=metrics)
        self.task: tf.keras.layers.Layer = task
            
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["AuthorId"])
        positive_recipe_embeddings = self.recipe_model(features["RecipeId"])
        
        return self.task(user_embeddings, positive_recipe_embeddings)
    
    
    
class RankingModel(tf.keras.Model):
    
    def __init__(self, unique_user_ids, unique_recipe_ids, embedding_dimension):
        super().__init__()
        embedding_dimension = 32
        
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids)+1, embedding_dimension)
        ])
        
        self.recipe_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_recipe_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_recipe_ids)+1, embedding_dimension)
        ])
        
        # Compute predictions
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)
        ])
        
    def call(self, inputs):
        
        user_id, recipe_id = inputs
        
        user_embedding = self.user_embeddings(user_id)
        recipe_embedding = self.recipe_embeddings(recipe_id)
        
        return self.ratings(tf.concat([user_embedding, recipe_embedding], axis=1))
    
    
    
class RecipeRankingModel(tfrs.models.Model):
    
    def __init__(self, verbose=False):
        super().__init__()
        self._verbose = verbose
        self.ranking_model: tf.keras.Model = RankingModel()
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = [tf.keras.metrics.RootMeanSquaredError()])
            
        
    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model((features["AuthorId"], features["RecipeId"]))
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        if self._verbose:
             print(features)
        labels = features.pop("Rating")
        rating_predictions = self(features)
        
        return self.task(labels=labels, predictions=rating_predictions)