from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
import pandas as pd


class PopularityBasedAlgorithm(AlgoBase):
    def __init__(self, ratings_df):
        AlgoBase.__init__(self)
        self.ratings_df = ratings_df

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

    def estimate(self, u, i):
        raw_user = self.trainset.to_raw_uid(u)
        raw_item = self.trainset.to_raw_iid(i)

        avg_ratings = self.ratings_df[(self.ratings_df.AuthorId!=raw_user) & 
                                    (self.ratings_df.RecipeId==raw_item)].Rating.mean()
        return avg_ratings

    def recommend_item(self, user_id, k=10):
        raw_user = self.trainset.to_raw_uid(user_id)
        popular_items = self.ratings_df[self.ratings_df.AuthorId!=raw_user].groupby(['AuthorId'], 
                                                                                    as_index = False, 
                                                                                    group_keys=False).size()
        popular_items = popular_items.sort_values(by=['size'], ascending=False)[:k]

        return popular_items