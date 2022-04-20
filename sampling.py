import pandas as pd
import numpy as np

def get_ratings_in_range(df, ratings, col_name='Rating'):
    '''
    Returns dataframe slice with ratings from specified range.
    Can be used for excluding ratings with 0.
    '''
    return df[df[col_name].isin(ratings)].copy()
    
    
def get_rating_with_min_number(df, min_number_of_ratings, col_name='AuthorId'):
    '''
    Returns dataframe slice for users/recipes that have min number of ratings.
    '''
    return df.groupby(col_name).filter(lambda x : len(x) > min_number_of_ratings).copy()

def get_ratings_with_min_number_list(df, min_number: list):
    author_min = get_rating_with_min_number(df, min_number[0], col_name='AuthorId')
    recipe_min = get_rating_with_min_number(df, min_number[1], col_name='RecipeId')

    return author_min.merge(recipe_min, how='inner')


def get_sample_with_identical_proportions(df, n_samples, col_name='Rating', random_state_sample=13):
    '''
    Returns stratified sample of df
    '''
    counts = df[col_name].value_counts(normalize=True)
    sample_df = pd.DataFrame()
    for index, value in counts.items():
        df_ = df.groupby(col_name).get_group(index)
        df_ = df_.sample(int(n_samples * value), replace=False, random_state=random_state_sample)
        sample_df = sample_df.append(df_)
    return sample_df


def get_sample_equal_classes(df, n_samples, col_name='Rating', random_state_sample=13):
    '''
    Returns dataframe slice with equal number of entries for each class
    '''
    n = min(n_samples, df[col_name].value_counts().min())
    df_ = df.groupby(col_name).apply(lambda x: x.sample(n, replace=False, random_state=random_state_sample))
    df_.index = df_.index.droplevel(0)
    return df_


def get_sample_without_outliers(df, max_std = 3.0, col_name="AuthorId"):
    ratingsByUser = ratings.groupby('AuthorId', as_index=False).agg({"Rating": "count"})
    ratingsByUser['Outlier'] = (abs(ratingsByUser.Rating - ratingsByUser.Rating.mean()) > ratingsByUser.Rating.std() * max_std)
    ratingsByUser = ratingsByUser.drop(columns=['Rating'])
    combined = ratings.merge(ratingsByUser, on='userId', how='left')
    filtered = combined.loc[combined['Outlier'] == False]
    filtered = filtered.drop(columns=['Outlier'])
    
    return filtered