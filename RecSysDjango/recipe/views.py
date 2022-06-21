from django.shortcuts import render
import re
import pandas as pd
import pickle

# Create your views here.

author_id = 564537 #310611 #74280 #2695

def home_view(request, *args, **kwargs):
    # print("Hello")
    s = request.path
    result = re.search('/(.*)/', s)
    try:
        selected_list = result.group(1)
    except:
        selected_list = ""


    recipes = pd.read_parquet("recipes.parquet")
    # author_id = 7802 #2178 #2695
    recommendations = {}
    type = ""
    if selected_list == "1":
        print("Content")
        type = "Filtrowanie oparte na treści"
        with open('knn_recommendations_sample500.obj', 'rb') as pickle_file:
            recommendations = pickle.load(pickle_file)

    elif selected_list == "2":
        print("Collaborative")
        type = "Filtrowanie kolaboratywne"
        with open('svd_recommendations_sample500.obj', 'rb') as pickle_file:
            recommendations = pickle.load(pickle_file)
    elif selected_list == "3":
        print("tfrs")
        type = "Filtrowanie hybrydowe"
        with open('tfrs_recommendations_sample500.obj', 'rb') as pickle_file:
            recommendations = pickle.load(pickle_file)
    else:
        print("Empty")
    try:
        recommendations = recommendations[author_id][:10]
    except:
        recommendations = []

    recipes_subset = recipes[recipes.RecipeId.isin(recommendations)]
    recipes_view = create_recipes_list(recipes_subset)

    context = {
        'type': type,
        'recipes_list': recipes_view
    }

    return render(request, "home.html", context)


def history_view(request, *args, **kwargs):
    recipes = pd.read_parquet("recipes.parquet")
    ratings = pd.read_parquet("ratings.parquet")


    history = list(ratings[ratings.AuthorId == author_id]['RecipeId'])
    historic_recipes = recipes[recipes.RecipeId.isin(history)]

    recipes_view = create_recipes_list(historic_recipes)

    context = {
        'recipes_list': recipes_view
    }

    return render(request, "history.html", context)


def create_recipes_list(recipes_subset):
    names = list(recipes_subset['Name'])

    ids = list(recipes_subset['RecipeId'])
    ids = [int(x) for x in ids]

    images_to_show = []
    image_placeholder = 'https://www.russorizio.com/wp-content/uploads/2016/07/ef3-placeholder-image.jpg'
    for images_per_recipe in list(recipes_subset['Images']):
        if len(images_per_recipe) == 0:
            images_to_show.append(image_placeholder)
        else:
            images_to_show.append(images_per_recipe[0])

    recipes_view = zip(ids, names, images_to_show)
    return recipes_view