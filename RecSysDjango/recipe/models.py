from django.db import models


class Category(models.Model):
    name = models.CharField(max_length=120)


class Keyword(models.Model):
    name = models.CharField(max_length=120)


class Product(models.Model):
    name = models.CharField(max_length=500)


class Recipe(models.Model):
    recipe_id = models.IntegerField()
    name = models.CharField(max_length=500)


