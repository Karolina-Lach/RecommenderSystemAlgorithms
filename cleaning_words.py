import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import string
from unidecode import unidecode
import re
import spacy
nlp = spacy.load("en_core_web_sm")


def read_common_words(common_words_path):
    with open(common_words_path, "r") as f:
        common_words = f.readlines()
    common_words = [word.replace('\n', '').lower() for word in common_words]
    return common_words


def transform_common_words(common_words):
    common_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in common_words]
    
#     filtered_words = []
#     for phrase in common_words:
# #         print(phrase)
#         clean_word = []
#         for word in phrase.split():
#             word = re.sub('\d', '', word)
#             word = re.sub(' ', '', word)
#             if word not in stop_words and word.isalpha():
#                 word = word.lower()
#                 word = unidecode(word)
#                 word = lemmatizer.lemmatize(word)
#                 if word:
#                     clean_word.append(word)
#         filtered_words.append(' '.join(clean_word))
    filtered_words = set(common_words)
    filtered_words = list(filtered_words)
    filtered_words.sort(key=lambda x: len(x.split()), reverse=True)
    return filtered_words


def read_common_words_spacy(common_words_path, lemmatizer, stop_words):
    with open(common_words_path, "r") as f:
        common_words = f.readlines()
#     print(common_words)
    common_words = [word.replace('\n', '').lower() for word in common_words]
    common_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in common_words]
    
    filtered_words = []
    for phrase in common_words:
#         print(phrase)
        clean_word = []
        for word in phrase.split():
            word = re.sub('\d', '', word)
            word = re.sub(' ', '', word)
            if word not in stop_words and word.isalpha():
                word = word.lower()
                word = unidecode(word)
                word = lemmatizer.lemmatize(word)
                if word:
                    clean_word.append(word)
        filtered_words.append(' '.join(clean_word))
    filtered_words = set(filtered_words)
    filtered_words = list(filtered_words)
    filtered_words.sort(key=lambda x: len(x.split()), reverse=True)
    return filtered_words


def remove_common_words(list_of_ingredients, common_words):
    '''
    Function removes common words from each string inside the list
    '''
    new_list_of_ingredients = []
    for ing in list_of_ingredients:
#         ing = re.sub('\d', '', ing)
#         for word in common_words:
#             pattern = r'(?<![a-zA-Z])' + word + '(?![a-z-Z])'
# #             if re.match(pattern, ing):
#             ing = re.sub(pattern, '', ing)
#             ing = re.sub(' +', ' ', ing)
        new_list_of_ingredients.append(remove_from_single_phrase(ing, common_words))
    return new_list_of_ingredients


def remove_from_single_phrase(phrase, common_words):
    phrase = re.sub('\d', '', phrase)
    for word in common_words:
        pattern = r'(?<![a-zA-Z])' + word + '(?![a-z-Z])'
#             if re.match(pattern, ing):
        phrase = re.sub(pattern, '', phrase)
        phrase = re.sub(' +', ' ', phrase)
        phrase = phrase.lstrip()
        phrase = phrase.rstrip()
    return phrase


def clean_phrases_in_list_spacy(ing_list):
    '''
    Cleaning words inside list with spacy module
        1. to lower
        2. remove unicode chars
        3. remove punctuation
        4. remove numbers
        5. lemmatize
        6. remove stopwords
        7. remove common words (i.e. fresh, chopped, cooked)
    '''
    all_ingredients_in_recipe = []
    for ing in ing_list:
        ing = clean_single_phrase(ing)
        if ing:
            all_ingredients_in_recipe.append(ing)
    return all_ingredients_in_recipe


def clean_single_phrase(ing, nlp=nlp):
    filtered_ingredient = []
    ing = ing.replace('/', ' ')
    ing = ing.replace('-', ' ')
    ing = ing.replace('&', ' ')
    ing = ing.replace(';', ' ')
    ing = ing.replace('(', ' ')
    ing = ing.lower()
    ing = unidecode(ing)
    ing = ing.translate(str.maketrans('', '', string.punctuation))
    ing = re.sub('\d', '', ing)
    if ing != '':
        doc = nlp(ing)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        for word in lemma_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                if lexeme:
                    filtered_ingredient.append(word) 
    ing = ' '.join(filtered_ingredient)
    ing = re.sub(' +', ' ', ing)
    return ing


def clean_phrases_in_list(ingredients_list, lemmatizer, stop_words):
    '''
        Cleaning words inside list with nltk module
        1. to lower
        2. remove punctuation & double spaces
        3. remove unicode chars
        5. lemmatize
        6. remove stopwords & not alphanumeric chars
    '''
    filtered_ingredients = []
    for ingredient in ingredients_list:
        clean_ingredient = ""
        words = word_tokenize(ingredient.lower())
        for word in words:
            
            word = word.translate(str.maketrans('', '', string.punctuation))
            word = re.sub(' ', '', word)
            if word not in stop_words and word.isalpha():
                word = word.lower()
                word = unidecode(word)
                word = lemmatizer.lemmatize(word)
                if clean_ingredient == "":
                    clean_ingredient = word
                else:
                    clean_ingredient = clean_ingredient + " " + word       
        filtered_ingredients.append(clean_ingredient)
    return sorted(filtered_ingredients)


def convert_list_from_dict(list_phrases, dictionary_phrases):
    filtered_phrases = []
    for phrase in list_phrases:
        if phrase in dictionary_phrases:
            filtered_phrase = dictionary_phrases[phrase]
            if filtered_phrase:
                filtered_phrases.append(filtered_phrase)
    return sorted(filtered_phrases)



def clean_keyword(word, nlp=nlp):
#     print(re.match('([/])', word))
#     if re.match('([/])*', word):
#         word = re.sub(r'([/])*', ' ', word)
    if word is not None:
        word = word.replace('/', ' ')
        word = word.lower()
        word = unidecode(word)
        word = word.translate(str.maketrans('', '', string.punctuation))

        doc = nlp(word)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
    #     print(lemma_list)
        word = ' '.join(lemma_list)
        word = word.lstrip().rstrip()

    #     print(word)
        if re.match('([\d])', word):
    #         print('o')
            word = re.sub(r'([\d]*) ', '\g<1>', word)
    #         print(word)

        word = re.sub(' +', ' ', word)
        return word
    return ""


def make_clean_keywords(keywords_list):
    if len(keywords_list) != 0:
        if keywords_list[0] is not None:
            clean_keywords_list = []
            for word in keywords_list:
                clean_keywords_list.append(clean_keyword(word))
            return sorted(clean_keywords_list)
    return []