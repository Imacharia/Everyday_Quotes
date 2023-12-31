import requests
import pandas as pd
from bs4 import BeautifulSoup
import scrapy 
#from pathlib import path

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from langdetect import detect
from googletrans import Translator

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def translate_to_english(text):
    """
Translates the given text to English if it is not already in English.

Args:
    text (str): The text to be translated.

Returns:
    str: The translated text in English, or the original text if it is already in English.

Example:
    ```python
    translated_text = translate_to_english("Bonjour")
    print(translated_text)  # Output: "Hello"
    ```
"""
    try:
        # Detect the language of the text
        lang = detect(text)

        if lang == 'en':
            return text
        translator = Translator()
        return translator.translate(text, src=lang, dest='en').text
    except:
        return text 
    

def preprocess_text(text):
    """
Preprocesses the given text by converting it to lowercase, tokenizing it, removing stopwords, and joining the tokens back into text.

Args:
    text (str): The text to be preprocessed.

Returns:
    str: The preprocessed text.

Example:
    ```python
    preprocessed_text = preprocess_text("This is a sample text.")
    print(preprocessed_text)  # Output: "sample text"
    ```
"""

    # Lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Join tokens back into text
    return ' '.join(tokens)