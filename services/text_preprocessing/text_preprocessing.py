import ir_datasets
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import pandas as pd
import pickle
import os
from spellchecker import SpellChecker

# Load the dataset
dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

# Prepare the corpus with a limit of 3000 documents
corpus = {}
counter = 0
for doc in dataset.docs_iter():
    if counter < 3000:
        corpus[doc.doc_id] = doc.title + " " + doc.summary + " " + doc.detailed_description + " " + doc.eligibility
        counter += 1
    else:
        break

documents = list(corpus.values())

# Custom tokenizer
def custom_tokenizer(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return tokens

# Helper functions for text preprocessing
def get_wordnet_pos(tag):
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def correct_sentence_spelling(tokens):
    spell = SpellChecker()
    misspelled = spell.unknown(tokens)
    for i, token in enumerate(tokens):
        if token in misspelled:
            corrected = spell.correction(token)
            if corrected is not None:
                tokens[i] = corrected
    return tokens

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]
    pos_tags = pos_tag(stemmed_words)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(lemmatized_words)
