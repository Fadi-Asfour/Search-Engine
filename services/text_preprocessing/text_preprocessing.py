from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from typing import List
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from spellchecker import SpellChecker
import string

app = FastAPI()


class TextPreprocessor:
    def __init__(self) -> None:
        self.tokenizer = word_tokenize
        self.stopwords_tokens = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.spell_checker = SpellChecker()

    def tokenize(self, text: str) -> List[str]:
        return self.tokenizer(text)

    @staticmethod
    def to_lower(tokens: List[str]) -> List[str]:
        return [token.lower() for token in tokens]

    @staticmethod
    def remove_punctuation(tokens: List[str]) -> List[str]:
        return [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords_tokens]

    def stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatizing(self, tokens: List[str]) -> List[str]:
        pos_tags = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(tag)) for token, tag in pos_tags]

    def correct_spelling(self, tokens: List[str]) -> List[str]:
        misspelled = self.spell_checker.unknown(tokens)
        return [self.spell_checker.correction(token) if token in misspelled else token for token in tokens]

    @staticmethod
    def get_wordnet_pos(tag: str) -> str:
        tag = tag[0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text: str) -> str:
        # Tokenize
        tokens = self.tokenize(text)
        # Convert to lowercase
        tokens = self.to_lower(tokens)
        # Correct spelling
        tokens = self.correct_spelling(tokens)
        # Remove punctuation
        tokens = self.remove_punctuation(tokens)
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        # Stemming
        tokens = self.stemming(tokens)
        # Lemmatizing
        tokens = self.lemmatizing(tokens)
        # Join tokens back to string
        return ' '.join(tokens)


class TextRequest(BaseModel):
    text: str


@app.post("/preprocess")
def preprocess_route(request: TextRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="No text provided")

    preprocessor = TextPreprocessor()
    result = preprocessor.preprocess(request.text)

    return {"processed_text": result}
