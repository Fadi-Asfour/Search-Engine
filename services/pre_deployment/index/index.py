import os

import pandas as pd
from nltk import corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List

from utils.functions import save_tfidf_data


class TFIDFVectorizerService:
    def __init__(self, tokenizer, preprocessor) -> None:
        self.vectorizer = TfidfVectorizer(tokenizer=tokenizer, preprocessor=preprocessor)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus.values())
        save_tfidf_data(self.tfidf_matrix, self.vectorizer)

    def build_model(self, documents: List[str]) -> pd.DataFrame:
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return pd.DataFrame.sparse.from_spmatrix(tfidf_matrix, columns=self.vectorizer.get_feature_names_out(), index=corpus.keys())

