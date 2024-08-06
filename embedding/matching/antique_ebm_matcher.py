import os
from typing import List

import numpy as np
from gensim.models import Word2Vec
from numpy import ndarray

from service.text_preprocessing.text_preprocessor import TextPreprocessing
from user_vector.user_vector import VectorDBHelper
from utils_functions.string_manager import antique_embedding_vector_db_path, antique_embedding_model_path


class AntiqueEmbMatcher:

    def __init__(self, text_processor: TextPreprocessing, n_result: int = 10):
        vectors_storage_path: str = antique_embedding_vector_db_path
        self.vector_db_instance = VectorDBHelper.get_instance(db_path=vectors_storage_path)
        self.text_processor = text_processor
        self.model_name = 'antique'
        self.n_result = n_result

        model_storage_path: str = antique_embedding_model_path
        self.model: Word2Vec = Word2Vec.load(model_storage_path)

        self.vector_size = self.model.vector_size
        # self.n_results = 0

    def match(self, text: str):
        print("Query: " + text)

        processed_query: List[str] = self.text_processor.process_text(text)
        query_embeddings: List = self.__vectorize_query(processed_query).tolist()

        results = self.vector_db_instance.query_db(
            self.model_name,
            query_embeddings,
            n_results=self.n_result
        )

        return results

    def __vectorize_query(self, query_words: list[str]) -> ndarray:

        query_vectors = [self.model.wv[word] for word in query_words if word in self.model.wv]

        if query_vectors:
            query_vec = np.mean(query_vectors, axis=0)
        else:
            query_vec = np.zeros(self.vector_size)

        return query_vec
