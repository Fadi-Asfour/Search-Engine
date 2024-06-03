from gensim.models import Word2Vec
import numpy as np
import pickle
from typing import List


class WordEmbeddingAntique:
    def __init__(self, vector_size=100, sg=1, epochs=35):
        self.vector_size = vector_size
        self.sg = sg
        self.epochs = epochs
        self.word2vec_model = None

    def train_model(self, processed_documents: List[str]):
        self.word2vec_model = Word2Vec([doc.split() for doc in processed_documents],
                                       vector_size=self.vector_size, sg=self.sg, epochs=self.epochs)
        self.word2vec_model.save("word2vec_model")

    def load_model(self, model_path: str = "word2vec_model"):
        self.word2vec_model = Word2Vec.load(model_path)



    @staticmethod
    def save_file(file_location: str, content):
        with open(file_location, 'wb') as file:
            pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_file(file_location: str):
        with open(file_location, 'rb') as file:
            loaded_file = pickle.load(file)
        return loaded_file
