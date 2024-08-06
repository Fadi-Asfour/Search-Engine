from typing import List

import numpy as np
from gensim.models import Word2Vec

from service.text_preprocessing.text_preprocessor import TextPreprocessing
from user_vector.user_vector import VectorDBHelper
from utils_functions.dataset_load import DatasetLoader
from utils_functions.string_manager import clinical_embedding_model_path, clinical_embedding_vector_db_path


class ClinicalEmbModel:
    def __init__(
            self,
            text_processor: TextPreprocessing,
            vector_size: int = 300,
            epochs: int = 35,
            batch_size: int = 4000,
            workers: int = 4,
    ):
        self.vector_size = vector_size
        self.model_name = 'clinical'
        self.batch_size = batch_size
        self.workers = workers
        self.epochs = epochs

        # initialize data
        self.model: Word2Vec | None = None
        self.model_storage_path: str = clinical_embedding_model_path
        self.vector_db_instance = VectorDBHelper.get_instance(db_path=clinical_embedding_vector_db_path)
        self.text_processor = text_processor
        self.dataset = DatasetLoader().load_clinicaltrials()

    def __process_docs(self) -> list[dict]:
        # processed data
        docs: list[dict] = []

        for doc_id, doc_content in self.dataset.items():
            data = {
                'doc_id': doc_id,
                'doc_content': doc_content,
                'doc_vector': None,
                'preprocessed_text': self.text_processor.process_text(doc_content),
            }
            docs.append(data)

        return docs

    def train_model(self) -> None:

        # Load the pre-processed documents
        documents: list[dict] = self.__process_docs()

        # Extract the processed documents for training the Word2Vec model
        tokenized_docs = [doc['preprocessed_text'] for doc in documents]

        # Initialize Word2Vec model
        self.model = Word2Vec(
            vector_size=self.vector_size,
            workers=self.workers,
            epochs=self.epochs,
            min_count=1,
            sg=1,
        )

        self.model.build_vocab(tokenized_docs)

        self.model.train(tokenized_docs, total_examples=self.model.corpus_count, epochs=self.epochs)

        self.save_model()

        self.save_vector_embeddings(documents)

    def save_model(self) -> None:
        self.model.save(self.model_storage_path)

    def save_vector_embeddings(self, documents: List[dict]) -> None:
        vectorized_documents = self.vectorize_documents(documents)
        self.__store_vectors_to_db(vectorized_documents)

    def vectorize_documents(self, documents: list[dict]) -> list[dict]:

        vectors: list[dict] = []

        for document in documents:
            zero_vector = np.zeros(self.vector_size)
            doc_vector = []
            for token in document['preprocessed_text']:
                if token in self.model.wv:
                    doc_vector.append(self.model.wv[token])
            if doc_vector:
                doc_vector = np.asarray(doc_vector)
                avg_vec = doc_vector.mean(axis=0)
                vec = avg_vec
            else:
                vec = zero_vector

            # create the updated dict
            document['doc_vector'] = vec.tolist()
            vectors.append(document)
        return vectors

    def __store_vectors_to_db(self, vectors: list[dict]):

        # insert the vectors using helper class
        self.vector_db_instance.insert_vectors(
            collection_name=self.model_name,
            chunk_size=self.batch_size,
            vectors=vectors,
        )
