from typing import List

import uvicorn
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service.text_preprocessing.text_preprocessor import TextPreprocessing
from user_vector.user_vector import VectorDBHelper
from utils_functions.dataset_load import DatasetLoader
from utils_functions.dataset_type_manager import DatasetTypeManager
from utils_functions.functions import FilesFunctions
import utils_functions.custom_functions
from utils_functions.custom_functions import preprocess_text, custom_tokenizer
from utils_functions.query_vector_function import query_vector
from utils_functions.string_manager import base_host, document_vectors_antique, document_vectors_clinic, \
    word2vec_model_clinic, word2vec_model_antique
from utils_functions.vectorize_documents import vectorize_documents
from word_embedding.word_embedding import WordEmbeddingAntique

corpusClinic = DatasetLoader().load_clinicaltrials()  # TODO:

corpusAntique = DatasetLoader().load_antique()


class QueryMatching:
    def __init__(self, matrix_filename: str, model_filename: str, corpus: dict):
        self.tfidf_matrix = FilesFunctions.load_file(matrix_filename)
        self.tfidf_model = FilesFunctions.load_file(model_filename)
        self.preprocessor = TextPreprocessing()
        self.corpus = corpus

    def process_query(self, query: str):
        preprocessed_query = query  # Or self.preprocessor.preprocess(query) if preprocessing is needed
        query_vector = self.tfidf_model.transform([preprocessed_query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked_doc_indices = cosine_similarities.argsort()[::-1]
        return ranked_doc_indices, cosine_similarities

    def process_query_word_embedding(self, dataset_name, query_vector):
        if dataset_name == "antique":
            documents_vectors = FilesFunctions.load_file(document_vectors_antique)
        else:
            documents_vectors = FilesFunctions.load_file(document_vectors_clinic)

        similarities = cosine_similarity(documents_vectors, query_vector).flatten()
        top_10_indices = similarities.argsort()[-10:][::-1]
        return [list(self.corpus.keys())[index] for index in top_10_indices]

    def process_query_user_vectors(self, query_text, dataset_name, n_results=10) -> List[str]:
        if dataset_name == "antique":
            word2vec_model = Word2Vec.load(word2vec_model_antique)
            documents_vectors = FilesFunctions.load_file(document_vectors_antique)
            corpus = corpusAntique
        else:
            word2vec_model = Word2Vec.load(word2vec_model_clinic)
            documents_vectors = FilesFunctions.load_file(document_vectors_clinic)
            corpus = corpusClinic
        # Preprocess documents
        complete_documents: list[dict] = []  # {doc_id: ___,doc_content:___, doc_vector:___ }
        for doc_id, doc_content in corpus.items():
            data = {
                'doc_id': doc_id,
                'doc_content': doc_content,
                'doc_vector': None,
                'preprocessed_text': preprocess_text(doc_content),
            }
            complete_documents.append(data)

        processed_documents = [item['preprocessed_text'] for item in complete_documents]

        # get instance of vector db
        vector_db: VectorDBHelper = VectorDBHelper.get_instance()

        # insert the vectors to the database
        vector_db.insert_vectors(dataset_name, documents_vectors)
        query_vec = query_vector(query_text, word2vec_model).tolist()
        result = vector_db.query_db(dataset_name, query_vec, n_results)
        relevance_scores = [item['id'] for item in result]
        return relevance_scores


class QueryRequest(BaseModel):
    query: str
    dataset_name: str


app = FastAPI()


@app.post("/query")
async def query_dataset(request: QueryRequest):
    try:
        datasetype = DatasetTypeManager(request.dataset_name)
        if request.dataset_name == "antique":
            corpus = corpusAntique
        else:
            corpus = corpusClinic  # Assuming corpusClinic is defined somewhere

        query_matching = QueryMatching(datasetype.tfidf_matrix, datasetype.tfidf_model, corpus)
        ranked_indices, similarities = query_matching.process_query(request.query)

        results = []
        for idx in ranked_indices[:10]:
            doc_id = list(corpus.values())[idx]
            results.append(doc_id)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    base_host = '127.0.0.1'  # Define your base_host
    uvicorn.run("query_matching:app", host=base_host, port=8007, reload=True)
