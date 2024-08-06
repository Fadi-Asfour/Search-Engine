import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from starlette.middleware.cors import CORSMiddleware

from utils_functions.dataset_load import DatasetLoader
from utils_functions.functions import FilesFunctions
from utils_functions.string_manager import document_vectors_antique, document_vectors_clinic, word2vec_model_clinic, \
    word2vec_model_antique, base_host

# Initialize the app
app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load datasets
corpusClinic = DatasetLoader().load_clinicaltrials()
corpusAntique = DatasetLoader().load_antique()


# Define the request model
class QueryRequest(BaseModel):
    query: str
    dataset_name: str


# WordEmbeddingAntique class
class WordEmbedding:
    def __init__(self, vector_size=100, sg=1, epochs=35):
        self.vector_size = vector_size
        self.sg = sg
        self.epochs = epochs
        self.word2vec_model = None

    def load_model(self, model_path: str):
        self.word2vec_model = Word2Vec.load(model_path)

    def documents_vectors(self, documents: List[str]) -> List[np.ndarray]:
        document_vectors = []
        for document in documents:
            vectors = [self.word2vec_model.wv[token] for token in document.split() if token in self.word2vec_model.wv]
            if vectors:
                document_vectors.append(np.mean(vectors, axis=0))
            else:
                document_vectors.append(np.zeros(self.vector_size))
        return document_vectors

    def query_vector(self, query: str) -> np.ndarray:
        tokens = query.split()
        vectors = [self.word2vec_model.wv[token] for token in tokens if token in self.word2vec_model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)


# Endpoint for querying with word embeddings
@app.post("/query")
async def query_with_word_embedding(request: QueryRequest):
    try:
        if request.dataset_name == "antique":
            corpus = corpusAntique
            model_path = word2vec_model_antique
            document_vectors_path = document_vectors_antique
        else:
            corpus = corpusClinic
            model_path = word2vec_model_clinic
            document_vectors_path = document_vectors_clinic

        # Load the word2vec model and document vectors
        word_embedding = WordEmbedding()
        word_embedding.load_model(model_path)
        document_vectors = FilesFunctions.load_file(document_vectors_path)

        # Process the query to get its vector representation
        query_vec = word_embedding.query_vector(request.query).reshape(1, -1)
        similarities = cosine_similarity(document_vectors, query_vec).flatten()

        # Get the top 10 most similar documents
        top_10_indices = similarities.argsort()[-10:][::-1]
        results = [list(corpus.values())[index] for index in top_10_indices]

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run("matching_with_embedding_service:app", host=base_host, port=8007, reload=True)
