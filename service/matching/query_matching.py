import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service.text_preprocessing.text_preprocessor import TextPreprocessing
# from services.text_preprocessing.text_preprocessor import TextPreprocessing
from utils_functions.dataset_load import DatasetLoader
from utils_functions.dataset_type_manager import DatasetTypeManager
from utils_functions.functions import FilesFunctions
import utils_functions.custom_functions
from utils_functions.custom_functions import preprocess_text, custom_tokenizer
from utils_functions.string_manager import base_host

# corpus='5555555555555555555555555'

corpusClinic = DatasetLoader().load_clinicaltrials() #TODO:

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