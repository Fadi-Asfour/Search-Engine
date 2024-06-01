from sklearn.feature_extraction.text import TfidfVectorizer
from utils_functions.dataset_load import DatasetLoader
from utils_functions.functions import FilesFunctions
from text_preprocessor import TextPreprocessing
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class TFIDFVectorizerService:
    def __init__(self, text_processor, text_tokenizer):
        self.text_processor = text_processor
        self.text_tokenizer = text_tokenizer
        self.tfidf_matrix = None
        self.tfidf_model = None

    def build_model(self, dataset_path: str):
        corpus = DatasetLoader().load_clinicaltrials(dataset_path)
        documents = list(corpus.values())
        self.tfidf_model = TfidfVectorizer(preprocessor=self.text_processor, tokenizer=self.text_tokenizer)
        self.tfidf_matrix = self.tfidf_model.fit_transform(documents)
        FilesFunctions().store_model(self.tfidf_matrix, self.tfidf_model, "path_to_matrix.pkl", "path_to_model.pkl")

app = FastAPI()

class DatasetRequest(BaseModel):
    dataset_path: str

@app.post("/index")
async def index_dataset(request: DatasetRequest):
    try:
        preprocessor = TextPreprocessing()
        vectorizer_service = TFIDFVectorizerService(preprocessor.preprocess, preprocessor.custom_tokenizer)
        vectorizer_service.build_model(request.dataset_path)
        return {"status": "Indexing completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("indexing:app", host=base_host, port=8008, reload=True)