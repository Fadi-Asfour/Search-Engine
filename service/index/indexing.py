from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from starlette.middleware.cors import CORSMiddleware

from service.text_preprocessing.text_preprocessor import TextPreprocessing
from utils_functions.dataset_load import DatasetLoader
from utils_functions.dataset_type_manager import DatasetTypeManager
from utils_functions.functions import FilesFunctions
from utils_functions.string_manager import base_host


class TFIDFVectorizerService:
    def __init__(self, text_processor, text_tokenizer):
        self.text_processor = text_processor
        self.text_tokenizer = text_tokenizer
        self.tfidf_matrix = None
        self.tfidf_model = None

    def build_model(self, dataset_name: str):
        datasetype = DatasetTypeManager(dataset_name)

        if dataset_name=="antique":
            corpus = DatasetLoader().load_antique()
        else:
            corpus = DatasetLoader().load_clinicaltrials()

        documents = list(corpus.values())
        self.tfidf_model = TfidfVectorizer(preprocessor=self.text_processor, tokenizer=self.text_tokenizer)
        self.tfidf_matrix = self.tfidf_model.fit_transform(documents)
        FilesFunctions().store_model(self.tfidf_matrix, self.tfidf_model, datasetype.tfidf_matrix, datasetype.tfidf_model)

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class DatasetRequest(BaseModel):
    dataset_name: str

@app.post("/index")
async def index_dataset(request: DatasetRequest):
    try:
        preprocessor = TextPreprocessing()
        vectorizer_service = TFIDFVectorizerService(preprocessor.preprocess, preprocessor.custom_tokenizer)
        vectorizer_service.build_model(request.dataset_name)
        return {"status": "Indexing completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("indexing:app", host=base_host, port=8004, reload=True)