import uvicorn
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from service.text_preprocessing.text_preprocessor import TextPreprocessing
# from services.text_preprocessing.text_preprocessor import TextPreprocessing
from utils_functions.dataset_load import DatasetLoader
from utils_functions.dataset_type_manager import DatasetTypeManager
from utils_functions.functions import FilesFunctions
import utils_functions.custom_functions
from utils_functions.custom_functions import preprocess_text, custom_tokenizer
from utils_functions.string_manager import base_host
from embedding.matching.antique_ebm_matcher import AntiqueEmbMatcher

from embedding.matching.antique_ebm_matcher import AntiqueEmbMatcher
from service.text_preprocessing.text_preprocessor import TextPreprocessing

from service.text_preprocessing.text_preprocessor import TextPreprocessing




class QueryRequest(BaseModel):
    query: str
    dataset_name: str

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/query")
async def query_dataset(request: QueryRequest):
    try:
        antique_matcher = AntiqueEmbMatcher(text_processor=TextPreprocessing())
        s = antique_matcher.match(request.query)
        m = {}
        l = []
        for match in s:
            l.append(match["documents"])
            print(match)
        # m["documents"] = l
        return {"results": l}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("vectore_store_match:app", host=base_host, port=8013, reload=True)


