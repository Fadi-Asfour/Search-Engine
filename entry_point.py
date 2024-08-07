from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from starlette.middleware.cors import CORSMiddleware

from utils_functions.string_manager import base_host, base_url

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class SearchQuery(BaseModel):
    query: str
    dataset_name: str
    type: str

@app.post("/preprocess")
async def preprocess_text(request: SearchQuery):
    try:
        response = requests.post(base_url+":8003/preprocess", json={"text": request.query})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_dataset(request: SearchQuery):
    try:
        response = requests.post(base_url+":8004/index", json={"dataset_name": request.dataset_name})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_dataset(request: SearchQuery):
    try:
        response = requests.post(base_url+":8005/query", json=request.dict())
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(data: SearchQuery):
    try:
        # Step 1: Preprocess the query
        preprocess_response = requests.post(base_url+":8003/preprocess", json={"text": data.query})
        preprocess_response.raise_for_status()
        processed_query = preprocess_response.json().get("processed_text")

        # Step 2: Index the dataset (if not already indexed)
       # index_response = requests.post(base_url+":8002/index", json={"dataset_name": data.dataset_name})
        # index_response.raise_for_status()

        # Step 3: Query the dataset with the processed query
        documents=""
        if data.type=="normal":
         query_response = requests.post(base_url+":8005/query", json={"query": processed_query, "dataset_name": data.dataset_name})
         query_response.raise_for_status()
         documents = query_response.json().get("results")
         return {"documents": documents}

        elif data.type =="crawl":
         query_response = requests.post(base_url+":8006/query", json={"query": processed_query})
         query_response.raise_for_status()
         documents = query_response.json().get("results")
         return {"documents": documents}

        elif data.type =="embed":
         query_response = requests.post(base_url+":8007/query", json={"query": processed_query, "dataset_name": data.dataset_name})
         query_response.raise_for_status()
         documents = query_response.json().get("results")
         return {"documents": documents}
        elif data.type =="vector":
         query_response = requests.post(base_url+":8013/query", json={"query": processed_query, "dataset_name": data.dataset_name})
         query_response.raise_for_status()
         documents = query_response.json().get("results")
         return { "documents" : documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    print("welcome")
    import uvicorn
    uvicorn.run("entry_point:app", host=base_host, port=8008, reload=True)