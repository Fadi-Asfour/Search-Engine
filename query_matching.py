from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from text_preprocessor import TextPreprocessing
from utils_functions.dataset_load import DatasetLoader
from utils_functions.functions import FilesFunctions
import custom_functions
from custom_functions import preprocess_text, custom_tokenizer
from utils_functions.string_manager import base_host

# corpus='5555555555555555555555555'

corpus = DatasetLoader().load_clinicaltrials()


class QueryMatching:
    def __init__(self, matrix_filename: str, model_filename: str, corpus: dict):
        # print("gggggggggg")
        self.tfidf_matrix = FilesFunctions.load_file(matrix_filename)
        self.tfidf_model = FilesFunctions.load_file(model_filename)
        print("ssssssssssssssss")
        # print(self.tfidf_model)
        self.preprocessor = TextPreprocessing()
        self.corpus = corpus

    def process_query(self, query: str):
        preprocessed_query = self.preprocessor.preprocess(query) #TODO: delete it
        print("fadiiiiiiiiii")
        # print(preprocessed_query)
        # print(self.tfidf_model)
        print("eeeeeeeeeeeeeeee")
        query_vector = self.tfidf_model.transform([preprocessed_query])
        print("waelllllllllll")
        # print(query_vector)
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked_doc_indices = cosine_similarities.argsort()[::-1]
        return ranked_doc_indices, cosine_similarities

    # def process_query(query: str, tfidf_model, tfidf_matrix):
    #     query_tfidf = tfidf_model.transform([query])
    #     cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    #     ranked_doc_indices = cosine_similarities.argsort()[::-1]
    #     return ranked_doc_indices, cosine_similarities


class QueryRequest(BaseModel):
    query: str
    dataset_path: str

app = FastAPI()

@app.post("/query")
async def query_dataset(request: QueryRequest):
    try:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(corpus["00000107"])
        tfidf_model = FilesFunctions.load_file("C:/ir_files/tfidf_model.pickle")
        print(tfidf_model)
        query_matching = QueryMatching("C:/ir_files/tfidf_matrix.pickle", "C:/ir_files/tfidf_model.pickle", corpus)
        ranked_indices, similarities = query_matching.process_query(request.query)
        print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;")
        # print(similarities)
        # results = sorted(zip(corpus.keys(), similarities[0]), key=lambda x: x[1], reverse=True)
        results = []
        for idx in ranked_indices[:10]:
            doc_id = list(corpus.values())[idx]
            results.append(doc_id)
        print({"results": results})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("query_matching:app", host=base_host, port=8007, reload=True)

