from sklearn.metrics.pairwise import cosine_similarity

from services.text_preprocessing.text_preprocessing import TextPreprocessor
from utils.functions import load_model


class QueryMatching:
    def __init__(self, matrix_filename: str, model_filename: str, corpus: dict):
        self.tfidf_matrix , self.tfidf_model = load_model(matrix_filename,model_filename)
        self.preprocessor = TextPreprocessor()
        self.corpus = corpus

    def process_query(self, query: str):
        query_tfidf = self.tfidf_model.transform([query])
        cosine_similarities = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        ranked_doc_indices = cosine_similarities.argsort()[::-1]
        return ranked_doc_indices, cosine_similarities

    def get_retrieved_queries(self, query: str, k=10):
        preprocessed_query = self.preprocessor.preprocess(query)
        ranked_indices, similarities = self.process_query(preprocessed_query)
        ids_list = []
        for idx in ranked_indices[:k]:
            doc_id = list(self.corpus.keys())[idx]
            ids_list.append(doc_id)
        return ids_list
