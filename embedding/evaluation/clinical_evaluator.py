from typing import List

import ir_datasets
from embedding.matching.clinical_emb_matcher import ClinicalEmbMatcher
from service.text_preprocessing.text_preprocessor import TextPreprocessing


class ClinicalEvaluator:

    def __init__(self):
        self.clinical_matcher = ClinicalEmbMatcher(text_processor=TextPreprocessing())
        self.clinical_dataset = ir_datasets.load('clinicaltrials/2021/trec-ct-2021')

    def compute_relevance_scores(self, query_text: str) -> List[str]:
        result = self.clinical_matcher.match(query_text)
        relevance_scores = [item['id'] for item in result]
        return relevance_scores

    @staticmethod
    def compute_precision_recall_at_k(self, relevant_docs, retrieved_docs, k):
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs[:k]]
        true_positives = sum([1 for i in range(len(y_true)) if y_true[i] == 1])
        recall_at_k = true_positives / len(relevant_docs)
        precision_at_k = true_positives / k
        return precision_at_k, recall_at_k

    def evaluate_map(self):
        queries_ids = {qrel[0]: '' for qrel in self.clinical_dataset.qrels_iter()}

        map_sum = 0
        for query_id in list(queries_ids.keys()):
            map_sum += self.calculate_MAP(query_id)

        print(f"Mean Average Precision : {map_sum / len(queries_ids)}")

    def evaluate_mrr(self):
        queries_ids = {}
        for qrel in self.clinical_dataset.qrels_iter():
            queries_ids.update({qrel.query_id: ''})

        mrr_sum = 0
        for query_id in list(queries_ids.keys()):
            mrr_sum += self.calculate_MRR(query_id)

        print(f"Mean Reciprocal Rank : {mrr_sum / len(queries_ids)}")

    def calculate_MAP(self, query_id):
        relevant_docs = []
        retrieved_docs = []

        # Get relevant documents for the query
        for qrel in self.clinical_dataset.qrels_iter():
            if qrel.query_id == query_id and qrel.relevance > 0:
                relevant_docs.append(qrel.doc_id)

        # Get retrieved documents for the query
        for query in self.clinical_dataset.queries_iter():
            if query.query_id == query_id:
                retrieved_docs = self.compute_relevance_scores(query.text)
                break

        # Compute mean average precision
        pk_sum = 0
        total_relevant = 0
        for i in range(1, 11):
            relevant_ret = 0
            for j in range(i):
                if j < len(retrieved_docs) and retrieved_docs[j] in relevant_docs:
                    relevant_ret += 1
            p_at_k = (relevant_ret / i) * (
                1 if i - 1 < len(retrieved_docs) and retrieved_docs[i - 1] in relevant_docs else 0)
            pk_sum += p_at_k
            if i - 1 < len(retrieved_docs) and retrieved_docs[i - 1] in relevant_docs:
                total_relevant += 1

        return 0 if total_relevant == 0 else pk_sum / total_relevant

    def calculate_MRR(self, query_id):
        relevant_docs = []
        for qrel in self.clinical_dataset.qrels_iter():
            if qrel.query_id == query_id and qrel.relevance > 0:
                relevant_docs.append(qrel.doc_id)

        retrieved_docs = []
        for query in self.clinical_dataset.queries_iter():
            if query.query_id == query_id:
                retrieved_docs = self.compute_relevance_scores(query.text)
                break

        for i, result in enumerate(retrieved_docs):
            if result in relevant_docs:
                return 1 / (i + 1)

        return 0
