from typing import List

import numpy as np
from gensim.models import Word2Vec


def vectorize_documents(documents: List[dict], word2vec_model: Word2Vec) -> List[dict]:
    vectorized_documents = []
    for document in documents:
        zero_vector = np.zeros(100)
        doc_vector = []
        for token in document['preprocessed_text']:
            if token in word2vec_model.wv:
                doc_vector.append(word2vec_model.wv[token])
        if doc_vector:
            doc_vector = np.asarray(doc_vector)
            avg_vec = doc_vector.mean(axis=0)
            vec = avg_vec
        else:
            vec = zero_vector

        # create the updated dict
        document['doc_vector'] = vec.tolist()
        vectorized_documents.append(document)
    return vectorized_documents
