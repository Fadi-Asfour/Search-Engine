import numpy as np

from service.text_preprocessing.text_preprocessor import TextPreprocessing


def query_vector(query: str, model) -> np.ndarray:
    preprocessor = TextPreprocessing()
    tokens = preprocessor.tokenizer(preprocessor.preprocess(query))

    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        vectors = np.asarray(vectors)
        avg_vec = vectors.mean(axis=0)
        return avg_vec
    else:
        return np.zeros(model.vector_size)



