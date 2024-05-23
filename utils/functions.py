import os
import pickle


def save_file(file_location: str, content):
    if os.path.exists(file_location):
        os.remove(file_location)
    with open(file_location, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_location: str):
    with open(file_location, 'rb') as handle:
        content = pickle.load(handle)
    return content


def save_tfidf_data(tfidf_matrix, tfidf_model):
    save_file("tfidf_matrix.pickle", tfidf_matrix)
    save_file("tfidf_model.pickle", tfidf_model)
