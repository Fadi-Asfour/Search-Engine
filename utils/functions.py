import pickle
import os


def store_file(file_location, content):
    if os.path.exists(file_location):
        os.remove(file_location)
    with open(file_location, 'wb') as handle:
        pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_location):
    with open(file_location, 'rb') as handle:
        content = pickle.load(handle)
    return content


def store_model(tfidf_matrix, tfidf_model, matrix_filename, model_filename):
    store_file(matrix_filename, tfidf_matrix)
    store_file(model_filename, tfidf_model)


def load_model(matrix_filename, model_filename):
    tfidf_matrix = load_file(matrix_filename)
    tfidf_model = load_file(model_filename)
    return tfidf_matrix, tfidf_model
