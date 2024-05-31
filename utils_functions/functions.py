import os
import pickle

# Import the needed functions and classes
# from services.text_preprocessing.text_preprocessing import preprocess_text, TextPreprocessing


class FilesFunctions:
    @staticmethod
    def store_file(file_location: str, content):
        if os.path.exists(file_location):
            os.remove(file_location)
        with open(file_location, 'wb') as handle:
            pickle.dump(content, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_file(file_location: str):
        with open(file_location, 'rb') as handle:
            content = pickle.load(handle)
        return content

    def store_model(self, tfidf_model, tfidf_matrix, matrix_filename, model_filename):
        self.store_file(matrix_filename, tfidf_matrix)
        self.store_file(model_filename, tfidf_model)
