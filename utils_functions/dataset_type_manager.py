from utils_functions.string_manager import dataset_antique, tfidf_matrix_antique, tfidf_model_antique, \
    embedding_model_antique, document_vectors_antique, dataset_clinic, tfidf_matrix_clinic, tfidf_model_clinic, \
    document_vectors_clinic, embedding_model_clinic


class DatasetTypeManager:
    def __init__(self,dataset_name:str):
        # Initialize your variables with default values here
        print("DatasetTypeManager")
        print("dataset_name")
        if dataset_name== "antique":
         self.dataset = dataset_antique
         self.tfidf_matrix = tfidf_matrix_antique
         self.tfidf_model = tfidf_model_antique
         self.vectors = document_vectors_antique
         self.embedding_model = embedding_model_antique
        else:
            self.dataset = dataset_clinic
            self.tfidf_matrix = tfidf_matrix_clinic
            self.tfidf_model = tfidf_model_clinic
            self.vectors = document_vectors_clinic
            self.embedding_model = embedding_model_clinic

    # def set_values(self, input_string):
    #     variable_value_pairs = input_string.split(';')
    #     for pair in variable_value_pairs:
    #         var_name, var_value = pair.strip().split('=')
    #         setattr(self, var_name, var_value)
    #
    # def get_values(self):
    #     # Return a dictionary containing the variable names and their current values
    #     return {
    #         'dataset': self.dataset,
    #         'tfidf_matrix': self.tfidf_matrix,
    #         'tfidf_model': self.tfidf_model,
    #         'vectors': self.vectors,
    #         'word_embedding_model': self.word_embedding_model,
    #         'tfidf_queries': self.tfidf_queries,
    #         'queries_model': self.queries_model
    #     }

# Example usage:

