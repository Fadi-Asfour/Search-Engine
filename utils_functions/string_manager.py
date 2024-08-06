
base_host= "192.168.43.60"

    # "127.0.0.1"
base_url="http://"+base_host

antique_name = 'antique'

dataset_antique = 'C:/ir_files/antique/collection.tsv'
tfidf_matrix_antique = 'C:/ir_files/antique/tfidf_matrix.pickle'
tfidf_model_antique = 'C:/ir_files/antique/tfidf_model.pickle'
crawling_tfidf_matrix_antique = 'C:/ir_files/antique/crawling/tfidf_matrix.pickle'
crawling_tfidf_model_antique = 'C:/ir_files/antique/crawling/tfidf_model.pickle'
document_vectors_antique = "C:/ir_files/antique/embedding/doc_vectors.pkl"
word2vec_model_antique = "C:/ir_files/antique/embedding/word2vec_model.kv"


clinic_name = 'clinic'
dataset_clinic = 'C:/ir_files/clinic/collection.tsv'
tfidf_matrix_clinic = 'C:/ir_files/clinic/tfidf_matrix.pickle'
tfidf_model_clinic = 'C:/ir_files/clinic/tfidf_model.pickle'
document_vectors_clinic = "C:/ir_files/clinic/embedding/doc_vectors.pkl"
word2vec_model_clinic = "C:/ir_files/clinic/embedding/word2vec_model.kv"


# Embedding Paths
antique_embedding_model_path = 'G:/ir_files/embedding/antique/model/antique_embedding_model.kv'
clinical_embedding_model_path = 'G:/ir_files/embedding/clinic/model/word2vec_model.kv'
antique_embedding_vector_db_path = 'G:/ir_files/embedding/antique/db'
clinical_embedding_vector_db_path = 'G:/ir_files/embedding/clinic/db'
