# from word_embedding.word_embedding import WordEmbeddingAntique
import chromadb


class VectorDBHelper:
    __instance = None

    @staticmethod
    def get_instance(db_path: str = 'D:/chromadb') -> 'VectorDBHelper':
        if VectorDBHelper.__instance is None:
            VectorDBHelper(db_path)
        return VectorDBHelper.__instance

    def __init__(self, db_path: str):
        if VectorDBHelper.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.db_client = chromadb.PersistentClient(db_path)
            VectorDBHelper.__instance = self

    def insert_vectors(
            self,
            collection_name: str,
            vectors: list[dict],
            chunk_size: int = 4000
    ):
        collection = self.db_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        for chunk in chunks(vectors, chunk_size):
            ids = [vector['doc_id'] for vector in chunk]
            documents = [vector['doc_content'] for vector in chunk]
            embeddings = [vector['doc_vector'] for vector in chunk]

            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
            )

    def query_db(self, collection_name: str, query: list, n_results: int = 10) -> list[dict]:
        collection = self.db_client.get_collection(collection_name)
        query_result = collection.query(
            query_embeddings=query,
            n_results=n_results,
        )

        formatted_result: list = []
        ids = query_result.get('ids', [[]])[0]
        documents = query_result.get('documents', [[]])[0]
        distances = query_result.get('distances', [[]])[0]

        for doc_id, content, dist in zip(ids, documents, distances):
            formatted_result.append({
                # 'id': doc_id,
                'documents': content,
                # 'distance': dist
            })

        return formatted_result
