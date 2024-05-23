from sklearn.feature_extraction.text import TfidfVectorizer

from utils.functions import store_model


# Custom tokenizer and preprocessor functions
def custom_tokenizer(text):
    # Implement your custom tokenizer here
    return text.split()


def preprocess_text(text):
    return text


class TFIDFVectorizerService:
    def __init__(self, text_processor, text_tokenizer):
        self.text_processor = text_processor
        self.text_tokenizer = text_tokenizer
        self.tfidf_matrix = None
        self.tfidf_model = None

    def build_model(self):
        corpus = DatasetLoader().load_dataset()
        documents = list(corpus.values())
        self.tfidf_model = TfidfVectorizer(preprocessor=self.text_processor, tokenizer=self.text_tokenizer)
        self.tfidf_matrix = self.tfidf_model.fit_transform(documents)
        store_model(self.tfidf_matrix, self.tfidf_model)










