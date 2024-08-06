import string
from typing import List
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from utils_functions.string_manager import base_host


class TextPreprocessing:
    def __init__(self) -> None:
        self.tokenizer = word_tokenize
        self.stopwords_tokens = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def to_lower(tokens: List[str]) -> List[str]:
        return [token.lower() for token in tokens]

    @staticmethod
    def remove_punctuation(tokens: List[str]) -> List[str]:
        return [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [token for token in tokens if token not in self.stopwords_tokens]

    def stemming(self, tokens: List[str]) -> List[str]:
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatizing(self, tokens: List[str]) -> List[str]:
        pos_tags = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(tag)) for token, tag in pos_tags]

    @staticmethod
    def get_wordnet_pos(tag: str) -> str:
        tag = tag[0].upper()
        tag_dict = {
            'J': wordnet.ADJ,
            'N': wordnet.NOUN,
            'V': wordnet.VERB,
            'R': wordnet.ADV
        }
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess(self, text: str) -> str:
        tokens = self.custom_tokenizer(text)
        tokens = self.to_lower(tokens)
        tokens = self.remove_punctuation(tokens)
        tokens = self.remove_stop_words(tokens)
        tokens = self.stemming(tokens)
        tokens = self.lemmatizing(tokens)
        return ' '.join(tokens)

    def custom_tokenizer(self, text: str) -> List[str]:
        return self.tokenizer(text)

    def process_text(self, text: str) -> List[str]:
        text = self.preprocess(text)
        return self.custom_tokenizer(text)


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str


@app.post("/preprocess")
async def preprocess_text(request: TextRequest):
    try:
        preprocessor = TextPreprocessing()
        processed_text = preprocessor.preprocess(request.text)
        print(processed_text)
        return {"processed_text": processed_text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("text_preprocessor:app", host=base_host, port=8003, reload=True)
