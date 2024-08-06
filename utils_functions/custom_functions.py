
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
def custom_tokenizer(text: str) -> list[str]:
    tokens = word_tokenize(text.lower())
    return tokens


def get_wordnet_pos(tag):
    # """Converts POS tag to a format that WordNetLemmatizer can understand."""
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
def preprocess_text(text):
    text = text.lower()

    # Tokenization
    words = word_tokenize(text)

    # # Spell Checking
    # words = correct_sentence_spelling(words)

    # Remove Punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]

    # Remove Stop Words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    # Part of Speech Tagging
    pos_tags = pos_tag(stemmed_words)

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]

    return ' '.join(lemmatized_words)
