# Information Retrieval System

## Datasets

- Clinical
- Antique

## Services

- Entry point service
- Text preprocessing service
- Indexing service
- Query matching service

## Project Structure

### Folders

- Notebook: contains .ipynb files
- Services: contains project services
- user_vector: contains user vector code
- word_embedding: contains word embedding code
- utils_functions: contains resuable function
- Crawling: contains scrapy framework project

### Files

- main
- entry_point: project gateway

## Libraries

- Uvicorn
- Sklearn
- Pydantic
- FastAPI
- NLTK
- Gensim
- chromadb
- numpy

## Flow

- User enter the query, select the dataset and then send the request.
- Request recived by entryPoint and then redirect to the preprocess service.
- Generate vector for processed text (query) and match it using matching service.
- Matching service match between the query vectors and the docs vectors with word embedding.
- Docs are sorted by cosine similarity and return the first 10 results using tf-idf matrix tf-idf model.
- Return the results to the frontend.

## Contributers

- Fadi Asfour
- Omar Zaghlouleh
- Philip Droubi
- Sham Tuameh
- Wael Orabi
