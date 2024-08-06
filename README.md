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

- Sklearn
- Pydantic
- NLTK
- Gensim
- FastAPI
- Uvicorn
- chromadb
- numpy

## Flow

- User Query Submission:

  The user inputs a query, selects the relevant dataset, and sends the request.
- Request Handling:

  The request is received by the entry point, which then redirects it to the preprocessing service.
Query Preprocessing:

  The preprocessing service generates a vector representation of the processed text (query).
- Vector Matching:

  The matching service compares the query vectors to document vectors.
- Results Sorting:

  Documents are sorted based on cosine similarity scores. The top 10 results are returned using the TF-IDF model.
- Results Delivery:

  The sorted results are sent back to the frontend for display to the user.
