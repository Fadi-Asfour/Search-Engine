import csv
import ir_datasets
import pandas as pd

from utils_functions.string_manager import dataset_antique


class DatasetLoader:

    @staticmethod
    def load_clinicaltrials():
      dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
      corpus = {}
      counter = 0
      for doc in dataset.docs_iter():
            corpus[doc.doc_id[
                   3:11]] = doc.title + " " + doc.summary + " " + doc.detailed_description + " " + doc.eligibility
            # counter+=1
            # if counter>=10:
            #     break
      # print(corpus["00000107"])
      return corpus
    @staticmethod
    def load_antique():
        df = pd.read_csv(dataset_antique, sep='\t', header=None, names=['doc_id', 'text'])

        # Build the corpus dictionary
        corpus = {}
        counter = 0
        for index, row in df.iterrows():

            if isinstance(row['text'], str):
                corpus[row['doc_id']] = row['text']
            else:
                corpus[row['doc_id']] = ""
        return corpus


