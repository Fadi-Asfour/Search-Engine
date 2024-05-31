import csv
import ir_datasets


class DatasetLoader:

    @staticmethod
    def load_clinicaltrials():
        # dataset_path: str
      #  documents = {}

      #  with open(dataset_path, encoding="utf-8") as documents:
       #     documents_load = csv.reader(documents, delimiter="\t")
       #     for doc in documents_load:
        #        documents.update({doc.id: doc.title + " " + doc.summary + " " + doc.detailed_description + " " + doc.eligibility })
      #  return documents
      dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")

      corpus = {}
      counter = 0
      for doc in dataset.docs_iter():
            corpus[doc.doc_id[
                   3:11]] = doc.title + " " + doc.summary + " " + doc.detailed_description + " " + doc.eligibility
            # counter+=1
            # if counter>=10:
            #     break
      print(corpus["00000107"])
      return corpus
