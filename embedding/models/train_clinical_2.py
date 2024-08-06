from datetime import datetime

from embedding.models.clinical_trained_emb_model import ClinicalTrainedEmbModel
from service.text_preprocessing.text_preprocessor import TextPreprocessing
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

now = datetime.now()
print("Start Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
ClinicalTrainedEmbModel(
    text_processor=TextPreprocessing(),
    batch_size=2000,
    vector_size=500,
).train_model()
print("End Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
