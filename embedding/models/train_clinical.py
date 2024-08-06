from datetime import datetime

from embedding.models.clinical_emb_model import ClinicalEmbModel
from service.text_preprocessing.text_preprocessor import TextPreprocessing
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

now = datetime.now()
print("Start Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
ClinicalEmbModel(
    text_processor=TextPreprocessing(),
    workers=5,
    batch_size=2000,
    epochs=40,
    vector_size=500,
).train_model()
print("End Time:" + now.strftime("%Y-%m-%d %H:%M:%S"))
