from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging


class Abstract_task(ABC):
    logger = logging.getLogger(__name__)


    def name(self):
        return 'no name!!'

    @abstractmethod
    def load_data(self, split='train', num_samples=1000, topics='all', label_filter=False):
        pass

    @abstractmethod
    def get_model_and_scorer(self, model, tokenizer, batch_size=128):
        pass

    @abstractmethod
    def train_model_from_scratch(self, raw_datasets, model_name):
        pass

    @abstractmethod
    def upload_pretrained_model(self, model_name):
        pass

    @abstractmethod
    def fine_tune_model(self, ntrain, prev_model_name, new_model_name, just_upload=False, num_epochs=30, batch_size=8):
        pass

    def upload_pretrained_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print('model is done')
        return tokenizer, model
    
    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]