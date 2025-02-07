from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import transformers
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding
import tqdm.auto as tqdm
import adatest
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from coalign.models.abstract_task import Abstract_task

class Toxicity(Abstract_task):

    def name(self):
        return 'toxicity'

    #load data and refactor it to have sentence as input and label as output and return a dataset
    def load_data(self, split='train', num_samples=1000, topics='all', label_filter=False):
        raw_datasets = load_dataset("tasksource/jigsaw", split=split)
        raw_datasets = raw_datasets.map(lambda x: {'sentence': x['comment_text']})
        raw_datasets = raw_datasets.rename_column("toxic", "label")
        if(num_samples!='all'):# and split=='train'):
            raw_datasets = raw_datasets.select(range(0,min(num_samples, len(raw_datasets))))

        return raw_datasets

    def get_model_and_scorer(self, model, tokenizer, batch_size=128):
        classifier = transformers.pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True, device=0)

        labels = ['non-toxic', 'toxic']
        def classifier2(inputs, batch_size=batch_size):
            preds = []
            for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):
                result = classifier(d)
                print('result is', result)
                labels = ['Non-Toxic', 'Toxic', 'LABEL_0', 'LABEL_1']
                for result_element in result: #each list
                    preds_element = []
                    for lb in labels:
                        for dic_item in result_element: #each dic part
                            if(dic_item['label'] == lb):
                                preds_element.append(dic_item['score'])
                    preds.append(preds_element)
                # preds.extend(res)
                print(d, preds)
            # preds = [[float(x['score']) for x in y] for a, y in zip(inputs, preds)]
            return np.array(preds)
            # preds = []
            # for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):
            #     print('the classifier', classifier(d))
            #     print('the classifier2', classifier(d)[0])
            #     print('the classifier3', classifier(d)[0][0]['score'])
            #     preds.extend(classifier(d))
            #     print('individual pred', preds)
            # # preds = [[float(x['score']) for x in y] for a, y in zip(inputs, preds)]
            # preds = [[y[0]['score'], 1-y[0]['score']] for a, y in zip(inputs, preds)]
            # print('new preds', preds)
            # return np.array(preds)
        
        tensor_output_model = classifier2
        tensor_output_model.output_names = labels
        scorer = adatest.TextScorer(tensor_output_model)
        return tensor_output_model, scorer
    

    def train_model_from_scratch(self, raw_datasets, model_name):

        # This trains a model and saves it.

        #some preprocessing for sentiment task
        train_labels = np.array([self.map_label_sentiment(x) for x in raw_datasets['train']['label']])
        train_sentences = list(raw_datasets['train']['sentence'])
        val_labels = np.array([self.map_label_sentiment(x) for x in raw_datasets['validation']['label']])
        val_sentences = list(raw_datasets['validation']['sentence'])
        train = Dataset.from_dict({'sentence' : train_sentences, 'label': train_labels})
        val = Dataset.from_dict({'sentence' : val_sentences, 'label': val_labels})


        model_name_to_load = './toxic-bert'
        tokenizer = AutoTokenizer.from_pretrained(model_name_to_load)
        def tokenize_function(examples):
            return tokenizer(examples["sentence"], truncation=True)
        tokenized_train = train.map(tokenize_function, batched=True)
        tokenized_val = val.map(tokenize_function, batched=True)
        tokenized_datasets = {'train': tokenized_train, 'validation': tokenized_val}
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_to_load, num_labels=3)
        
        training_args = TrainingArguments("test_trainer", learning_rate=5e-6, logging_steps=300,
                                        save_strategy='no', eval_steps=500,
                                        label_names=['label'], 
                                        per_device_train_batch_size=8,
                                        num_train_epochs=15,
                                        weight_decay=0.01,
                                        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator, 
        )
        # trainer.train()
        trainer.save_model(model_name)
        print('model is done')
        return tokenizer, model


    def fine_tune_model(self, ntrain, prev_model_name, new_model_name, just_upload=False, num_epochs=30, batch_size=8):

        if(just_upload):
            return AutoTokenizer.from_pretrained(new_model_name), AutoModelForSequenceClassification.from_pretrained(new_model_name)
        # Reload the original finetuned checkpoint in case you updated it somewhere in the notebook

        model_name = prev_model_name
        model_new = AutoModelForSequenceClassification.from_pretrained(model_name)#, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tensor_output_model, scorer = get_model_scorer(model, tokenizer)


        def tokenize_function(examples):
            return tokenizer(examples["sentence"], truncation=True)



        tokenized_train = ntrain.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_d = tokenized_train.shuffle()#.select(range(900))
        training_args = TrainingArguments("test_trainer", learning_rate=5e-6, logging_steps=50,
                                        save_strategy='no', 
                                        # eval_steps=500,
                                        label_names=['label'], 
                                        per_device_train_batch_size=batch_size,
                                        num_train_epochs=num_epochs,
                                        weight_decay=0.01,
                                        )
        trainer = Trainer(
                model=model_new, 
                args=training_args, 
                train_dataset=train_d, 
                tokenizer=tokenizer,
                data_collator=data_collator, 
        )
        trainer.train()

        trainer.save_model(new_model_name)

        print('model is saved')

        return tokenizer, model_new

#write main
if __name__ == '__main__':
    task = Toxicity()
    # ds = task.load_data(num_samples=10)
    # print(ds[0])
    model_name_to_load = 'toxic-bert'
    # model_name_to_load = 'mohsenfayyaz/toxicity-classifier'
    tokenizer = AutoTokenizer.from_pretrained(model_name_to_load)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_to_load)
    # tokenizer, model = task.fine_tune_model(None, model_name_to_load, '../fine_tuned_models/toxicity_classifier', just_upload=True)
    # model = torch.hub.load('unitaryai/detoxify','toxic_bert')
    from transformers import pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=None, device=0)
    print(pipe("I love everyone."))

