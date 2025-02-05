from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
import tqdm.auto as tqdm
import adatest
from transformers import AutoTokenizer
import transformers

class QQP:

    def name(self):
        return 'qqp'
        
    def chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def load_data(self, topics=None, split='train', num_samples=500, label_filter=None):
        raw_datasets = load_dataset('glue', 'qqp', split=split)
        if(num_samples!='all'):# and split=='train'):
            raw_datasets = raw_datasets.select(range(0,min(num_samples, len(raw_datasets))))
        return raw_datasets

    def get_model_and_scorer(self, model, tokenizer, batch_size=128):
        nli_tk = tokenizer
        nli_model = model
        classifier = transformers.pipeline("sentiment-analysis", 
                                            model=nli_model, 
                                            tokenizer=nli_tk, 
                                            framework="pt", 
                                            device=0, 
                                            top_k=None)
        def classifier2(inputs, batch_size=batch_size):
            inputs = [dict(zip(['text', 'text_pair'], x.split(' | '))) for x in inputs]
            inputs = [x if len(x) == 2 else {'text': 'aaa', 'text_pair': 'aaa'} for x in inputs]
            preds = []
            # for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):
            #     print('inside classifier2', d, classifier(d))
            #     preds.extend(classifier(d))
            #     print(preds)
            #     exit(0)
            # preds = [[float(x['score']) for x in y] if a != ('a', 'a') else [-10000, -10000] for a, y in zip(inputs, preds)]
            # return np.array(preds)


            for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):
                result = classifier(d)
                print('result is', result)
                labels = ['LABEL_0', 'LABEL_1']
                for result_element in result: #each list
                    preds_element = []
                    for lb in labels:
                        for dic_item in result_element: #each dic part
                            if(dic_item['label'] == lb):
                                preds_element.append(dic_item['score'])
                    preds.append(preds_element)
                # preds.extend(res)
                print(d, preds)
            return np.array(preds)

        tensor_output_model = classifier2
        labels = ['not duplicate', 'duplicate']
        tensor_output_model.output_names = labels
        scorer = adatest.TextScorer(tensor_output_model)
        return tensor_output_model, scorer


    def fine_tune_model(self, ntrain, prev_model_name, new_model_name, 
                        just_upload=False, batch_size=8,num_epochs=int(30)):
        # Reload the original finetuned checkpoint in case you updated it somewhere in the notebook

        raw_datasets = self.load_data()
        model_name = prev_model_name
        model_new = RobertaForSequenceClassification.from_pretrained(model_name)#, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tensor_output_model, scorer = get_model_scorer(model, tokenizer)


        def tokenize_function(examples):
            return tokenizer(examples["question1"], examples['question2'], truncation=True)


        tokenized_train = ntrain.map(tokenize_function, batched=True)
        # tokenized_val = raw_datasets['validation'].map(tokenize_function, batched=True).shuffle().select(range(200))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        # model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
        train_d = tokenized_train.shuffle()#.select(range(900))
        training_args = TrainingArguments("test_trainer", learning_rate=5e-6, logging_steps=50,
                                        # save_strategy='no', eval_steps=500,
                                        label_names=['label'], 
                                        per_device_train_batch_size=batch_size,
                                        num_train_epochs=num_epochs,
                                        weight_decay=0.01,
                                        )
        trainer = Trainer(
                model=model_new, 
                args=training_args, 
                train_dataset=train_d, 
                # eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator, 
        )
        trainer.train()

        trainer.save_model(new_model_name)

        print('model is saved')

        return model_new