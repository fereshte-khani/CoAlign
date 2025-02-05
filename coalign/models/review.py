
from datasets import load_dataset, load_metric,Value
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
from datasets import load_metric
import transformers
import tqdm.auto as tqdm
import adatest
from utils.test_tree_functions import adding_new_sentences_to_test_tree
import sentence_transformers


print(int(time.time()*10000)%1000000000)
# random.seed(int(time.time()*100000)%1000000000)
# np.random.seed(int(time.time()*100000)%1000000000)
MAX_LENGTH = 200
metric = load_metric("accuracy")

class Review():

    def name(self):
        return "review"

    def original_topic(self):
        return 'digital_ebook_purchase'

    def base_model(self):
        return 'distilbert-base-uncased'#'roberta-large' #'distilbert-base-uncased' #
    
    #load a dataset from a file_path
    def load_dataset_from_file_name(self, file_path):
        #sometimes file_path has weird characters and it causes error so here we just rename it!
#check if file_path have special characters

        # if(' ' in file_path):
        #     new_file_path = file_path.replace(' ', '_')
        #     os.rename(file_path, new_file_path)
        #     file_path = new_file_path

        print(file_path)
        f1 = open(file_path, 'r')
        f2 = open('tmp_file2.json', 'w')
        f2.write(f1.read())
        f1.close()
        f2.close()

        data_files = {"train": 'tmp_file2.json'}
        review_dataset = load_dataset("json", data_files=data_files, split="train")
        #in review dataset the stars column type changes so we need to cast it
        review_dataset = review_dataset.cast_column("label", Value(dtype='int32', id=None))
        return review_dataset

    #every topic has its own test-tree (initialized by training data) each load_data call adatest to generate a set of new data and only chose the ones that
    #have disagreements.
    def load_data_from_test_tree(self, test_tree, scorer, predictor, topic):

        print('launching browser')
        roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
        browser = test_tree(scorer=scorer, embedding_model=roberta_embed_model, max_suggestions=100, 
        recompute_scores=True, predictor = predictor)
        suggestions = browser._generate_suggestions(topic)
        for index, row in suggestions.iterrows():
            try:
                if(float(row['score']) > 0): #it means we have disagreements
                    print(row)
                    print('disagreement found--------------------------')
                    print(row['value1'], row['value2'], row['score value1 outputs'])
                    adding_new_sentences_to_test_tree(test_tree, [row['value1']], [row['value2']], topic)
            except ValueError:
                print('not float I guess!', row['score'])



        # adding_new_sentences_to_test_tree(test_tree, list1, list2, topic)
        # adatest_to_dictionary(task, test_tree)

    def load_data(self, topics='all', split='train', num_samples=500, label_filter=False):
        raw_dataset = load_dataset('amazon_reviews_multi', 'en', split=split)
        print(raw_dataset)
        #some preprocessing
        # raw_dataset = raw_dataset.filter(lambda x: x['label'] !=3)
        # raw_dataset = raw_dataset.map(lambda x: {'label': 1 if x['label'] >=4 else 0})
        raw_dataset = raw_dataset.map(lambda x: {'review_body': x['review_title']+': '+x['review_body'], 'stars': x['stars'] - 1})
        raw_dataset = raw_dataset.rename_column("stars", "label")


        #choose rows with the specified product topics or specific labels
        if(label_filter):
            print('in load data', topics, label_filter)
            if(topics!='all'):
                raw_dataset = raw_dataset.filter(lambda x: x['label'] in topics)
        else:
            #for product categories we only look at [0,2,4] for simplicity?
            # raw_dataset = raw_dataset.filter(lambda x: x['label'] in [0,2,4])
            raw_dataset = raw_dataset.filter(lambda x: x['label'] in [0,4])
            if(topics!='all'):
                raw_dataset = raw_dataset.filter(lambda x: x['product_category'] in topics)

        #shuffle the datset
        # np.random.seed(int(time.time()*1000000)%1000000000)
        raw_dataset = raw_dataset.shuffle(seed = (int(time.time()*100000)%1000000000))

        # choose num_samples samples from each product category
        if(num_samples!='all'):# and split=='train'):
            raw_dataset = raw_dataset.select(range(0,min(int(num_samples), len(raw_dataset))))
        
        raw_dataset = raw_dataset.map(lambda x: {'label': 1 if x['label'] == 4 else 0, 
                'sentence': x['review_body']})  
        
        return raw_dataset

    def input(self):
        return 'review_body'
    
    def max_length(self):
        return MAX_LENGTH
    

    def tokenize_function(self, examples):
            return self.tokenizer(examples[self.input()], truncation=True, max_length=self.max_length())

    def fine_tune_model(self, review_dataset, original_model_path, output_path, just_upload=False, num_epochs=30):

        if(just_upload):
            return AutoModelForSequenceClassification.from_pretrained(output_path)

            
        #load the model from original_model_path
        model = AutoModelForSequenceClassification.from_pretrained(original_model_path)


        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        self.tokenizer = tokenizer
        print('in fine tune', review_dataset)
        review_dataset = review_dataset.shuffle(seed = (int(time.time()*100000)%1000000000))
        tokenized_train = review_dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = {'train': tokenized_train}

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(original_model_path)
        
        training_args = TrainingArguments("test_trainer-fine-tune", 
                                        learning_rate=5e-6, 
                                        logging_steps=50,
                                        save_strategy='no', 
                                        # eval_steps=500, 
                                        # save_total_limit = 2,
                                        # evaluation_strategy="steps",
                                        # save_strategy='steps',
                                        label_names=['label'], 
                                        per_device_train_batch_size=4,
                                        num_train_epochs=num_epochs,
                                        weight_decay=0.01,
                                        # report_to="wandb"
                                        # metric_for_best_model = 'train_loss',
                                        # load_best_model_at_end = True
                                        )
                        
        trainer = Trainer(
            model=model, 
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            tokenizer=tokenizer,
            data_collator=data_collator, 
            # callbacks = [EarlyStoppingCallback(early_stopping_threshold=0.01)],
        )
        trainer.train()
        trainer.save_model(output_path)
        return model

    def compute_predictions(self,dataset, tokenizer, model):

        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = {'validation': tokenized_val}



        trainer = Trainer(model = model, tokenizer = tokenizer)
        pred =  trainer.predict(tokenized_datasets['validation'])
        predictions, labels = pred.predictions, pred.label_ids
        return predictions, labels

    def compute_accuracy(self, dataset, model_name, model, tokenizer, tc=None, round=None, iteration=None, output_file=None):
        self.tokenizer = tokenizer
        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = {'validation': tokenized_val}

        trainer = Trainer(model = model, tokenizer = tokenizer)
        pred =  trainer.predict(tokenized_datasets['validation'])
        predictions, labels = pred.predictions, pred.label_ids

        results = np.mean(np.argmax(predictions, axis=1) == labels)

        print(f'{model_name},{round},{iteration},accuracy,{tc},{results}')
        if(output_file is not None):
            output_file.write(f'{model_name},{round},{iteration},accuracy,{tc},{results}\n')

        # results = mean_squared_error(predictions, labels, squared=False)
        # print('mean_squared_error', results)
        # output_file.write(f'{name},{round},{iteration},mean_squared_error,{tc},{results}\n')
        # results = r2_score(predictions, labels)
        # print('r2', results)
        # output_file.write(f'{name},{round},{iteration},r2,{tc},{results}\n')

        # for i in range(len(labels)):
        #     if(labels[i] != np.argmax(predictions, axis=1)[i]):
        #         print('wrong prediction', labels[i], np.argmax(predictions, axis=1)[i])
        #         print(dataset[i]['review_body'])
        #         print(dataset[i]['product_category'])
        return results

    def validation_accuracy(self, topics, model_name, model, tokenizer, round=None, iteration=None, output_file=None, label_filter=False):
        print('output file is ', output_file)
        # raw_dataset = load_dataset('amazon_reviews_multi', 'en', cache_dir='/datadrive/research/ddbug/synthetic/amazon_multi/cache', split='validation')
        # raw_dataset = raw_dataset.filter(lambda x: x['label'] == 5 or x['label'] == 1)
        # #map 5 stars to positive and 1 star to negativ3e
        # raw_dataset = raw_dataset.map(lambda x: {'label': 1 if x['label'] == 5 else 0})

        # #fiter raw_dataset to only contains baby_product
        # if(topics!=None):

        if(label_filter):
            for sf in topics:
                raw_dataset_filter = self.load_data(topics = sf, split='validation', num_samples='all', label_filter=label_filter)
                self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,sf, round, iteration, output_file)
        else:
            for tc in topics:
                raw_dataset_filter = self.load_data(split='test', num_samples='all', topics=tc)
                self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,tc, round, iteration, output_file)



        # raw_dataset_filter = self.load_data(split='validation', num_samples=1000, topics=[topic])
        # self.compute_accuracy(raw_dataset_filter, model, tokenizer, topic, round, iteration, name, output_file)

    def train_model_from_scratch(self, raw_dataset, model_path, validation_dataset = None, 
                                just_upload=False, num_epochs=15, batch_size=16):
  
        #check if model_name path exists
        if not just_upload:


            # This trains a model and saves it.
            print('Training model from scratch')
            model_name_to_load = self.base_model()
            tokenizer = AutoTokenizer.from_pretrained(model_name_to_load)
            self.tokenizer = tokenizer

            tokenized_train = raw_dataset.map(self.tokenize_function, batched=True)



            if(validation_dataset is not None):

                tokenized_val = validation_dataset.map(self.tokenize_function, batched=True)






            # tokenized_datasets = {'train': tokenized_train}#, 'validation': tokenized_val}

            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            model = AutoModelForSequenceClassification.from_pretrained(model_name_to_load)#, num_labels=2) 
            metric = load_metric("accuracy")
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                return metric.compute(predictions=predictions, references=labels)
            training_args = TrainingArguments("test_trainer-from-scrach", 
                                            learning_rate=5e-6, 
                                            logging_steps=50,
                                            save_strategy='no', 
                                            # eval_steps=50, 
                                            # do_eval=True,
                                            # evaluation_strategy="steps",
                                            label_names=['label'], 
                                            per_device_train_batch_size=batch_size,
                                            num_train_epochs=num_epochs,
                                            weight_decay=0.01,
                                            # report_to="wandb",
                                            )
                            
            trainer = Trainer(
                model=model, 
                args=training_args,
                train_dataset=tokenized_train,
                # eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator, 
                # compute_metrics=compute_metrics,
            )
            trainer.train()
            trainer.save_model(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print('model is done')
        return tokenizer, model

    def chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_model_and_scorer(self, model, tokenizer, batch_size=128):
        classifier = transformers.pipeline(
            "sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True, device=0)

        labels = ['0', '1', '2', '3', '4']
        def classifier2(inputs, batch_size=batch_size):
            print('inputs are', inputs)
            preds = []
            for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):
                preds.extend(classifier(d))
            preds = [[float(x['score']) for x in y] for a, y in zip(inputs, preds)]
            for i in range(len(inputs)):
                print('input is ', inputs[i], preds[i])
            return np.array(preds)
        
        tensor_output_model = classifier2
        tensor_output_model.output_names = labels
        scorer = adatest.TextScorer(tensor_output_model)
        return tensor_output_model, scorer


if __name__ == "__main__":
    task = Review()
    name1, name2 = ('apparel', 'lawn_and_garden')
    tokenizer, global_model = task.train_model_from_scratch(raw_dataset = None, 
                                                    model_path=f'../fine_tuned_models/amazon_review_{name1}_{name2}_roberta_large', just_upload=True)  
    tokenizer, local_model = task.train_model_from_scratch(raw_dataset = None, 
                                                    model_path=f'../fine_tuned_models/amazon_review_{name1}_roberta_large', just_upload=True)  
    tensor_output_model, scorer = task.get_model_and_scorer(global_model, tokenizer)
    tensor_output_model_local, scorer_local = task.get_model_and_scorer(local_model, tokenizer)
    test_tree = adatest.TestTree('./tmp', auto_save=True)
    # dataset = task.load_data(topics=name1, split='train', num_samples=20)
    # adding_new_sentences_to_test_tree(test_tree, dataset['review_body'], [str(x) for x in dataset['label']], name1)
    task.load_data_from_test_tree(test_tree, scorer, predictor=tensor_output_model_local, topic=name1)

    



    # dataset = rv.load_data(topics='digital_ebook_purchase', num_samples='all')
    # print(len(dataset))
    # dataset_filter = dataset.filter(lambda x: x['product_category'] == 'baby_product')
    # dataset_filter = dataset.select(range(0,min(1000, len(dataset))))
    # dataset_filter = dataset_filter.map(lambda x: rv.add_sp1(x))
    
    # tokenizer, model = rv.train_model_from_scratch(dataset_filter, f'/datadrive/research/ddbug/synthetic/fine_tuned_models/classification_{rv.original_topic()}', just_upload=True)
    # tokenizer, model = rv.train_model_from_scratch(dataset_filter, f'/datadrive/research/ddbug/synthetic/amazon_multi/fine_tuned_model/global_{rv.name()}', just_upload=True)

    
    # topics = np.unique(dataset['product_category'])
    # # rv.validation_accuracy(topics, model, tokenizer)
    # # rv.compute_accuracy(dataset_filter, model, tokenizer, 'baby_product')


    # raw_dataset_filter = rv.load_data(split='validation', num_samples=1000, topics=rv.original_topic())
    # rv.compute_accuracy(raw_dataset_filter, model, tokenizer, rv.original_topic())

    # raw_dataset = load_dataset('amazon_reviews_multi', 'en', cache_dir='/datadrive/research/ddbug/synthetic/amazon_multi/cache', split='validation')
    # raw_dataset = raw_dataset.filter(lambda x: x['label'] == 5 or x['label'] == 1)
    # #map 5 stars to positive and 1 star to negativ3e
    # raw_dataset = raw_dataset.map(lambda x: {'label': 1 if x['label'] == 5 else 0})
    # dataset_filter = raw_dataset.map(lambda x: rv.add_sp2(x))
    # rv.compute_accuracy(dataset_filter, model, tokenizer, 'baby_product')


    #print the list of unique categories inthe dataset

    # topics = np.unique(dataset['product_category'])
    # for tc in topics:    
    #     dataset = rv.load_data(topics=tc)
    #     dataset_filter_1 = dataset.filter(lambda x: x['product_category'] == tc and x['label']==1)
    #     dataset_filter_2 = dataset.filter(lambda x: x['product_category'] == tc and x['label']==0)
    #     print(tc, len(dataset_filter_1), len(dataset_filter_2))


