from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import transformers
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification
import tqdm.auto as tqdm
import adatest
import datasets
import sentence_transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from coalign.utils.test_tree_functions import adding_new_sentences_to_test_tree
from coalign.models import review
MAX_LENGTH = 200

from coalign.models.abstract_task import Abstract_task

class Sentiment(Abstract_task):

    def name(self):
        return 'sentiment'
        
    def chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def map_label_sentiment(self, score):
        if score <= .4:
            return 0
        elif score <= .6:
            return 1
        else:
            return 2

    def original_topic(self):
        return 'sst2'

    def load_data(self, split='train', num_samples=1000, topics='all', label_filter=False):
        # dataset = load_dataset('sst2', split=split)

        rv = review.Review()
        dataset = rv.load_data(topics = 'all', num_samples='all', split='train')
        list_to_filter = [#'expensive', 'cheap', 'price', 'cost', 'money', 'worth', 'value', 'pay', 'paid',
                    'easy to install', 'hard to install']
        dataset = dataset.filter(lambda x: any(y in x['sentence'] for y in list_to_filter))
        ds_pos = dataset.filter(lambda x: x['label'] == 1)
        ds_neg = dataset.filter(lambda x: x['label'] == 0)
        print(len(ds_pos), len(ds_neg))
        num = min(len(ds_pos), len(ds_neg))
        dataset = datasets.concatenate_datasets([ds_pos.select(range(num)), 
                                        ds_neg.select(range(num))])
        print('ds original', dataset)
        print(dataset[0])
        if(num_samples!='all'):# and split=='train'):
            dataset = dataset.select(range(0,min(num_samples, len(dataset))))
        return dataset


        #preprocess:
        # raw_dataset = raw_dataset.map(lambda x: {'label': int(self.map_label_sentiment(x['label']))})
        # raw_dataset = raw_dataset.cast_column("label", Value(dtype='int32', id=None))

        #cast the label to int
        #shuffle the datset
        # np.random.seed(int(time.time()*1000000)%1000000000)
        raw_dataset = raw_dataset.shuffle(seed = (int(time.time()*100000)%1000000000))


        if(num_samples!='all'):# and split=='train'):
            raw_dataset = raw_dataset.select(range(0,min(num_samples, len(raw_dataset))))
        return raw_dataset

    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], truncation=True, max_length=MAX_LENGTH)

    def get_model_and_scorer(self, model, tokenizer, batch_size=128):
        # classifier = transformers.pipeline(
        #     "sentiment-analysis", model=model, tokenizer=tokenizer, return_all, device=0)
        classifier = transformers.pipeline("sentiment-analysis", 
                                model=model, 
                                tokenizer=tokenizer, 
                                framework="pt", 
                                device=0, 
                                top_k=None,
                                # return_all_scores=True
                                )

        labels=[]
        labels = ['negative',  'positive']#'neutral',
        def classifier2(inputs, batch_size=batch_size):
            print('inputs are', inputs)
            preds = []
            for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):

                result = classifier(d)
                print('result is', result)
                labels = ['NEGATIVE', 'POSITIVE', 'LABEL_0', 'LABEL_1']
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
        
        tensor_output_model = classifier2
        tensor_output_model.output_names = labels
        scorer = adatest.TextScorer(tensor_output_model)
        return tensor_output_model, scorer
    
    def compute_predictions(self,dataset, tokenizer, model):
        self.tokenizer = tokenizer
        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        trainer = Trainer(model = model, tokenizer = tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=4))
        pred =  trainer.predict(tokenized_val)
        predictions, labels = pred.predictions, pred.label_ids
        return predictions, labels

    def compute_accuracy(self, dataset, model_name, model, tokenizer, tc=None, round=None, iteration=None, output_file=None):


        self.tokenizer = tokenizer
        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        tokenized_datasets = {'validation': tokenized_val}

        trainer = Trainer(model = model, tokenizer = tokenizer)
        print(tokenized_datasets['validation'])
        print(tokenized_datasets['validation'][0:20])
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

        self.tokenizer = tokenizer
        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        trainer = Trainer(model = model, tokenizer = tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=4))
        pred =  trainer.predict(tokenized_val)
        predictions, labels = pred.predictions, pred.label_ids
        results = np.mean(np.argmax(predictions, axis=1) == labels)
        
        print(f'{model_name},{round},{iteration},accuracy,{tc},{results}')
        # print(f'{model_name},{tc},{results}')

        if(output_file is not None):
            output_file.write(f'{model_name},{round},{iteration},accuracy,{tc},{results}\n')
            # output_file.write(f'{model_name},{tc},{results}\n')

        for i in range(len(labels)):
            if(labels[i] != np.argmax(predictions, axis=1)[i]):
                print('wrong!', 'true label=',labels[i], 'prediction=',np.argmax(predictions, axis=1)[i])
                print(dataset[i]['sentence'])
                # print(dataset[i]['product_category'])
            # else:
            #     print('correct prediction', labels[i], np.argmax(predictions, axis=1)[i])
            #     print(dataset[i]['review_body'])
            #     print(dataset[i]['product_category'])
        return results, np.where(labels != np.argmax(predictions, axis=1))[0]

    def validation_accuracy(self, topics, model_name, model, tokenizer, round=None, iteration=None, output_file=None, label_filter=None):
        print('output file is ', output_file)
        # raw_dataset = load_dataset('amazon_reviews_multi', 'en', cache_dir='/datadrive/research/ddbug/synthetic/amazon_multi/cache', split='validation')
        # raw_dataset = raw_dataset.filter(lambda x: x['label'] == 5 or x['label'] == 1)
        # #map 5 stars to positive and 1 star to negativ3e
        # raw_dataset = raw_dataset.map(lambda x: {'label': 1 if x['label'] == 5 else 0})

        # #fiter raw_dataset to only contains baby_product
        if(topics!=None):
            for tc in topics:
                raw_dataset_filter = self.load_data(split='validation', topics=tc, num_samples=1000)
                self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,tc, round, iteration, output_file)

        ds_biased, ds_biased_all, ds_unbiased_all, ds_test_all, ds1, ds2, ds3, ds4 = generate_datasets(num=5)
        self.compute_accuracy(ds_test_all, model_name, model, tokenizer,f'ds_test_all', round, iteration, output_file)
        datasets = [ds1, ds2, ds3, ds4]
        for i in range (4):
            self.compute_accuracy(datasets[i], model_name, model, tokenizer,f'ds_{i+1}', round, iteration, output_file)


        # if(label_filter):
        #     for sf in label_filter:
        #         # raw_dataset_filter = self.load_data(split='validation', num_samples=1000, label_filter=sf, spurious=True)
        #         # self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,sf, round, iteration, output_file, spurious=True)
        #         raw_dataset_filter = self.load_data(split='validation', num_samples='all', label_filter=sf)
        #         self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,sf, round, iteration, output_file)

    def fine_tune_model(self, review_dataset, original_model_path, 
                        output_path, just_upload=False, num_epochs=30, batch_size=4):

        if(just_upload):
            return AutoTokenizer.from_pretrained(output_path), AutoModelForSequenceClassification.from_pretrained(output_path)

            
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
                                        per_device_train_batch_size=batch_size,
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
        return tokenizer, model

    #generate bunch of examples from test_tree and choose the ones that glbal and local disagree to add to test_tree
    def add_data_to_test_tree(self, test_tree, global_model, local_model, global_tokenizer, local_tokenizer, topic, k):
        tensor_output_model_global, scorer_global = self.get_model_and_scorer(global_model, global_tokenizer)
        tensor_output_model_local, scorer_local = self.get_model_and_scorer(local_model, local_tokenizer)

        roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
        num_added = 0
        for i in range(10): #do this 5 times, return when it reaches k data points
            browser = test_tree(scorer=scorer_global, embedding_model=roberta_embed_model, max_suggestions=100, 
                recompute_scores=True, predictor= tensor_output_model_local)
            suggestions = browser._generate_suggestions(topic)
            disagreement = 0
            for index, row in suggestions.iterrows():
                try:
                    if(float(row['score']) > 0 and #disagreement
                       len(row['value1'].split(':'))==2 and #write format
                       ('skin' in row['value1'] or 'battery' in row['value1']) #stay in topic
                       ): 
                        disagreement += 1
                        print('row is', row)
                        print('disagreement found--------------------------')
                        print(row['value1'], row['value2'], row['score value1 outputs'])
                        adding_new_sentences_to_test_tree(test_tree, [row['value1']], [row['value2']], topic)
                        num_added += 1
                        if(num_added >= k):
                            return test_tree
                except ValueError:
                    print('not float I guess!', row['score'])

            print('disagreement found in round ',i, disagreement/len(suggestions))
        test_tree.to_csv(test_tree._tests_location)
        return test_tree

    
    # This trains a model and saves it.
    def train_model_from_scratch(self, raw_dataset, model_path, just_upload=False, batch_size=8, num_epochs=15):
        if(just_upload):
            return self.upload_pretrained_model(self, model_name)

        abstract_task.logger.info(f'the raw dataset is: {raw_dataset}')
        abstract_task.logger.info(f'the first instnace is {raw_dataset[0]}')

        #some preprocessing for sentiment task
        # train_labels = np.array([self.map_label_sentiment(x) for x in raw_dataset['label']])
        train_labels = np.array(raw_dataset['label'])

        train_sentences = list(raw_dataset['sentence'])
        # val_labels = np.array([self.map_label_sentiment(x) for x in raw_dataset['validation']['label']])
        # val_sentences = list(raw_dataset['sentence'])
        train = Dataset.from_dict({'sentence' : train_sentences, 'label': train_labels})
        # val = Dataset.from_dict({'sentence' : val_sentences, 'label': val_labels})


        model_name_to_load = 'roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(model_name_to_load)
        self.tokenizer = tokenizer

        tokenized_train = train.map(self.tokenize_function, batched=True)
        # tokenized_val = val.map(self.tokenize_function, batched=True)
        tokenized_datasets = {'train': tokenized_train}#, 'validation': tokenized_val}
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_to_load)
        
        training_args = TrainingArguments("test_trainer", learning_rate=5e-6, logging_steps=300,
                                        save_strategy='no', eval_steps=500,
                                        label_names=['label'], 
                                        per_device_train_batch_size=batch_size,
                                        num_train_epochs=num_epochs,
                                        weight_decay=0.01,
                                        )
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_datasets['train'],
            # eval_dataset=tokenized_datasets['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator, 
        )
        trainer.train()
        trainer.save_model(model_path)
        print('model is done')
        return tokenizer, model


        # Reload the original finetuned checkpoint in case you updated it somewhere in the notebook
        if(just_upload):
            return AutoModelForSequenceClassification.from_pretrained(new_model_name)
        raw_datasets = self.load_data()
        model_name = prev_model_name
        model_new = AutoModelForSequenceClassification.from_pretrained(model_name)#, num_labels=3)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # tensor_output_model, scorer = get_model_scorer(model, tokenizer)

        self.tokenizer = tokenizer


        tokenized_train = ntrain.map(self.tokenize_function, batched=True)
        # tokenized_val = raw_datasets.map(self.tokenize_function, batched=True).shuffle().select(range(200))
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        # model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        # model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=3)
        train_d = tokenized_train.shuffle()#.select(range(900))
        training_args = TrainingArguments("test_trainer", learning_rate=5e-6, logging_steps=50,
                                        save_strategy='no', eval_steps=500,
                                        label_names=['label'], 
                                        per_device_train_batch_size=8,
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

#write main function
if __name__ == '__main__':
    model_name = f'../fine_tuned_models/amazon_review_0_1_roberta_large'
    model_name2= "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    task = Sentiment()
    tensor_output_model, scorer = task.get_model_and_scorer(model, tokenizer)
    print('result is', tensor_output_model(["Sketchers Unisex Foldaway Golf Cart Walking Shoes - Black/Orange/Yellow by Skechers: My husband purchased these shoes to play golf in, but was concerned that the high tops may limit his mobility on the course. His concern was unwarranted as these shoes were extremely lightweight, the high tops allowed him to maintain the mobility he needed, he said they were comfortable and stylish. He would definitely recommend these shoes.",
   " Thumbs Up! - Great Glove: My husband is a forester and will be using it working outside in the woods so the tougher the glove the better. This is the perfect glove!! He loves it. No damages after use and extremely comfortable. While operating the glove it almost seems it is not there. I would say this was a very smart purchase."
    ,"WOW: The content is amazing, the speakers are dynamic and energetic. I also purchased the Deluxe version as a gift which is fantastic. Highly, highly recommend."
    ]))

    test_tree = adatest.TestTree('tmp.csv', auto_save=True)

    roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
    browser = test_tree(scorer=scorer, embedding_model=roberta_embed_model, max_suggestions=100, 
    recompute_scores=True, predictor = tensor_output_model)
    adatest.serve(browser, host='0.0.0.0', port=5000 )