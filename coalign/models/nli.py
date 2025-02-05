
from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict, Value
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
import transformers
import tqdm.auto as tqdm
import csv
import collections
from coalign.utils.test_tree_functions import adding_new_sentences_to_test_tree, generate_training_data
import adatest
import sentence_transformers

# print(int(time.time()*10000)%1000000000)
# random.seed(int(time.time()*100000)%1000000000)
# np.random.seed(int(time.time()*100000)%1000000000)
MAX_LENGTH = 200

class NLI():

    def name(self):
        return "nli"

    def original_topic(self):
        return 'normal'

    def base_model(self):
        return 'roberta-large'
    

    #load data from other sources returns a dataset of the shape 'premise', 'hypothesis', 'label'
    def read_from_other_sources(self, topic):
        eval_data = collections.defaultdict(lambda: ([], []))
        eval_med = collections.defaultdict(lambda: ([], []))

        file = open('../synthetic/marco_experiments/diagnostic-full.tsv')
        
        reader = csv.DictReader(file, delimiter='\t')
        map_label = {'neutral': 0, 'contradiction': 0, 'entailment': 1}
        # map_label = {'neutral': 1, 'contradiction': 0, 'entailment': 2}
        all_eval = set()
        all_diagnostics = []
        all_diag_labels = []

        for line in  reader:
            keys = ['Lexical Semantics', 'Predicate-Argument Structure', 'Logic', 'Knowledge' ]
            subgroups = []
            for k in keys:
                if line[k]:
                    subgroups.append(k)
                    subgroups.extend(line[k].split(';'))
            # if len(subgroups) > 1:
            #     print(subgroups)
            item = (line['Premise'], line['Hypothesis'])
            all_eval.add((line['Premise'].lower().strip(), line['Hypothesis'].lower().strip()))
            label = map_label[line['Label']]
            all_diagnostics.append(item)
            all_diag_labels.append(label)
            for g in subgroups:
                eval_data[g][0].append(item)
                eval_data[g][1].append(label)
        # eval_data['anli'] = (a_data, a_labels)
        # eval_data['sick'] = (s_data, s_labels)
        eval_data['diagnostics_all'] = (all_diagnostics, all_diag_labels)


        file = open('../synthetic/marco_experiments/MED.tsv')
        reader = csv.DictReader(file, delimiter='\t')
        map_label = {'neutral': 0, 'contradiction': 0, 'entailment': 1}
        i = 0 
        for line in reader:
            keys = ['upward_monotone', 'downward_monotone', 'disjunction', 'conjunction', 'conditionals', 'all', 'non_monotone']
            subgroups = ['MED-all']
            subgroups = []
        #     if not line['genre'].startswith('paper:FraCaS_GQ:'):
        #         continue
        #     if line['genre'].split(':')[0] != 'paper':
        #         continue
            for key in keys:
                if key in line['genre'].lower().split(':'):
                    subgroups.append('MED-' + key)
            item = (line['sentence1'], line['sentence2'])
            # if (line['sentence1'].lower().strip(), line['sentence2'].lower().strip()) in all_eval:
            #     continue
            label = map_label[line['gold_label']]
            for g in subgroups:
                eval_med[g][0].append(item)
                eval_med[g][1].append(label)
        print(eval_med.keys())
        print(eval_data.keys())
        if(topic in eval_med.keys()):
            ev = eval_med[topic]
        else:
            ev = eval_data[topic]
        ds = Dataset.from_dict({'premise' : [x[0] for x in ev[0]], 
                                'hypothesis': [x[1] for x in ev[0]], 
                                'label': ev[1]})
        print(ds.features)
        ds = ds.shuffle(seed=42)
        
        train_valid = ds.train_test_split(test_size=0.3, shuffle=False)
        train_valid_dataset = DatasetDict({
            'train':      train_valid['train'],
            'validation': train_valid['test']})

        return train_valid_dataset


    #load a dataset from a file_path
    def load_dataset_from_file_name(self, file_path):
        name = ''.join(x for x in file_path if x.isalnum() or x == '/' or x == '_' or x == '.')
        #sometimes file_path has weird characters and it causes error so here we just rename it!

        print("file_path is", file_path)
        print("new file path", name)
        if(name != file_path):
            f1 = open(file_path, 'r')
            f2 = open(name, 'w')
            f2.write(f1.read())
            f1.close()
            f2.close()

        data_files = {"train": name}
        review_dataset = load_dataset("json", data_files=data_files, split="train")
        #in review dataset the stars column type changes so we need to cast it
        review_dataset = review_dataset.cast_column("label", Value(dtype='int32', id=None))
        return review_dataset


        # data_files = {"train": file_path}
        # dataset = load_dataset("json", data_files=data_files, split="train")
        # return dataset
        
    def load_data(self, topics='normal', split='train', num_samples=500, label_filter=None):

        if(topics != 'normal'):
            raw_dataset = self.read_from_other_sources(topics)
            raw_dataset = raw_dataset[split]
            print('in load data no normal', raw_dataset)
            print(raw_dataset.features)
        else:
            if(split!='train'):
                split = split+'_matched'
            raw_dataset = load_dataset('glue', 'mnli', split=split)
            mapz = {0: 1, 1:0, 2:0, -1: -1}
            map_label = {'neutral': 0, 'contradiction': 0, 'entailment': 1}
            raw_dataset = raw_dataset.remove_columns("idx")


            raw_dataset = raw_dataset.map(lambda x: {'label': mapz[x['label']]})
            raw_dataset = raw_dataset.cast_column('label', Value(dtype='int64', id=None))
            print('in load data real normal', raw_dataset)


        
  
        #shuffle the datset
        # np.random.seed(int(time.time()*1000000)%1000000000)
        raw_dataset = raw_dataset.shuffle(seed = (int(time.time()*100000)%1000000000))


        if(num_samples!='all'):# and split=='train'):
            raw_dataset = raw_dataset.select(range(0,min(num_samples, len(raw_dataset))))
        return raw_dataset


    def tokenize_function(self, examples):
        return self.tokenizer(examples["premise"], examples['hypothesis'], truncation=True)
    
    def chunks(self,l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_model_and_scorer(self, model, tokenizer, batch_size=128):
        nli_tk = tokenizer
        nli_model = model
        classifier = transformers.pipeline("sentiment-analysis", 
                                            model=nli_model, 
                                            tokenizer=nli_tk, 
                                            framework="pt", 
                                            device=0, 
                                            return_all_scores=True)
                                            
        def classifier2(inputs, batch_size=batch_size):
            inputs = [dict(zip(['text', 'text_pair'], x.split(' | '))) for x in inputs]
            inputs = [x if len(x) == 2 else {'text': 'aaa', 'text_pair': 'aaa'} for x in inputs]
            preds = []
            for d in tqdm.tqdm(list(self.chunks(inputs, batch_size))):
                preds.extend(classifier(d))
            preds = np.array([[float(x['score']) for x in y] if a != ('a', 'a') else [0, 0] for a, y in zip(inputs, preds)])
            return preds
    #         new_preds = np.zeros((preds.shape[0], 2))
    #         new_preds[:, 1] = preds[:, 0]
    #         new_preds[:, 0] += preds[:, 1]
    #         new_preds[:, 0] += preds[:, 2]
    #         return new_preds
        tensor_output_model = classifier2
        labels = ['not entailment', "entailment"]
        tensor_output_model.output_names = labels
        scorer = adatest.TextScorer(tensor_output_model)
        return tensor_output_model, scorer


    def fine_tune_model(self, dataset, original_model_path, output_path, 
                        just_upload=False, batch_size=8,num_epochs=int(30)):

        if(just_upload):
            return AutoTokenizer.from_pretrained(original_model_path), AutoModelForSequenceClassification.from_pretrained(output_path)

            
        #load the model from original_model_path
        model = AutoModelForSequenceClassification.from_pretrained(original_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(original_model_path)

        tokenizer = AutoTokenizer.from_pretrained(original_model_path)

        tokenized_train = dataset.map(self.tokenize_function, batched=True)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(original_model_path, num_labels=2)
        
        sd = np.random.randint(1000000000)
        print('seed is', sd)
        training_args = TrainingArguments("test_trainer-fine-tune", 
                                        learning_rate=5e-6, 
                                        logging_steps=50,
                                        save_strategy='no', 
                                        # eval_steps=500, 
                                        # evaluation_strategy="epoch",
                                        label_names=['label'], 
                                        per_device_train_batch_size=batch_size,
                                        num_train_epochs=num_epochs,
                                        weight_decay=0.01,
                                        # report_to="wandb"
                                        seed = sd,
                                        data_seed = sd+1,
                                        )
                        
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_train,
            tokenizer=tokenizer,
            data_collator=data_collator, 
        )
        trainer.train()
        trainer.save_model(output_path)
        return tokenizer, model


    # def tokenize_function(self, examples,):
    #     return tokenizer(examples["sentence"], truncation=True)

    def compute_predictions(self,dataset, tokenizer, model):
        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        trainer = Trainer(model = model, tokenizer = tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=4))
        pred =  trainer.predict(tokenized_val)
        predictions, labels = pred.predictions, pred.label_ids
        return predictions, labels

    def compute_accuracy(self, dataset, model_name, model, tokenizer, tc=None, round=None, iteration=None, 
                        output_file=None, batch_size=4):
        self.tokenizer = tokenizer
        tokenized_val = dataset.map(self.tokenize_function, batched=True)
        trainer = Trainer(model = model, tokenizer = tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=batch_size))
        pred =  trainer.predict(tokenized_val)
        predictions, labels = pred.predictions, pred.label_ids
        results = np.mean(np.argmax(predictions, axis=1) == labels)
        
        print(f'{model_name},{round},{iteration},accuracy,{tc},{results}')
        if(output_file is not None):
            output_file.write(f'{model_name},{round},{iteration},accuracy,{tc},{results}\n')
        return results


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
        # if(label_filter):
        #     for sf in label_filter:
        #         # raw_dataset_filter = self.load_data(split='validation', num_samples=1000, label_filter=sf, spurious=True)
        #         # self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,sf, round, iteration, output_file, spurious=True)
        #         raw_dataset_filter = self.load_data(split='validation', num_samples='all', label_filter=sf)
        #         self.compute_accuracy(raw_dataset_filter, model_name, model, tokenizer,sf, round, iteration, output_file)


    def generate_initial_test_tree(self, test_tree):
        #########adding faracas data to test tree ##########
        # df = pd.read_csv('/datadrive/research/ddbug/synthetic/marco_experiments/MED.tsv', sep='\t')
        # df = df[df['genre'] == 'paper:FraCaS_GQ:downward_monotone']
        # for index, x in df.iterrows():
        #     print(x['sentence1'], x['sentence2'], x['gold_label'])
        #     if(x['gold_label'] == 'entailment'):
        #         label = 'entailment'
        #     else:
        #         label = 'not entailment'
        #     adding_new_sentences_to_test_tree(test_tree, 
        #                                         [x['sentence1'] + ' | ' + x['sentence2']], 
        #                                         [label], 
        #                                         'dm')

        ########## adding 500 random examples ############
        data = self.load_data(split='train', num_samples=500, topics = 'MED-downward_monotone')
        data = data.map(lambda x: {'label': 'entailment' if x['label'] == 1 else 'not entailment'})
        data = data.map(lambda x: {'together': x['premise'] + ' | ' + x['hypothesis']})

        adding_new_sentences_to_test_tree(test_tree, data['together'], data['label'], 'dm')

    #generate bunch of examples from test_tree and choose the ones that glbal and local disagree to add to test_tree
    def add_data_to_test_tree(self, test_tree, global_model, local_model, tokenizer, topic, k):
        tensor_output_model_global, scorer_global = self.get_model_and_scorer(global_model, tokenizer)
        tensor_output_model_local, scorer_local = self.get_model_and_scorer(local_model, tokenizer)

        roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
        num_added = 0
        for i in range(10): #do this 5 times, return when it reaches k data points
            browser = test_tree(scorer=scorer_global, embedding_model=roberta_embed_model, max_suggestions=100, 
                recompute_scores=True, predictor= tensor_output_model_local)
            suggestions = browser._generate_suggestions(topic)
            disagreement = 0
            for index, row in suggestions.iterrows():
                try:
                    if(float(row['score']) > 0 and len(row['value1'].split(' | '))==2): #it means we have disagreements
                        disagreement += 1
                        print(row)
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




    def load_data_from_test_tree(self, test_tree, scorer, predictor, topic):

        print('launching browser')
        roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
        browser = test_tree(scorer=scorer, embedding_model=roberta_embed_model, max_suggestions=100, 
        recompute_scores=True, predictor = predictor)
        suggestions = browser._generate_suggestions(topic)
        disagreement = 0
        for index, row in suggestions.iterrows():
            try:
                if(float(row['score']) > 0 and len(row['value1'].split(' | '))==2): #it means we have disagreements
                    disagreement += 1
                    print(row)
                    print('disagreement found--------------------------')
                    print(row['value1'], row['value2'], row['score value1 outputs'])
                    adding_new_sentences_to_test_tree(test_tree, [row['value1']], [row['value2']], topic)
            except ValueError:
                print('not float I guess!', row['score'])

        print('disagreement found', disagreement/len(suggestions))
        return test_tree

        # adding_new_sentences_to_test_tree(test_tree, list1, list2, topic)
        # adatest_to_dictionary(task, test_tree)

    def train_model_from_scratch(self, train_dataset, model_path, validation_dataset = None, just_upload=False, num_epochs=15):
  
        #check if model_name path exists
        if not just_upload:

            model_name_to_load = self.base_model()
            tokenizer = AutoTokenizer.from_pretrained(model_name_to_load, cache_dir='/datadrive/research/ddbug/synthetic/amazon_multi/cache')
            model = AutoModelForSequenceClassification.from_pretrained(model_name_to_load, cache_dir='/datadrive/research/ddbug/synthetic/amazon_multi/cache', num_labels=2)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

            tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
            tokenized_val = validation_dataset.map(self.tokenize_function, batched=True)
            training_args = TrainingArguments("test_trainer-from-scrach", learning_rate=5e-6, 
                                            logging_steps=50,
                                            save_strategy='no', 
                                            eval_steps=500, 
                                            evaluation_strategy="epoch",
                                            label_names=['label'], 
                                            per_device_train_batch_size=4,
                                            num_train_epochs=num_epochs,
                                            weight_decay=0.01,
                                            # report_to="wandb"
                                            )
                            
            trainer = Trainer(
                model=model, args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_val,
                tokenizer=tokenizer,
                data_collator=data_collator, 
                compute_metrics=self.compute_metrics,
            )
            trainer.train()
            trainer.save_model(model_path)

        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        print('model is done')
        return tokenizer, model


#if this is the main function
if __name__ == "__main__":
    task = NLI()

    test_tree = adatest.TestTree('tmp_nli3.csv', auto_save=True)
    print(test_tree)
    # f = open('./tmp_nli.txt', "a").close()


    ###add 50 data from topics and normal
    topic = 'MED-upward_monotone'
    dataset = task.load_data(topics=topic, split='train', num_samples=50)
    inputs = [dataset[i]['premise']+' | '+dataset[i]['hypothesis'] for i in range(len(dataset))]
    mapz = {0: 'not entailment', 1: 'entailment'}
    adding_new_sentences_to_test_tree(test_tree, inputs, [mapz[x] for x in dataset['label']], topic)
    tokenizer, local_model = task.train_model_from_scratch(None, 
                                    model_path=f'../fine_tuned_models/local_{topic}_roberta_large', just_upload=True)  
    tensor_output_model_local, scorer_local = task.get_model_and_scorer(local_model, tokenizer)    
    roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')

    browser = test_tree(scorer=scorer_local, embedding_model=roberta_embed_model, max_suggestions=100, 
        recompute_scores=True, predictor = None)
    exit(0)

    topic = 'MED-upward_monotone'
    original_model_path = f'../fine_tuned_models/global_nli_all_roberta_large'

    topics = ['normal', 'MED-upward_monotone', 'MED-downward_monotone']

    for i in range(3):

        #global and local model! 
        topic = 'MED-upward_monotone'
        tokenizer, local_model = task.train_model_from_scratch(None, 
                                        model_path=f'../fine_tuned_models/local_{topic}_roberta_large', just_upload=True)  
        tensor_output_model_local, scorer_local = task.get_model_and_scorer(local_model, tokenizer)

        tokenizer, global_model = task.train_model_from_scratch(None, 
                                        model_path=f'../fine_tuned_models/global_nli_all_roberta_large', just_upload=True)  
        tensor_output_model_global, scorer_global = task.get_model_and_scorer(global_model, tokenizer)

        test_tree = adatest.TestTree('./tmp_nli', auto_save=True)


        ###add 50 data from topics and normal
        topic = 'MED-upward_monotone'
        dataset = task.load_data(topics=topic, split='train', num_samples=50)
        inputs = [dataset[i]['premise']+' | '+dataset[i]['hypothesis'] for i in range(len(dataset))]
        mapz = {0: 'not entailment', 1: 'entailment'}
        adding_new_sentences_to_test_tree(test_tree, inputs, [mapz[x] for x in dataset['label']], topic)

        topic = 'normal'
        dataset = task.load_data(topics=topic, split='train', num_samples=50)
        inputs = [dataset[i]['premise']+' | '+dataset[i]['hypothesis'] for i in range(len(dataset))]
        mapz = {0: 'not entailment', 1: 'entailment'}
        adding_new_sentences_to_test_tree(test_tree, inputs, [mapz[x] for x in dataset['label']], topic)

        for i in range(3):
            task.load_data_from_test_tree(test_tree, scorer_global, predictor=tensor_output_model_local, topic='normal')
            task.load_data_from_test_tree(test_tree, scorer_local, predictor=tensor_output_model_global, topic='MED-upward_monotone')


        ds_normal, _ = generate_training_data(task, 'normal', test_tree)
        ds_upward, _ = generate_training_data(task, 'MED-upward_monotone', test_tree)
        ds = concatenate_datasets([ds_normal, ds_upward])
        print(ds)
        model = task.fine_tune_model(ds, original_model_path, original_model_path, just_upload=False, num_epochs=10, batch_size=8)
        tokenizer = AutoTokenizer.from_pretrained(original_model_path)

        f = open(f'../results/tmp.txt', 'a')
        task.validation_accuracy(topics, 'all', model, tokenizer, output_file=f)
        f.close()
