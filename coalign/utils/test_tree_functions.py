import os
import sys
from statistics import mode
import argparse
import json
from datasets import load_dataset, Dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import transformers
import pickle
# from checklist.test_suite import TestSuite
# from checklist.pred_wrapper import PredictorWrapper
import json
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
import tqdm.auto as tqdm
import os
import sentence_transformers
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import transformers
import numpy as np
import transformers
import importlib
import urllib
import csv
import collections
import random
import re
import uuid
import pandas as pd
from munch import Munch
import shutil

# Put adatest data into a per-topic dictionary
def adatest_to_dictionary(task, test_tree):
    topic_data = collections.defaultdict(lambda: [])
    topic_label = collections.defaultdict(lambda: [])
    # for test in test_tree._tests[test_tree._tests['score'] > 0].iterrows():
    # removing score restriction for v3
    for test in test_tree._tests.iterrows():
        x, y, topic = test[1]['value1'], test[1]['value2'], test[1]['topic']
        print('x, y, value1, value2, topic', x,y,topic)
        # topic = topic.replace('/', '-')
        if(task.name() == 'sentiment'):
            mapz = {'positive': 1,  'negative': 0} #TODO 'neutral': 1,
            topic_label[topic].append(mapz[y])
            topic_data[topic].append(x)
        elif(task.name() == 'toxicity'):
            mapz = {'non-toxic': 0, 'toxic': 1} #TODO
            topic_label[topic].append(mapz[y])
            topic_data[topic].append(x)
        elif(task.name() == 'qqp'):
            mapz = {'duplicate': 1, 'not duplicate': 0}
            topic_data[topic].append(tuple(x.split(' | ')))
            topic_label[topic].append(mapz[y])
        elif(task.name() == 'review'):
            # if(y=='2'):
            topic_data[topic].append(x)
            topic_label[topic].append(int(y))
        elif(task.name() == 'nli'):
            if(len(x.split('|'))==2):
                mapz = {'not entailment': 0, 'entailment': 1}
                topic_data[topic].append(tuple(x.split('|')))
                topic_label[topic].append(mapz[y])
        
    
    print('keys', topic_data.keys(), topic_label.keys())
    return topic_data, topic_label


def launching_browser(task, test_tree, scorer, predictor, dport = 5000):
    print('launching browser')
    roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')
    browser = test_tree(task = task, scorer=scorer, embedding_model=roberta_embed_model, max_suggestions=100, 
    recompute_scores=True, predictor = predictor)
    import adatest
    adatest.serve(browser, host='0.0.0.0', port=dport, )


# Create new training data using the topics, 
#mix it with an equal quantity of original training data 
# if there is no topic then use all original training data
def generate_training_data(task, topic, test_tree, 
                            num=0,   multiplier=1):
    print('in generating training data for topic', topic)
    topic_data, topic_label = adatest_to_dictionary(task, test_tree)
    with_orig = False
    if(topic==None):
        subjs = list(topic_label.keys())
        with_orig=True
        print(f'number of subjects is {len(subjs)}')
        print(f'subjs are, {subjs}')
    else:
        subjs=[]
        for tpc in list(topic_label.keys()):
            print('topic, tpc', topic, tpc)
            if(topic in tpc):
                subjs.append(tpc)
        print(f'number of subjects is {len(subjs)}')
        print(f'subjs are, {subjs}')

    training = []
    train_labels = []
    evalz = {}
    # you could restrict it to specific topics here if you wanted
    for topic in subjs:
        print('topic is', topic)
        td = [x for x in topic_data[topic]]
        tl = [x for x in topic_label[topic]]
        training.extend(td)
        train_labels.extend(tl)
    # log.debug('size of training')
    # log.debug(len(training))

    #add data from the original model
    raw_datasets = task.load_data(num_samples = len(training))         

    if(task.name() == 'sentiment'):
        if(with_orig):
            n_add = max(len(training), num)
            to_add = [int(i) for i in np.random.choice(len(raw_datasets['sentence']), int(n_add) * 1, replace=False)]
            # map_label=  {0: 2, 1:1, 2:0}
            mapz = {'positive': 1,  'negative': 0} #'neutral': 1,
            training += [raw_datasets['sentence'][i] for i in to_add]
            train_labels += [raw_datasets['label'][i] for i in to_add]

        dictz = {'sentence': training,
        'label': train_labels,
        }

    if(task.name() == 'toxicity'):
        if(with_orig):
            n_add = max(len(training), num)
            to_add = [int(i) for i in np.random.choice(len(raw_datasets['sentence']), int(n_add) * 1, replace=False)]
            training += [raw_datasets['sentence'][i] for i in to_add]
            train_labels += [raw_datasets['label'][i] for i in to_add]

        dictz = {'sentence': training,
        'label': train_labels,
        }

    if(task.name() == 'review'):
        raw_datasets = task.load_data(topics=task.original_topic(), label_filter=True)
        print('raw_datasets', raw_datasets)
        print('len(raw_datasets)', len(raw_datasets))

        if(with_orig):
            n_add = max(len(training), num)
            to_add = [int(i) for i in np.random.choice(len(raw_datasets['review_body']), int(multiplier * n_add) * 1, replace=False)]
            print('to add', to_add)
            training += [raw_datasets['review_body'][i] for i in to_add]
            train_labels += [raw_datasets['label'][i] for i in to_add]

        dictz = {'review_body': training,
        'label': train_labels,
        }

    if(task.name() == 'qqp'):
        if(with_orig):
            n_add = max(len(training), num)
            to_add = [int(i) for i in np.random.choice(len(raw_datasets), len(training) * 1, replace=False)]
            # map_label=  {0: 2, 1:1, 2:0}
            map_label=  {0: 0, 1:1, 2:2}
            training += [(raw_datasets['question1'][i], raw_datasets['question2'][i]) for i in to_add]
            train_labels += [map_label[raw_datasets['label'][i]] for i in to_add]

        dictz = {'question1' : [x[0] for x in training], 
                'question2': [x[1] for x in training], 
                'label': train_labels}

    if(task.name() == 'nli'):
        if(with_orig):
            n_add = max(len(training), num)
            to_add = [int(i) for i in np.random.choice(len(raw_datasets), len(training) * 1, replace=False)]
            map_label=  {0: 0, 1:1}
            # map_label={"not entailment":0, "entailment":1}

            training += [(raw_datasets['premise'][i], raw_datasets['hypothesis'][i]) for i in to_add]
            train_labels += [map_label[raw_datasets['label'][i]] for i in to_add]
        print(training)
        dictz = {'premise' : [x[0].strip() for x in training], 
                'hypothesis': [x[1].strip() for x in training], 
                'label': train_labels}
        
        
    print('training data is:')
    print(dictz)

    ntrain = Dataset.from_dict(dictz)
    return ntrain, dictz

#generate a fine tune model for a topic or if topic=None then for eveyone + original
def generate_fine_tuned_models_of_a_file(task, test_tree, topic, original_model_path, output_path, 
                                        multiplier=1, num_epochs=30, batch_size=8):

    print('generating fine tuned model for topic', topic)
    ntrain, _ = generate_training_data(task, topic, test_tree, multiplier = multiplier)
    print('len of ntrain is', len(ntrain))
    for i in range(len(ntrain)):
        print(ntrain[i])
    print('ntrain', ntrain)
    model = task.fine_tune_model(ntrain, original_model_path, output_path, num_epochs=num_epochs, batch_size=batch_size)
    return ntrain, model


def adding_new_sentences_to_test_tree(test_tree, list1, list2, topic):
    for i in range(len(list1)):
        row = {
            "topic": '/'+topic,
            "prefix":  "The model output for",
            "value1": list1[i],
            "comparator": 'should be',
            "value2": list2[i].strip().lower(),
            "labeler": "imputed"
            # "focus": 0,
            # 'seen': False,
            # 'batch_round': -1,
            # 'label_round': -1,
            # 'focus_topic': "",
            # 'score': np.nan
        }
        # for c in test_tree.score_columns:
        #     row[c] = np.nan
        #     row[c + " value1 outputs"] = "{}"
        #     row[c + " value2 outputs"] = "{}"
        new_add = pd.DataFrame([row], index=[uuid.uuid4().hex])

        test_tree._tests = test_tree._tests.append(new_add, sort=False)
    #save test_tree in the csv file
    test_tree.to_csv(test_tree._tests_location)
    print('final test tree is for topic', topic, len(test_tree._tests))

