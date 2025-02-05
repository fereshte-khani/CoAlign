import os
import argparse
from datasets import load_dataset, concatenate_datasets, Value
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from transformers import AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import torch
from coalign.models import nli,review, sentiment
import adatest
from utils.test_tree_functions import adding_new_sentences_to_test_tree, generate_training_data
from utils.amazon_preprocess import generate_datasets
from transformers import TrainingArguments, Trainer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--suffix")
parser.add_argument("--gpu", default='0', help="gpu id")
parser.add_argument("--strategy", default='ddbug', type=str, help="ddbug, random, or active_learning")
parser.add_argument("--dataset", default='nli', type=str, help="The dataset to use")

with open(os.path.expanduser('~/keys/.openai_api_key'), 'r') as file:
    adatest.backend = adatest.backends.OpenAI('davinci', api_key=file.read().replace('\n', ''))

args = parser.parse_args()
suffix = args.suffix
strategy = args.strategy
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)




with open(os.path.expanduser('~/keys/.openai_api_key'), 'r') as file:
    openai.api_key = file.read().replace('\n', '')


#generate a fine tune model for a topic or if topic=None then for eveyone + original
def generate_dataset(task, file_path_original, file_path, topics_to_filter, global_model=None,  tokenizer=None):

    print('in generating dataset')
    if(using_test_tree):
        test_tree = adatest.TestTree(f'{file_path}.test_tree', auto_save=True)
        print(test_tree)
        ds = None
        for topic in topics_to_filter:
            ds_normal, _ = generate_training_data(task, topic, test_tree)
            if(ds is None):
                ds = ds_normal
            else:
                ds = concatenate_datasets([ds, ds_normal])
        return ds

    #load the whole dataset
    print("file path is", file_path)
    print("task is", task.name())
    review_dataset = task.load_dataset_from_file_name(file_path)
    review_dataset = review_dataset.filter(lambda x: x['topic'] in topics_to_filter)
    return review_dataset

    #### right now we ignore the orig and stuff!
    k = len(review_dataset)
    print('review_dataset', review_dataset)
    with_orig = True
    if(topics_to_filter is not None):
        #filter dataset with the product category is one of the topics
        # review_dataset = review_dataset.filter(lambda x: x['product_category'] in topics_to_filter)
        with_orig=False



    #add data from the original model
    if(with_orig):
        if(global_model is not None):
            # np.random.seed(int(time.time()*1000000)%1000000000)

            orig_dataset = task.load_data(topics=task.original_topic(), num_samples = k*20, label_filter=label_filter)
            tokenized_val = orig_dataset.map(task.tokenize_function, batched=True)
            global_trainer = Trainer(global_model, tokenizer = tokenizer)
            pred = global_trainer.predict(tokenized_val)
            predictions, labels = pred.predictions, pred.label_ids
            y_global_p = softmax(predictions, axis=1)
            labels = np.array(labels)
            weights = 1 - y_global_p[np.arange(len(labels)), labels]
            weights = 1 - np.max(y_global_p, axis=1)
            diff_indices = np.random.choice(len(weights), int(k*coef_for_orig_data), p=weights/np.sum(weights), replace=False)

            print('sentences that are near the border = ')
            for i in diff_indices:
                print(orig_dataset[int(i)], weights[i], y_global_p[i], labels[i])
            orig_dataset = orig_dataset.select(diff_indices)
        else:
            # orig_dataset = task.load_data(topics = task.original_topic(), num_samples = k, label_filter=label_filter)
            orig_dataset = task.load_dataset_from_file_name(file_path_original)
            orig_dataset = orig_dataset.select(range(int(k*coef_for_orig_data)))

    else:
        orig_dataset = task.load_dataset_from_file_name(file_path_original)
        orig_dataset = orig_dataset.select([])#range(int(k*0.25)))    

    review_dataset = concatenate_datasets([review_dataset, orig_dataset])

    print(review_dataset)
    return review_dataset

  

def upload_model(model_path):
    return AutoModelForSequenceClassification.from_pretrained(model_path)



def find_most_related_ones(task, topic_dataset, file_path, k):
    print('in finding most related ones')
    data_files = {"train": file_path}
    review_dataset = load_dataset("json", data_files=data_files, split="train")
    review_dataset = review_dataset.cast_column("stars", Value(dtype='int32', id=None))

    dataset1 = review_dataset.to_pandas()
    dataset2 = topic_dataset.to_pandas()

    all_datset1 = '\n'.join(dataset1[task.input()])
    print(all_datset1)
    all_dataset1_embeddings = get_embedding(all_datset1, engine='text-search-babbage-doc-001')

    dataset2['embeddings'] = dataset2[task.input()].apply(lambda x: get_embedding(x, engine='text-search-babbage-query-001'))
    dataset2['similarity'] = dataset2['embeddings'].apply(lambda x: cosine_similarity(x, all_dataset1_embeddings))
    print('comuting embedding and similarity is done!!!') 
    #top k indices that have the biggest similarity in dataset2
    top_k_indices = dataset2['similarity'].sort_values(ascending=False).index[:k]
    #print similarity and the review title of the top k indices
    for i in top_k_indices: 
        print(dataset2['similarity'][i], dataset2[task.input()][i])

    topic_dataset = topic_dataset.select(top_k_indices)
    print(topic_dataset['product_category'])
    return topic_dataset



#1) choose unlabeled data
#2) choose ones that are close to the embeddings
#3) choose ones that maximize disagreements
def choose_data(task, topic, file_path, global_model, 
            local_model, global_tokenizer, local_tokenizer, k):
    print('in data_selection', local_model)
    fasle_agreements = 1

    if(using_test_tree):
        test_tree = adatest.TestTree(f'{file_path}.test_tree', auto_save=True)
        task.add_data_to_test_tree(test_tree, global_model, local_model, 
        global_tokenizer, local_tokenizer, topic, k) #run 10 times, add max k data to test_tree
        return None, None, None


    #this is ddbug strategy for choosing data points
    if(local_model is not None):
        #select using a generator with embeddings
        # topic_dataset = task.load_data(topics='all', num_samples=k*50, label_filter=label_filter)
        # topic_dataset = find_most_related_ones(task, topic_dataset, file_path, k*10)
        #normal selection
        topic_dataset = task.load_data(topics=topic, num_samples=k*20,  label_filter=label_filter)
        tokenized_val = topic_dataset.map(task.tokenize_function, batched=True)



        local_trainer = Trainer(local_model, tokenizer = local_tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=4))
        global_trainer= Trainer(global_model, tokenizer = global_tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=4))
        print('prediction is done')
        local_pred =  local_trainer.predict(tokenized_val).predictions
        global_pred = global_trainer.predict(tokenized_val).predictions

        # print('local_pred', local_pred)
        # print('global_pred', global_pred)

        y_local_p = softmax(local_pred, axis=1)
        y_global_p = softmax(global_pred, axis=1)


        y_local = np.argmax(y_local_p,axis=1)
        y_global = np.argmax(y_global_p,axis=1)
        y_equal =  np.equal(y_local, y_global)*2 -1 # -1 if local is different from global, +1 if local is the same as global
        weights = np.abs(np.max(y_local_p,axis=1) - y_equal * np.max(y_global_p,axis=1))


        if(sampling_strategy == 'sampling'):
            diff_indices = np.random.choice(len(weights), k, p=weights/np.sum(weights), replace=False)
        elif(sampling_strategy == 'const'):
            diff_indices=np.argsort(weights.flatten())[-k:]
            diff_indices = diff_indices[weights[diff_indices] > 1] #there is disagreement! 

        disagreement_rate = (weights > 1).mean()
        fasle_agreements = ((y_local == y_global) & (y_local != topic_dataset['label']) ).mean()
        # topic_dataset_non_diff = topic_dataset.select(non_diff_indices).to_pandas()
        # topic_dataset_non_diff['stars'] = np.round(global_pred[non_diff_indices]).astype(np.int32)
        # topic_dataset_non_diff = Dataset.from_pandas(topic_dataset_non_diff)
        # topic_dataset = concatenate_datasets([topic_dataset_diff, topic_dataset_non_diff])

        print('sentences with highest disagreement! = ')#, val_sentences[diff_indices], val_labels[diff_indices])
        for x in diff_indices:
            print(topic_dataset[int(x)], topic_dataset['label'][x], y_local[x], y_global[x],  weights[x], y_local_p[x], y_global_p[x])

        topic_dataset = topic_dataset.select(diff_indices)

    elif(global_model is not None): #this is active learning strategy for choosing data points
        #normal selection
        topic_dataset = task.load_data(topics=topic, num_samples=k*20,  label_filter=label_filter)
        tokenized_val = topic_dataset.map(task.tokenize_function, batched=True)


        global_trainer= Trainer(global_model, tokenizer = global_tokenizer, args=TrainingArguments("test", per_device_eval_batch_size=4))
        print('prediction is done')
        global_pred = global_trainer.predict(tokenized_val).predictions
        y_global_p = softmax(global_pred, axis=1)
        y_global = np.argmax(y_global_p,axis=1)
        
        weights = 1-np.max(y_global_p,axis=1) #these are the least certain data points


        if(sampling_strategy == 'sampling'):
            diff_indices = np.random.choice(len(weights), k, p=weights/np.sum(weights), replace=False)
        elif(sampling_strategy == 'const'):
            diff_indices=np.argsort(weights.flatten())[-k:]


        print('sentences with most uncertainty! = ')
        for x in diff_indices:
            print(topic_dataset[int(x)], topic_dataset['label'][x], y_global[x],  weights[x],  y_global_p[x])

        topic_dataset = topic_dataset.select(diff_indices) 
        disagreement_rate = 1
      
    else: #randomly choose a subset of the topic_dataset
        topic_dataset = task.load_data(topics=topic, num_samples=k,  label_filter=label_filter)
        disagreement_rate = 1





    #add a column topic to the topic_dataset 
    tmp_file_name = file_path+'_'+str(suffix)+'.tmp'
    topic_dataset = topic_dataset.map(lambda x: {'topic': topic})
    topic_dataset.to_json(tmp_file_name)

    #open file_path as f1 and tmp_file as f2
    f1 = open(file_path, 'a')
    f2 = open(tmp_file_name, 'r')
    f1.write(f2.read())
    f1.close()
    f2.close()

    return topic_dataset, disagreement_rate, fasle_agreements



def run(task, topics, round, strategy,  num_iteration, num_data_selection, 
        result_file, file_path_original, file_path_ddbug, file_path_random, file_path_active_learning):
    #import any model (right now we have flg, qqp, and sentiment)
    # task = review.Review()
    #address to a .csv file to run your session

    torch.cuda.empty_cache()

    if(strategy == 'ddbug'):
        #clean the file path to save the selected data######
        # _,original_model = task.train_model_from_scratch(None, original_model_path, just_upload=True)
        # result_file = open(result_file_path, 'a')
        # task.validation_accuracy(None, 'ddbug_local_model', original_model, tokenizer, round, 0,  result_file, label_filter=[topic, task.original_topic()] )
        # result_file.close()

        #if we are using test tree we first add 1000 data points to the file_path test tree
        if(using_test_tree):
            pass
            #entailment experiments
            # if os.path.isfile(f'{file_path_ddbug}.test_tree'):
            #     os.remove(f'{file_path_ddbug}.test_tree')
            # for topic in topics:
            #     test_tree = adatest.TestTree(f'{file_path_ddbug}.test_tree', auto_save=True)
            #     dataset = task.load_data(topics=topic, split='train', num_samples=300) #num_dataselection
            #     inputs = [dataset[i]['premise']+' | '+dataset[i]['hypothesis'] for i in range(len(dataset))]
            #     mapz = {0: 'not entailment', 1: 'entailment'}
            #     adding_new_sentences_to_test_tree(test_tree, inputs, [mapz[x] for x in dataset['label']], topic)
            #     test_tree.to_csv(test_tree._tests_location)

            #biased expperiment
            # test_tree = adatest.TestTree(f'{file_path_ddbug}.test_tree', auto_save=True)
            # ds_biased, ds_biased_all, ds_unbiased_all, ds_test_all, ds1, ds2, ds3, ds4 = generate_datasets(num=10)
            # ds_biased = ds_biased.map(lambda x: {'label': 'positive' if x['label'] == 1 else 'negative'})
            # adding_new_sentences_to_test_tree(test_tree, ds_biased['review_body'], ds_biased['label'], 'biased')
        #write a random subset of local concepts into the file_path (including the original topic) this happens when ddbug_file_path is empty
        else:
            open(file_path_ddbug, 'w').close()
            for topic in topics:
                topic_dataset, _, _ = choose_data(task, topic, file_path_ddbug,  global_model = None, local_model=None, 
                                                    global_tokenizer=None, local_tokenizer=None, k=num_data_selection)
                result_file.write(f'ddbug_global_model,{round},{0},num_disagreement,{topic},{len(topic_dataset)}\n')
                result_file.flush()

        for iteration in range(0,num_iteration,1):
            torch.cuda.empty_cache()
            result_file = open(result_file_path, 'a')

            #fine-tune the global model
            dataset_global = generate_dataset(task, file_path_original, file_path_ddbug, topics)
            print(dataset_global)
            tokenizer_global, ddbug_global_model = task.fine_tune_model(dataset_global, original_model_path, 
                                                      ddbug_global_model_path, just_upload=False, batch_size=4)
            task.validation_accuracy(topics, 'ddbug_global_model', ddbug_global_model, tokenizer_global, round, iteration+1, 
                                    result_file, label_filter=label_filter)
            ######## uncomment for the biased experiment #########
            # dataset_lists = [ds_test_all, ds1,ds2,ds3,ds4]
            # for i in range(len(dataset_lists)):
            #     task.compute_accuracy(dataset_lists[i], 'ddbug_global_model', ddbug_global_model, tokenizer_global,f'ds_{i}', round, iteration+1, result_file)
            
            #adding data for each topic and update their local model
            for topic in topics:
                dataset_local = generate_dataset(task, file_path_original, file_path_ddbug, [topic])
                print('dataset_local', dataset_local)
                if(topics_path[topic] is not None):
                    tokenizer_local, ddbug_local_model  = task.fine_tune_model(dataset_local, original_model_path, 
                                                              topics_path[topic], just_upload=True)
                else:
                    tokenizer_local, ddbug_local_model  = task.fine_tune_model(dataset_local, original_model_path, ddbug_local_model_path, 
                                                              just_upload=False, batch_size=4)
                    task.validation_accuracy(topics, f'ddbug_local_model_{topic}', ddbug_local_model, tokenizer_local, round, 
                                             iteration+1,  result_file, label_filter=label_filter)
                    # print('big error')
                    # exit(1)
                topic_dataset, disagreement_rate, false_agreement = choose_data(task,topic,file_path_ddbug, 
                                    global_model = ddbug_global_model, local_model=ddbug_local_model, 
                                    global_tokenizer=tokenizer_global, local_tokenizer=tokenizer_local,
                                    k=num_data_selection)
                result_file.write(f'ddbug_global_model,{round},{iteration+1},num_disagreement,{topic},{len(topic_dataset)}\n')
                result_file.write(f'ddbug_global_model,{round},{iteration+1},disagreement_rate,{topic},{disagreement_rate}\n')
                result_file.write(f'ddbug_global_model,{round},{iteration+1},false_agreement,{topic},{false_agreement}\n')
            torch.cuda.empty_cache()

            result_file.close()

    elif(strategy == 'random'):
        ##random procedure:
        # _,original_model = task.train_model_from_scratch(None, original_model_path, just_upload=True)
        # task.validation_accuracy([topic, task.original_topic()], 'random_local_model', original_model, tokenizer, round, 0,  result_file, label_filter=label_filter)
        # choose_data_ddbug(task,topic,file_path_random, global_model = None, local_model=None, tokenizer=None, k=num_data_selection)
        
        open(file_path_random, 'w').close()
        #copy the first k rows of file_path_ddbug into file_path_random
        f1 = open(file_path_ddbug, 'r')
        f2 = open(file_path_random, 'a')
        for i in range(num_data_selection*len(topics)):
            f2.write(f1.readline())
        f1.close()
        f2.close()

        for iteration in range(num_iteration):
            result_file = open(result_file_path, 'a')

            #random local model
            # dataset_local = generate_dataset(task, file_path_original, file_path_random, [topic])
            # random_local_model  =  task.fine_tune_model(dataset_local, original_model_path, random_local_model_path, just_upload=False)

            #random global model
            dataset_global = generate_dataset(task, file_path_original, file_path_random, topics)
            global_tokenizer, random_global_model =  task.fine_tune_model(dataset_global,  original_model_path, random_global_model_path, just_upload=False)
            task.validation_accuracy(topics, 'random_global_model', random_global_model, global_tokenizer, round, iteration+1,  result_file, label_filter=label_filter)

            for topic in topics:
                choose_data(task,topic,file_path_random, global_model = None, local_model=None, 
                global_tokenizer=None, local_tokenizer=None, k=num_data_selection)
            result_file.close()

    elif(strategy == 'active_learning'):
        open(file_path_active_learning, 'w').close()
        #copy the first k rows of file_path_ddbug into file_path_active_learning
        f1 = open(file_path_ddbug, 'r')
        f2 = open(file_path_active_learning, 'a')
        for i in range(num_data_selection*len(topics)):
            f2.write(f1.readline())
        f1.close()
        f2.close()

        for iteration in range(num_iteration):
            result_file = open(result_file_path, 'a')

            #active_learning global model
            dataset_global = generate_dataset(task, file_path_original, file_path_active_learning, topics)
            global_tokenizer, active_learning_global_model =  task.fine_tune_model(dataset_global,  original_model_path, active_learning_global_model_path, just_upload=False)
            task.validation_accuracy(topics, 'active_learning_global_model', active_learning_global_model, global_tokenizer, round, iteration+1,  result_file, label_filter=label_filter)

            for topic in topics:
                choose_data(task,topic,file_path_active_learning, global_model = active_learning_global_model, 
                                  local_model=None, global_tokenizer=global_tokenizer, local_tokenizer=None, k=num_data_selection)
            result_file.close()





def run_all():


    ##### uncomment this if you never saved the orignal model ########
    # original_dataset = task.load_data(topics = task.original_topic(), num_samples=200000, label_filter=label_filter)
    # print('original dataset', original_dataset)
    # validation_dataset = task.load_data(topics = task.original_topic(), split='validation', label_filter=label_filter)
    # tokenizer, model = task.train_model_from_scratch(raw_dataset = original_dataset, 
    #                                 model_path = original_model_path, validation_dataset = validation_dataset,
    #                                 just_upload=False, num_epochs=15)
    # task.validation_accuracy([topic, task.original_topic()], 'orignal_model', model, tokenizer, None, None, None, label_filter=label_filter )

    result_file = open(result_file_path, 'w')
    result_file.write('name,round,iteration,metric,topic,value\n')
    # tokenizer_global, ddbug_global_model = task.fine_tune_model(None, original_model_path, 
    #                                         original_model_path, just_upload=True, batch_size=4)
    # task.validation_accuracy(topics, 'ddbug_global_model', ddbug_global_model, tokenizer_global, 0, 0, 
    #                         result_file, label_filter=label_filter)

    for round in range(0,num_rounds,1):
        # original_dataset = task.load_data(topics = task.original_topic(), num_samples=1000, label_filter=label_filter)
        # original_dataset.select(range(num_iteration*3*num_data_selection)).to_json(file_path_original)

        result_file = open(result_file_path, 'a')

        file_path_ddbug  = f'{directory_path}/data/reviews_ddbug_{task.original_topic()}_{topics}_{round}_{suffix}.json'
        file_path_random = f'{directory_path}/data/reviews_random_{task.original_topic()}_{topics}_{round}_{suffix}.json'
        file_path_active_learning = f'{directory_path}/data/reviews_active_learning_{task.original_topic()}_{topics}_{round}_{suffix}.json'

        # shutil.copyfile(f"{directory_path}/data/reviews_ddbug_{task.original_topic()}_{['MED-downward_monotone', 'normal']}_0_None.json",
        #                 file_path_random)
        for strategy in ['ddbug', 'random', 'active_learning']:
            print('---------------------------', round, 'with', strategy, '---------------------------')

            run(task, topics, round, strategy, num_iteration, num_data_selection, result_file, 
                file_path_original, file_path_ddbug, file_path_random, file_path_active_learning)
        result_file.close()


directory_path = '..'
strategy = args.strategy

if(args.dataset == 'nli'):
    task = nli.NLI()
    topics=['MED-upward_monotone',  'normal']#, task.original_topic()]#,
    topics_path={'MED-upward_monotone':None,#f'{directory_path}/fine_tuned_models/local_MED-downward_monotone_500_roberta_large',
                'normal':f'{directory_path}/fine_tuned_models/roberta-large-multinli-binary',
                }
    # new_topics = ['MED-upward_monotone']
    sampling_strategy = 'const'
    label_filter = False
    original_model_path = f'{directory_path}/fine_tuned_models/roberta-large-multinli-binary'
    warm_up = False
    using_test_tree = False
    suffix = 'simple_check_with_numbers'

if(args.dataset == 'review'):
    task = review.Review()
    # topics=['home_improvement', 'pet_products', 'wireless', 'home', task.original_topic()]
    topics_path={'MED-downward_monotone':None,#f'{directory_path}/fine_tuned_models/local_MED-downward_monotone_500_roberta_large',
            'normal':f'{directory_path}/fine_tuned_models/roberta-large-multinli-binary',
            }
    sampling_strategy = 'const'
    label_filter = False
    original_model_path = f'{directory_path}/fine_tuned_models/amazon_review_{task.original_topic()}_roberta_large'
    warm_up = False
    using_test_tree = True
    suffix = 'amazon_biased'

if(args.dataset == 'sentiment'):
    task = sentiment.Sentiment()
    topics=['biased']#'normal', 
    topics_path={'normal': "../fine_tuned_models/amazon_review_0_1_subtract_roberta_large",
            'biased':f'{directory_path}/fine_tuned_models/amazon_review_0_1_roberta_large',
            }
    sampling_strategy = 'const'
    label_filter = False
    original_model_path = "../fine_tuned_models/amazon_review_0_1_subtract_roberta_large"
    warm_up = False
    using_test_tree = True
    suffix = 'install_myskin_batterylife_only_in_context'

print('running the code with the suffix: ', suffix, 'and strategy: ', strategy, 'on gpu: ', args.gpu, 'and dataset: ', args.dataset)



file_path_original = f'{directory_path}/data/reviews_original_{task.original_topic()}_{topics}_{suffix}.json'

base_model_path = f'{directory_path}/fine_tuned_models/classification_base_{task.original_topic()}_roberta_large'
ddbug_global_model_path   = f'{directory_path}/fine_tuned_models/global_{task.name()}_{suffix}'
ddbug_local_model_path    = f'{directory_path}/fine_tuned_models/local_{task.name()}_{topics}_{suffix}'

random_local_model_path   =  f'{directory_path}/fine_tuned_models/random_local_{task.name()}_{topics}_{suffix}'
random_global_model_path  =  f'{directory_path}/fine_tuned_models/random_global_{task.name()}_{topics}_{suffix}'

active_learning_local_model_path   =  f'{directory_path}/fine_tuned_models/active_learning_local_{task.name()}_{topics}_{suffix}'
active_learning_global_model_path  =  f'{directory_path}/fine_tuned_models/active_learning_global_{task.name()}_{topics}_{suffix}'

result_file_path = f'{directory_path}/results/results_{task.original_topic()}_{topics}_{suffix}.csv'

num_rounds=10
num_iteration = 8
num_data_selection = 50

run_all()

