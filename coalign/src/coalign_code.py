import os
import adatest
from coalign.models import toxicity
from coalign.utils.test_tree_functions import generate_fine_tuned_models_of_a_file, launching_browser
from coalign.src import coalign_code_config


with open(os.path.expanduser('~/keys/.openai_api_key'), 'r') as file:
    adatest.backend = adatest.backends.OpenAI('davinci', api_key=file.read().replace('\n', ''))

import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(name)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def update_local_global(task, session, test_tree, topic, update_local, update_global, compute_metrics=False):
    for num_epochs in [30]:
        for number in [len(test_tree._tests)]:
            if(update_local):
                generate_fine_tuned_models_of_a_file(task, test_tree, topic, coalign_code_config.original_model_path, 
                                                        coalign_code_config.local_model_path, num_epochs=num_epochs)
            if(topic != None):
                tokenizer, local_model = task.upload_pretrained_model(coalign_code_config.local_model_path) 
                local_predictor, local_scorer = task.get_model_and_scorer(local_model, tokenizer)
            else:
                local_predictor = None #works just like adatest
                local_scorer = None


            if(update_global):
                generate_fine_tuned_models_of_a_file(task, test_tree, None, coalign_code_config.original_model_path, 
                                                        coalign_code_config.global_model_path, num_epochs=num_epochs)
                tokenizer, global_model = task.upload_pretrained_model(coalign_code_config.global_model_path) 
            else:
                #check if the directory global_model_path exists if not copy  original_model_path directory to global_model_path directory
                if(os.path.exists(coalign_code_config.global_model_path)):
                    tokenizer, global_model = task.upload_pretrained_model(coalign_code_config.global_model_path) 
                    # tokenizer, global_model = task.upload_pretrained_model(original_model_path) 

                    # shutil.copytree(original_model_path, global_model_path)
                else:
                    tokenizer, global_model = task.upload_pretrained_model(coalign_code_config.original_model_path)

            global_predictor, global_scorer = task.get_model_and_scorer(global_model, tokenizer)

                    # print(local_predictor([['fereshte is good!']]))
            # if(compute_metrics):
            #     calculate_some_metrics(task, local_model, global_model, number, num_epochs)
    return global_scorer, global_predictor, local_scorer, local_predictor


if __name__ == "__main__":

    #import any model (right now we have nli, qqp, and sentiment)
    # task = sentiment.Sentiment()
    task = toxicity.Toxicity()

    test_tree = adatest.TestTree(coalign_code_config.session_path + coalign_code_config.session, auto_save=True)

    # task.generate_initial_test_tree(test_tree)
    # print(test_tree)    

    ################## generate a tree from the reviews zexue paper ##################
    # if(len(test_tree._tests) == 0 and coalign_code_config.csv_file_to_read != None):
    #     df = pd.read_csv(coalign_code_config.csv_file_to_read)
    #     list1 = []
    #     for i in range(len(df)):
    #         list1.append(df['premise'][i] + ' | ' + df['hypothesis'][i])
    #     # list2 = [1 if x=='entailment' else 0 for x in df['label']]
    #     adding_new_sentences_to_test_tree(test_tree, list1, df['label'], 'cluster5')

    ################## generate a tree from biased reviews ############################
    # if(len(test_tree._tests) == 0 and coalign_code_config.csv_file_to_read != None):
    #     # ds_biased, ds_biased_all, ds_unbiased_all, ds_test_all, ds1, ds2, ds3, ds4 = generate_datasets(num=5)
    #     # ds_biased = ds_biased.map(lambda x: {'label': 'positive' if x['label'] == 1 else 'negative'})
    #     # adding_new_sentences_to_test_tree(test_tree, ds_biased['review_body'], ds_biased['label'], 'dm')

    global_scorer, global_model, local_scorer, local_model = update_local_global(task, coalign_code_config.session, test_tree, coalign_code_config.topic, 
                                                    update_local=False, update_global=False, compute_metrics=False)



    # test_tree = adatest.TestTree('tmp.csv', auto_save=True)
    # if(len(test_tree._tests) == 0):
    #     df = pd.read_csv('../examples/task_embedding_no_superclass/cluster5_aligned.csv')
    #     list1 = []
    #     for i in range(len(df)):
    #         list1.append(df['premise'][i] + ' | ' + df['hypothesis'][i])
    #     adding_new_sentences_to_test_tree(test_tree, list1, df['label'], 'cluster5')

    # new_path = '/datadrive/research/ddbug/examples/adatest_qqp/qqp-How_can_I_become_a_X_person_==_How_can_I_become_a_person_who_is_not_antonym(X)-r2'
    # new_path ="/datadrive/research/ddbug/examples/adatest_qqp/qqp-Modifier:_adj-r2_with_conf"
    # test_tree = adatest.TestTree(new_path, auto_save=True)
    launching_browser(task = task,
                      test_tree=test_tree, 
                      scorer=global_scorer, 
                      predictor=local_model, 
                      dport=5001)

 