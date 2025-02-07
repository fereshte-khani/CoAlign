import os
import adatest
from coalign.models import toxicity
from coalign.utils.test_tree_functions import generate_fine_tuned_models_of_a_file, launching_browser
from coalign.src import coalign_code_config


adatest.backend = adatest.backends.OpenAI('davinci', api_key=os.getenv('OPENAI_API_KEY'))

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


    return global_scorer, global_predictor, local_scorer, local_predictor


if __name__ == "__main__":
    #import any model (right now we have nli, qqp, and sentiment)
    task = toxicity.Toxicity()

    test_tree = adatest.TestTree(coalign_code_config.session_path + coalign_code_config.session, auto_save=True)


    global_scorer, global_model, local_scorer, local_model = update_local_global(task, coalign_code_config.session, test_tree, coalign_code_config.topic, 
                                                    update_local=False, update_global=False, compute_metrics=False)

    # test_tree = adatest.TestTree(new_path, auto_save=True)
    launching_browser(task = task,
                      test_tree=test_tree, 
                      scorer=global_scorer, 
                      predictor=local_model, 
                      dport=5001)

 