session_path = '../examples/'#adatest_qqp_orig/'
# session = 'sst_amazon_wish_it_only_thing.csv'#'review_islam_terrorist.csv'
session = 'toxicity_niloufar.csv'#qqp-Modifier:_adj-r2_orig'
#topic that you want to focus on! if you put None then normal adatest would run
topic= None#''#None#''#None#''#None#''#'/'#'cluster5'#'terrorist_and_islam'#'cluster5'#dm
csv_file_to_read = None#'../examples/task_embedding_no_superclass/cluster5.csv'

#the original model path + the path you want for local and global model! 
# original_model_path = f'../fine_tuned_models/original_{task.name()}'
# original_model_path = "distilbert-base-uncased-finetuned-sst-2-english"
# original_model_path = "unitary/toxic-bert"
# original_model_path = f'../fine_tuned_models/amazon_review_{task.original_topic()}_roberta_large'
original_model_path = 'mohsenfayyaz/toxicity-classifier'#'/datadrive/research/ddbug/fine_tuned_models/roberta-large-qqp'#f'../fine_tuned_models/roberta-large-multinli-binary'
global_model_path   = f'../fine_tuned_models/global_{session}'
local_model_path    = f'../fine_tuned_models/local_{session}_{topic}'
num_epochs = 30

