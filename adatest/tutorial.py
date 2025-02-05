import adatest
import openai
import os
import sentence_transformers
import logging
from transformers import pipeline

# Load your openai api key to run GPT-3, set that as a backend
# with open(os.path.expanduser('~/research/adatest/adatest/.openai_api_key'), 'r') as file:
#     adatest.backend = adatest.backends.OpenAI('davinci-msft', api_key=file.read().replace('\n', ''))

with open(os.path.expanduser('~/research/adatest/adatest/.openai_api_key'), 'r') as file:
    adatest.backend = adatest.backends.OpenAI('davinci', api_key="sk-BdDMgejmZKhxTM5KCtP8JsVXTd1uu9JitK7Q1kX3")

logging.basicConfig(
    filename='/tmp/adatest.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S'
)

roberta_embed_model = sentence_transformers.SentenceTransformer('stsb-roberta-large')


# from adatest.utils import cogservices
# with open(os.path.expanduser('~/.sentiment_key'), 'r') as file:
#     key = file.read().replace('\n', '')
# model = cogservices.SentimentModel('./newazure.pkl', key, wait_time=0)


# sen_model = pipeline('sentiment-analysis')
# def model(x):
#     returns = []
#     results=sen_model(x)
#     print(results)
#     for rs in results:
#         if(rs['label']=='POSITIVE'):
#             returns.append([1-rs['score'], 0, rs['score']])
#         else:
#             returns.append([rs['score'], 0, 1-rs['score']])
#     return returns





import nltk
import numpy as np
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def model(ls):
    returns = []
    for x in ls:
        result=sid.polarity_scores(x)
        returns.append([result['neg'], result['neu'], result['pos']])
    return np.array(returns)



    
print(model(['I hate this', 'I love this']))




labels = ['negative', 'neutral', 'positive']
wrapped_fn = lambda x: model(x)
wrapped_fn.output_names = labels
scorer = adatest.ClassifierScorer(wrapped_fn)



test_tree = adatest.TestTree('/tmp/new_session.csv', auto_save=True)
test_tree



browser = test_tree(scorer=scorer, embedding_model=roberta_embed_model, max_suggestions=150,
                    recompute_scores=True)


adatest.serve(browser, host='0.0.0.0', port=5000)
