from datasets import load_dataset, Dataset
from pandas import DataFrame
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from transformers import AutoTokenizer, DataCollatorWithPadding, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class Syn:
    def name(self):
        return "syn"

    def original_topic(self):
        return 'one'

    def load_data(self, topic=None):
        n1=1000 
        n2=1000
        if(topic is None):
            var = 0.1
            x_g1 = np.concatenate ((np.random.multivariate_normal([2,0], np.eye(2)*var, n1), 
                                    np.random.multivariate_normal([-2,0], np.eye(2)*var, n1)), 0)
            y_g1 = np.concatenate((np.zeros(n1), np.ones(n1)))


            cov=0.6
            x_g2 = np.concatenate ((np.random.multivariate_normal([0.6,1.2], np.array([[1,cov], [cov,1]]) * var, n2), 
                                    np.random.multivariate_normal([-0.6,2.3], np.array([[1,cov], [cov,1]]) * var, n2)), 0)
            y_g2 = np.concatenate((np.zeros(n2), np.ones(n2)))


            x = np.concatenate((x_g1, x_g2), 0)
            y = np.concatenate((y_g1, y_g2))
        else:
            var = 0.2
            x_g1 = np.concatenate ((np.random.multivariate_normal([0.8,6], np.array([[1,0],[0,var]])*var, n1), 
                                    np.random.multivariate_normal([-0.8,4], np.array([[1,0],[0,var]])*var, n1)), 0)
            y_g1 = (x_g1.T[0] > 0)*1
            print(y_g1)

            x_g2 = (np.random.multivariate_normal([0,0], np.eye(2) * var, n2))
            y_g2 = (x_g2.T[0] > 0)*1

            x = np.concatenate((x_g1, x_g2), 0)
            y = np.concatenate((y_g1, y_g2))


        df = DataFrame(dict(x1=x[:,0], x2=x[:,1], label=y))
        dataset = Dataset.from_pandas(df)
        print(df)
        print(dataset)

        plt.scatter(*x[y==0].T, s=8, alpha=0.5, color='green')
        plt.scatter(*x[y==1].T, s=8, alpha=0.5, color='red')
        plt.savefig('feri1')

        return df


    def input(self):
        return 'review_title'

    def fine_tune_model(self, review_dataset, original_model_path, output_path, just_upload=False):

        if(just_upload):
            return AutoModelForSequenceClassification.from_pretrained(output_path)

            
        #load the model from original_model_path
        model = AutoModelForSequenceClassification.from_pretrained(original_model_path)

        input= self.input()
        #some preprocessing for sentiment task
        train_labels = np.array(review_dataset['stars'], dtype=np.float32)
        train_sentences = list(review_dataset[input])
        train = Dataset.from_dict({'sentence' : train_sentences, 'label': train_labels})


        tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        def tokenize_function(examples):
            return tokenizer(examples["sentence"], truncation=True, max_length=50)
        tokenized_train = train.map(tokenize_function, batched=True)
        tokenized_datasets = {'train': tokenized_train}

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = RobertaForSequenceClassification.from_pretrained(original_model_path, num_labels=1, problem_type="regression")
        
        training_args = TrainingArguments("test_trainer", learning_rate=5e-6, logging_steps=50,
                                        save_strategy='no', eval_steps=500, #evaluation_strategy="epoch",
                                        label_names=['label'], 
                                        per_device_train_batch_size=4,
                                        num_train_epochs=15,
                                        weight_decay=0.01,
                                        )
                        
        trainer = Trainer(
            model=model, args=training_args,
            train_dataset=tokenized_datasets['train'],
            tokenizer=tokenizer,
            data_collator=data_collator, 
        )
        trainer.train()
        trainer.save_model(output_path)
        return model


    # def tokenize_function(self, examples,):
    #     return tokenizer(examples["sentence"], truncation=True)


    def compute_accuracy(self, dataset, model, tokenizer, round, iteration, name, output_file, tc):
            val_labels = np.array(dataset['stars'], dtype=np.float32)
            val_sentences = list(dataset[self.input()])
            val = Dataset.from_dict({'sentence' : val_sentences, 'label': val_labels})
            def tokenize_function(examples):
                return tokenizer(examples["sentence"], truncation=True, max_length=50)
            tokenized_val = val.map(tokenize_function, batched=True)
            tokenized_datasets = {'validation': tokenized_val}



            trainer = Trainer(model = model, tokenizer = tokenizer)
            pred =  trainer.predict(tokenized_datasets['validation'])
            predictions, labels = pred.predictions, pred.label_ids

            results = mean_squared_error(predictions, labels, squared=False)
            print('mean_squared_error', results)
            output_file.write(f'{name},{round},{iteration},mean_squared_error,{tc},{results}\n')
            results = r2_score(predictions, labels)
            print('r2', results)
            output_file.write(f'{name},{round},{iteration},r2,{tc},{results}\n')



    def validation_accuracy(self, topic, model, tokenizer, round, iteration, name, output_file):
        raw_dataset = load_dataset('amazon_reviews_multi', 'en', cache_dir='/datadrive/research/ddbug/synthetic/amazon_multi/cache', split='validation')
        
        #fiter raw_dataset to only contains baby_product
        for tc in [topic, self.original_topic()]:
            raw_dataset_filter = raw_dataset.filter(lambda x: x['product_category'] == tc)
            self.compute_accuracy(raw_dataset_filter, model, tokenizer, round, iteration, name, output_file, tc)



syn = Syn()
syn.load_data()
