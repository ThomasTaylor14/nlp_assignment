from typing import List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
import torch
import transformers as ppb
import pickle

# # Loading pre-trained model
# # Here we are using DistilBERT Model
# model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

# # We can also use BERT instead of DistilBERT then need to uncomment the below line
# # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# # Load pretrained model/tokenizer
# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights) #.to('cuda')

class Classifier:
    """
    Trains the classifier model on the training set stored in file trainfile
    PLEASE:
        - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        - PUT THE MODEL and DATA on the specified device! Do not use another device
        - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
        OF MODEL HYPERPARAMETERS
    """

    def __init__(self) -> None:
        # Loading BERT pre-trained model
        self.model_class, self.tokenizer_class, self.pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'bert-base-uncased')

        # Load pretrained model/tokenizer
        self.model = self.model_class.from_pretrained(self.pretrained_weights)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)


    #############################################
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Reading the data and loading into a dataframe
        df = pd.read_csv(train_filename, delimiter='\t', header=None, names=["polarity", "aspect_category", "aspect_term", "position", "sentence"])

        # Tokenize the sentences - break them up into word and subwords 
        tokenized = df['sentence'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))

        # Finding the length of the largest sentence
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        # To represent the input as one 2-d array pad all lists to the same size
        padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])

        # Masking helps in ignoring the padding while processing the input
        attention_mask = np.where(padded != 0, 1, 0)

        # Converting 'pad' and 'mask' into tensor
        input_ids = torch.tensor(padded).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        # Running our data (sentences) through BERT model
        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        # Saving the 'features' for logistics regression model
        features = last_hidden_states[0][:,0,:].cpu().numpy()

        # Saving the 'polarity' of the sentence to the 'labels'
        labels = df['polarity']

        # Training the model
        clf = LogisticRegression(C=0.1, max_iter=2000) #Acc.: 82.45

        clf.fit(features, labels)

        # Saving weights of the model
        filename = 'final_model.h5'
        pickle.dump(clf, open(filename, 'wb'))
        print('Training is finished')


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Reading the data and loading into a dataframe
        df = pd.read_csv(data_filename, delimiter='\t', header=None, names=["polarity", "aspect_category", "aspect_term", "position", "sentence"])

        # Tokenization
        # Tokenize the sentences - break them up into word and subwords 
        tokenized = df['sentence'].apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))

        # Finding the length of the largest sentence
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        # To represent the input as one 2-d array, pad all lists to the same size
        padded = np.array([i + [0]*(max_len - len(i)) for i in tokenized.values])

        # Masking helps in ignoring the padding while processing the input
        attention_mask = np.where(padded != 0, 1, 0)

        # Converting 'pad' and 'mask' into tensors
        input_ids = torch.tensor(padded).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        # Running the data (sentences) through BERT model
        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        # Saving the 'features' for logistic regression model
        features = last_hidden_states[0][:,0,:].cpu().numpy()

        # Saving the 'polarity' of the sentence to the 'labels'
        labels = df['polarity']

        # Loading and predicting with the model
        filename = 'model.h5'
        loaded_model = pickle.load(open(filename, 'rb'))
        predictions = loaded_model.predict(features)

        return predictions
