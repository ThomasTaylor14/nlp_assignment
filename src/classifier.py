from typing import List

import numpy as np
import pandas as pd
import torch
import pickle
import transformers as tsf

from sklearn.linear_model import LogisticRegression


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please do not change
     """

    def __init__(self) -> None:
        # Loading BERT pre-trained model
        self.model_class, self.tokenizer_class, self.pretrained_weights = (tsf.DistilBertModel, tsf.DistilBertTokenizer, 'distilbert-base-uncased')

        # Load pretrained model/tokenizer
        self.model = self.model_class.from_pretrained(self.pretrained_weights)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        
        # Defines the name of the model file.
        self.file_name = 'model.h5'

    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        print("    Training is starting...")
        # Load the train csv.
        df_train = pd.read_csv(train_filename, delimiter='\t', header=None, names=["polarity", "aspect_category", "aspect_term", "position", "sentence"])
        
        # Creates the features and labels DataFrames.
        features = self.get_text_features(df_train['sentence'], device)
        labels = df_train['polarity']
        print("    -----> Data preprocessing done.")

        # Training the model.
        clf = LogisticRegression(C=0.1, max_iter=2000)
        clf.fit(features, labels)
        print("    -----> Model trained.")

        # Saves the model
        pickle.dump(clf, open(self.file_name, 'wb'))
        print("    -----> Model saved.")
        print("    Training done !")
        
    def get_text_features(self, text_series: pd.Series, device: torch.device):
        """
        Tokenizes and transformes the sentence so that they can be used to train models.
        """
        # Tokenize the sentences - break them up into word and subwords.
        tokenized = text_series.apply(lambda x: self.tokenizer.encode(x, add_special_tokens=True))

        # Finding the length of the largest sentence.
        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        # To represent the input as one 2-d array pad all lists to the same size.
        padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

        # Masking helps in ignoring the padding while processing the input.
        attention_mask = np.where(padded != 0, 1, 0)

        # Converting 'pad' and 'mask' into tensor
        input_ids = torch.tensor(padded).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        # Running our data (sentences) through BERT model.
        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        # Returns the features in the last hidden features.
        return last_hidden_states[0][:, 0, :].cpu().numpy()


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        # Load the test csv.
        df_test = pd.read_csv(data_filename, delimiter='\t', header=None)

        # Adds name to the column. If the csv contains 4 columns, polarity column is missing.
        names = ["polarity", "aspect_category", "aspect_term", "position", "sentence"]
        if len(df_test.columns) < 5:
            names = names[1:]
        df_test = df_test.set_axis(names, axis=1, copy=False)

        # Creates the features and labels DataFrames.
        features = self.get_text_features(df_test['sentence'], device)

        # Loading and predicting with the model
        loaded_model = pickle.load(open(self.file_name, 'rb'))
        return loaded_model.predict(features)
        





