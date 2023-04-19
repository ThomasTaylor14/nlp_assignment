from typing import List

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.svm import SVC 
from transformers import DistilBertModel, DistilBertTokenizer
import pickle

import warnings
warnings.filterwarnings("ignore")




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
        self.model_class, self.tokenizer_class, self.pretrained_weights = (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased')
        
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
        #clf = LogisticRegression(C=0.1, max_iter=2000) #Acc.: 82.45
        #LR {'C': 0.1, 'max_iter': 100, 'penalty': 'l2', 'solver': 'newton-cg'} # 83
        # SVC {'C': 100, 'gamma': 'auto', 'kernel': 'rbf', 'max_iter': 2000}

        # models = {'Logistic Regression': LogisticRegression(random_state = 42), 'Random Forest': RandomForestClassifier(random_state=4 ), 'Gradient Boosting': GradientBoostingClassifier(), 'SVM': SVC()}

        # use cross validation to find the best parameters

        # seperate in train and val set the train set

        # use cross validation to find the best parameters


        # for name, model in models.items():
        #     print(name)
        #     print(model)
        #     cv_score = cross_val_score(model, X_train, y_train, cv=5)
        #     print('Cross Validation Score: ', cv_score)
        #     print('Mean Cross Validation Score: ', np.mean(cv_score))
        #     print('Standard Deviation of Cross Validation Score: ', np.std(cv_score))
        #     print('--------------------------------------')

        #     model.fit(X_train, y_train)

        #     y_pred = model.predict(X_val)

        #     print('Accuracy Score: ', accuracy_score(y_val, y_pred))
        #     print('F1 Score: ', f1_score(y_val, y_pred, average='weighted'))
        #     print('Precision Score: ', precision_score(y_val, y_pred, average='weighted'))
        #     print('Recall Score: ', recall_score(y_val, y_pred, average='weighted'))
        #     print('--------------------------------------')

        #     # Results

        #     Logistic Regression
        #     LogisticRegression(random_state=42)
        #     Cross Validation Score:  [0.80497925 0.82572614 0.80833333 0.83333333 0.85416667]
        #     Mean Cross Validation Score:  0.8253077455048409
        #     Standard Deviation of Cross Validation Score:  0.017881838367496353
        #     --------------------------------------
        #     Accuracy Score:  0.8272425249169435
        #     F1 Score:  0.8172293332593132
        #     Precision Score:  0.8078933236244482
        #     Recall Score:  0.8272425249169435
        #     --------------------------------------
        #     Random Forest
        #     RandomForestClassifier(random_state=4)
        #     Cross Validation Score:  [0.83817427 0.78008299 0.7875     0.7875     0.7875    ]
        #     Mean Cross Validation Score:  0.7961514522821578
        #     Standard Deviation of Cross Validation Score:  0.02120686667960377
        #     --------------------------------------
        #     Accuracy Score:  0.7973421926910299
        #     F1 Score:  0.7740023918157873
        #     Precision Score:  0.763704134776171
        #     Recall Score:  0.7973421926910299
        #     --------------------------------------
        #     Gradient Boosting
        #     GradientBoostingClassifier()
        #     Cross Validation Score:  [0.8340249  0.82572614 0.79583333 0.8        0.81666667]
        #     Mean Cross Validation Score:  0.8144502074688796
        #     Standard Deviation of Cross Validation Score:  0.014633005912053588
        #     --------------------------------------
        #     Accuracy Score:  0.813953488372093
        #     F1 Score:  0.7996893791504212
        #     Precision Score:  0.7876324166949619
        #     Recall Score:  0.813953488372093
        #     --------------------------------------
        #     SVM
        #     SVC()
        #     Cross Validation Score:  [0.82157676 0.83817427 0.8125     0.80833333 0.79166667]
        #     Mean Cross Validation Score:  0.8144502074688796
        #     Standard Deviation of Cross Validation Score:  0.015322717230208845
        #     --------------------------------------
        #     Accuracy Score:  0.8272425249169435
        #     F1 Score:  0.8013343152909604
        #     Precision Score:  0.8050403174414356
        #     Recall Score:  0.8272425249169435

        # Saving weights of the model
        
        clf = LogisticRegression(random_state=42, C= 0.1, max_iter = 100, penalty = 'l2', solver= 'newton-cg', n_jobs=-1)
        # clf =  SVC(random_state=42, C= 100, gamma= 'auto', kernel='rbf', max_iter= 2000) # 80.59
        clf.fit(features, labels)
        filename = 'LR_tuned_v2.h5'
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
        filename = 'LR_tuned_v2.h5'
        loaded_model = pickle.load(open(filename, 'rb'))
        predictions = loaded_model.predict(features)

        # print accuracy, recall and precision, f1-score 

        print('Accuracy: ', accuracy_score(labels, predictions))
        print('Recall: ', recall_score(labels, predictions, average='macro'))
        print('Precision: ', precision_score(labels, predictions, average='macro'))
        print('F1-score: ', f1_score(labels, predictions, average='macro'))

        return predictions