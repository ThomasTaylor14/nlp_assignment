# Natural Language Processing - Final project

This is the final project for the Natural Language Processing course at CentraleSupélec, in the Data Science & Business Analytics Master.
The authours of the project are:
- [Pierre Chagnon]
- [Hugo Chikli]
- [Marina Pellet]
- [Thomas Taylor]

## Project description

The goal of this assignment is to implement a classifier that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences.

## Data

The data is composed of several features concerning the restaurants reviews. We have the following features:
- 'polarity': the polarity of the review (positive, negative or neutral). This is the label that we are trying to predict, our target variable.
- 'aspect category': the aspect categories of the review.
- 'target term' : the aspect term of the review.
- 'sentence' : the text review.
- 'character offset' : the position of the target term in the sentence. 

## Modeling

### Choice of tokenizer and text embedding model

In this implementation, we use the DistilBertTokenizer from the transformers library. The tokenizer is used to break down the sentences into tokens or words, which can be inputted to the DistilBert model. The DistilBertTokenizer is trained on the same corpus as the DistilBertModel and can be used to tokenize the data for training and inference.

We load the pretrained DistilBertModel and DistilBertTokenizer using the from_pretrained method of their respective classes. The tokenizer is used to tokenize the sentences by encoding them, adding special tokens such as '[CLS]' (beginning of sentence) and '[SEP]' (end of sentence), and breaking them down into word and subword tokens. Additionally, we add the token '[ASP]' (before the aspect term) and '[/ASP]' (after the aspect term) to wrap the target word and point to it.

We also use padding and masking to ensure that all sequences are of the same length before feeding them into the model. Padding is added to the end of sequences that are shorter than the longest sequence, while masking is used to ignore the padding during training.

After tokenization and preprocessing, the sentences are then run through the DistilBert model to generate embeddings or feature vectors. These features are then used as inputs to a logistic regression model to predict the sentiment of the input sentences.


### Choice of predictive model

The best model for this Natural Language Processing (NLP) assignment was selected using cross-validation. The following machine learning models were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

The cross_val_score function was used to find the best parameters for each model with a 5-fold cross-validation, which divides the data into 5 equal parts and uses one part for validation and the other four for training in each iteration.

The results of the cross-validation are shown below for each model:

#### Logistic Regression
- Mean Cross Validation Score: 0.8253
- Standard Deviation of Cross Validation Score: 0.0179
- Accuracy Score: 0.8272
- F1 Score: 0.8172
- Precision Score: 0.8079
- Recall Score: 0.8272

#### Random Forest
- Mean Cross Validation Score: 0.7962
- Standard Deviation of Cross Validation Score: 0.0212
- Accuracy Score: 0.7973
- F1 Score: 0.7740
- Precision Score: 0.7637
- Recall Score: 0.7973

#### Gradient Boosting
- Mean Cross Validation Score: 0.8145
- Standard Deviation of Cross Validation Score: 0.0146
- Accuracy Score: 0.8139
- F1 Score: 0.7997
- Precision Score: 0.7876
- Recall Score: 0.8139

#### SVM
- Mean Cross Validation Score: 0.8145
- Standard Deviation of Cross Validation Score: 0.0153
- Accuracy Score: 0.8272
- F1 Score: 0.8013
- Precision Score: 0.8050
- Recall Score: 0.8272

Based on these results, we chose to use the Logistic Regression model to classify the sentiment of the movie reviews. The Logistic Regression model achieved the highest mean accuracy score, and also achieved the highest F1 score and precision score on the validation set.


# Results

Our best result is obtained using Logistic Regression tuned with GridSearchCV, and yields an accuracy of 82.45 % on the dev set. 

