# Modular class that loads data for the given justice and evaluates using the model that is added. 
# balances using undersampling by default

import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer




class JusticeModel:
    
    def __init__(self, justice, mode='crossval', model=None):
        
        self.justice = justice.capitalize()
        self.load_justice()
        self.mode=mode
        self.model = model
        self.tfidf = TfidfVectorizer()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_vectors = None
        self.test_vectors = None
        
    def load_justice(self):
        '''
        Loads the appropriate dataframe for the justice. 
        Splits the dataframe into a training dataset and testing dataset.
        '''
        fpath = 'data/clean/%s.csv' % self.justice
        df = pd.read_csv(fpath)
        df['cleanText'] = df['cleanText'].fillna(' ')
        self.data = df

    def add_model(self, model):
        self.model = model
        
    def fit(self, balance_method='undersampling'):
        '''
        Vectorizes training data, compiles the model, and trains it.
        '''
        
        if self.model == None:
            print('Add Model First')
            return
        
        X = self.data['cleanText'].to_numpy()
        y = self.data['vote'].to_numpy()
        
        if self.mode == 'full':
            X_train, y_train = self.balance_train_data(X.reshape(X.shape[0], 1), y, method=balance_method)
            
            self.X_train = X_train
            self.y_train = y_train
            
            self.train_vectors = self.tfidf.fit_transform(X_train.flatten())
            self.model.fit(self.train_vectors, y_train)
            print('Model trained successfully.')
            
        elif self.mode == 'crossval':


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
            X_train, y_train = self.balance_train_data(X_train.reshape(X_train.shape[0],1), y_train, method=balance_method)

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            self.train_vectors = self.tfidf.fit_transform(X_train.flatten())
            self.model.fit(self.train_vectors, y_train)
            print('Model trained successfully.')
        
    def predict(self, X=np.array([])):
        if len(X) == 0:
            self.test_vectors = self.tfidf.transform(self.X_test)
            self.y_preds = self.model.predict(self.test_vectors)
        else:
            self.test_vectors = self.tfidf.transform(X)
            self.y_preds = self.model.predict(self.test_vectors)
            return self.y_preds

    def balance_train_data(self, X, y, method=None):
        if method == None:
            return X, y

        elif method == 'undersampling':
            rus = RandomUnderSampler()
            X_train, y_train = rus.fit_resample(X, y)
            return X_train, y_train

        elif method == 'oversampling':    
            ros = RandomOverSampler()
            X_train, y_train = ros.fit_resample(X, y)
            return X_train, y_train

        elif method == 'smote':
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X, y)
            return X_train, y_train

        elif method == 'both':
            smote = SMOTE(sampling_strategy=0.75)
            under = RandomUnderSampler(sampling_strategy=1)
            X_train, y_train = smote.fit_resample(X, y)
            X_train, y_train = under.fit_resample(X_train, y_train)
            return X_train, y_train

        else:
            print('Incorrect balance method')
            return