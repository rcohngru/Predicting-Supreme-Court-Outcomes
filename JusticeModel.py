# Modular class that loads data for the given justice and evaluates using the model that is added. 
# Evaluates accuracy and f-1 score
# balances using undersampling by default

class JusticeModel:
    
    def __init__(self, justice, model=None):
        
        self.justice = justice.capitalize()
        self.load_justice()
        self.model = model
        self.tfidf = TfidfVectorizer()
        
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
       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)
        X_train, y_train = self.balance_train_data(X_train, y_train, method=balance_method)
    
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
                      
        self.train_vectors = self.tfidf.fit_transform(X_train)
        self.model.fit(self.train_vectors, y_train)
        
    def predict(self):
        self.test_vectors = self.tfidf.transform(self.X_test)
        self.y_preds = self.model.predict(self.test_vectors)
        
    def evaluate(self):
        acc = accuracy_score(self.y_test, self.y_preds)
        f1 = f1_score(self.y_test, self.y_preds)
        print('Accuracy: %.3f' % acc)
        print('F1 Score: %.3f' % f1)
        return acc, f1

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