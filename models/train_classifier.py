# import packages
import warnings
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')


def load_data(database_filepath = '../data/cleaned_data.db'):
    '''
    Load in data from designated filepath
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('cleaned_data', engine)
    X = df[['message', 'genre', 'original']]
    y = df.drop(['id', 'genre', 'message', 'original'], axis = 1)
    category_names = list(y.columns)

    return X, y, category_names


def tokenize_fct(text):
    
    '''
    Tokenize by word, remove stop words, lemmatize all phrases in the dataframe
    
    INPUT:
    text
    
    OUTPUT:
    df - df with three new columns: one normalised, one tokenized, another with stop words removed in addition, and the
    last one lemmatized in addition to previous steps
    '''

    new_text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    new_text = word_tokenize(new_text)

    sw_list = stopwords.words("english")
    eng_words = set(nltk.corpus.words.words())
    new_text = [w for w in new_text if w not in sw_list]
    new_text = [w for w in new_text if w in eng_words]

    for postag in ['a','s','r','n','v']:
        new_text = [WordNetLemmatizer().lemmatize(w, pos = postag) for w in new_text]        
    
    return new_text

class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y = None, col_name = ['message'], vect_size = 50):
        '''
        Tokenize sentences, then fit with w2v model (for word vectors) and tfidf 
        for weighting
        '''
        self.vect_size = vect_size
        
        self.tok_X = []
        for line in X[col_name].values:
            self.tok_X.append(tokenize_fct(line[0]))
        
        # Fit w2v model
        model = Word2Vec(self.tok_X, size=vect_size)
        self.word_vector = dict(zip(model.wv.index2word, model.wv.vectors))
        
        # Fit tfidf for weighting
        tfidf = TfidfVectorizer(analyzer=lambda x: x, lowercase = False)
        tfidf.fit(self.tok_X)
        self.word_list = list(tfidf.vocabulary_.keys())
        self.idf_list = list(tfidf.idf_[list(tfidf.vocabulary_.values())])
        self.word_weight = dict(zip(word_list, idf_list))
        
        print('Embedding transformer fitted!')
        return self

    def transform(self, X):
        '''
        return weighted mean vector of vect size as specified above
        '''
        self.weighted_vector = np.empty((0, self.vect_size), float)
        for line in self.tok_X:
            if len(line)>0:
                new_weight = [self.word_vector[w] * self.word_weight[w] for w in line \
                              if (w in self.word_vector) and (w in self.word_weight)]\
                            + [np.zeros(self.vect_size) for w in line \
                               if (w not in self.word_vector) or (w not in self.word_weight)]
                new_weight_mean = np.mean(new_weight, axis = 0).reshape(1, -1)
                self.weighted_vector = np.concatenate((self.weighted_vector, new_weight_mean), axis = 0)
            else:
                self.weighted_vector = np.concatenate((self.weighted_vector, 
                                                       np.zeros(self.vect_size).reshape(1,-1)), axis = 0)
            
        return pd.DataFrame(self.weighted_vector)

class tfidfExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, col_name = 'message'):
        self.col_name = col_name
        self.countvect = CountVectorizer(tokenizer = tokenize_fct)
        X_cvect = self.countvect.fit_transform(X[col_name].values.ravel())
        self.tfidftrans = TfidfTransformer(smooth_idf=False)
        self.tfidftrans.fit(X_cvect)
        
        print('tfidf transformer fitted!')
        return self

    def transform(self, X):
        # apply tfidf
        X_cvect = self.countvect.transform(X[self.col_name].values.ravel())
        X_tfidf = self.tfidftrans.transform(X_cvect)
        return X_tfidf
    
    
class GenreExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, col_name = 'genre'):
        self.col_name = col_name
        print('Genre transformer fitted!')
        return self

    def transform(self, X):
        # apply one hot encoding
        X_one_hot = pd.get_dummies(X[self.col_name], drop_first = True)
        return X_one_hot

    
class LengthExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, col_name = 'original'):
        self.col_name = col_name
        print('Length transformer fitted!')
        return self

    def transform(self, X):
        new_col = pd.DataFrame(X[self.col_name].fillna('').apply(lambda x:len(x)))
        return new_col.values


def build_model():
    # text processing and model pipeline
    eng_words = set(nltk.corpus.words.words())
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('length', LengthExtractor()),
            ('tfidf', tfidfExtractor()),
            ('genre_onehot', GenreExtractor())
        ])),
        ('mo_class', MultiOutputClassifier(RandomForestClassifier(random_state = 42)))
    ])

    # define parameters for GridSearchCV
    parameters = {
        'mo_class__estimator__n_estimators': [100],
        'mo_class__estimator__min_samples_split': [50, 100]
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(pipeline, parameters, verbose = 5, n_jobs = -1)

    return model_pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Split model into train and test sets, run training data through the model pipeline and
    print out classification report results for both test and train data
    '''
    
    Y_test_pred = model.predict(X_test.values.ravel())
    
    print(classification_report(Y_test.values, Y_test_pred, target_names = category_names))


def save_model(model, model_filepath='../models/trained_model.pkl'):
    '''
    Exports model to pickle file in models folder. File name will be 'trained_model.pkl'
    '''
    joblib.dump(value=model, filename=model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/cleaned_data.db trained_model.pkl')


if __name__ == '__main__':
    main()