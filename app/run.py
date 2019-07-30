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
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('words')

app = Flask(__name__)

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
        
        print('Embedding fitted!')
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
        
        print('tfidf fitted!')
        return self

    def transform(self, X):
        # apply tfidf
        X_cvect = self.countvect.transform(X[self.col_name].values.ravel())
        X_tfidf = self.tfidftrans.transform(X_cvect)
        return X_tfidf
    
    
class GenreExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, col_name = 'genre'):
        self.col_name = col_name
        print('Genre fitted!')
        return self

    def transform(self, X):
        # apply one hot encoding
        X_one_hot = pd.get_dummies(X[self.col_name], drop_first = True)
        return X_one_hot

    
class LengthExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None, col_name = 'original'):
        self.col_name = col_name
        return self

    def transform(self, X):
        new_col = pd.DataFrame(X[self.col_name].fillna('').apply(lambda x:len(x)))
        return new_col.values

# load data
def load_data(data_filepath = '../data/cleaned_data.db'):
    '''
    Load in data from designated filepath
    '''
    engine = create_engine('sqlite:///' + data_filepath)
    df = pd.read_sql_table('cleaned_data', engine)
    return df

df = load_data()

# load model
model = joblib.load("../models/trained_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    Y = df.drop(['id', 'genre', 'message', 'original'], axis = 1)
    top5_cat = (Y.sum()/len(Y)).sort_values(ascending = False)[:5]
    top5_cat_val = top5_cat.values * 100
    top5_cat_names = [i.replace('_', ' ') for i in list(top5_cat.index)]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x=top5_cat_names,
                    y=top5_cat_val
                )
            ],
            
            'layout': {
                'title': 'Top 5 message categories',
                'yaxis': {
                    'title': "Proportion of messages in the category (%)"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
