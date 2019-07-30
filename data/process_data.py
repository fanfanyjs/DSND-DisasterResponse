import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories csv files as dataframes
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    print('Shape of messages file: ', messages.shape)
    print('Shape of categories file: ', categories.shape)
    
    messages_unique = messages.drop_duplicates(subset=['id'])
    categories_unique = categories.drop_duplicates(subset=['id'])
    
    print('Num of duplicate ids dropped from messages file: ', \
          len(messages) - len(messages_unique))
    print('Num of duplicate ids dropped from categories file: ', \
          len(categories) - len(categories_unique))
    
    df = messages_unique.merge(categories_unique, left_on = 'id', right_on = 'id')
    
    return df


def clean_data(df):
    '''
    Parse categories column in df and return cleaned df
    '''
              
    if 'categories' in df.columns:
        categories = df['categories'].str.split(";", expand=True)
        row = categories.iloc[0,:].values
        category_colnames = [i.split('-')[0] for i in row]
        categories.columns = category_colnames
        
        for column in category_colnames:
            categories[column] = categories[column].apply(lambda x:int(float(x[-1])))
     
        df = df.drop(['categories'], axis = 1)
        df = df.join(categories)
        
    return df



def save_data(df, database_filepath = '../data/cleaned_data.db'):
    '''
    Writes data to indicated SQLite filepath. Table named as 'cleaned_data'
    '''
          
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('cleaned_data', engine, index=False)
    print('Cleaned data saved to database!')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()