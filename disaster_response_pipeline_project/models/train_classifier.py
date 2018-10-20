import sys
import pickle

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    '''
    load data from sqlite database for use in ML model
    input : database_filepath: flepath of database
    output: X: features
    y: targets
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table = 'messages'
    df = pd.read_sql_table(table, engine)
    #create X
    X = df['message'].values
    #create y
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    #create list of category names
    category_names = list(y.columns)
    y = y.values
    return X, y, category_names


def tokenize(text):
    '''  tokenization function to process text data
    Input: text: text to tokenize 
    Ouput: clean_tokens: list of cleaned tokens
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    ''' build model for classification
    output: cv
    
    '''
    # build pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)), ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))])
    #paramters for grid search
    parameters = {'clf__estimator__n_estimators': [10,50,100],}
    #build model
    cv = GridSearchCV(pipeline, param_grid=parameters , verbose = 2)
    return cv
	



def evaluate_model(model, X_test, y_test, category_names):
    '''
    Evaluate model against test data (Caculate accuracy, precision, and recall of model) 
    Input:
    model: trained model
    X_test: test features 
    Y_test: test labels 
    category_names: names of categories
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=category_names))
    print('Accuracy score of model:', (y_pred == y_test).mean())




def save_model(model, model_filepath):
     
    '''
    Save model to pickel file
    input: 
    model: trained ML model_filepath
    model_filepath: filepath to save pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

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
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()