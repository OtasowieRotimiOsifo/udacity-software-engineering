import sys

import pandas as pd
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler


import pickle

sys.path.append('../')
from basic_utilities import basic_utils

#nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
#stopwords = nltk.corpus.stopwords.words('english')
#punctuations = list(string.punctuation)
#url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    
def load_data(database_filepath: str) -> basic_utils.Entity:
    """
    retrieves masseages data from an sqlite database file 

    Parameters
    ----------
    database_filepath : str
        path to sqlite database file.

    Returns
    -------
    Entity
        abstract representation of data retrieved from a database.

    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(con=engine, table_name='DisasterResponse')
    
    df = basic_utils.remove_empty(df)
    
    df['punt_perc'] = df['message'].apply(lambda x: basic_utils.count_punct(x))
    df['text_len'] = df['message'].apply(lambda x: len(x) - x.count(" "))
    
    category_name_list = df.columns[4:40]
    
    x = [df.message, df.text_len, df.punt_perc]
    headers = ['message', 'text_len', 'punt_perc']
    X = pd.concat(x, axis=1, keys=headers)
    
    Y = df.iloc[:, range(4, 40)]
    
    genre = df.genre
    
    original = df.original
    
    entity = basic_utils.Entity(X, Y, category_name_list,  df, original, genre)
    return entity




    
def build_model() -> Pipeline:
    """
    creates a piline for natural language processing based on tfidf

    Returns
    -------
    Pipeline
        pipline contining the steps for training a classifier for nlp.

    """
    clf = RandomForestClassifier(
                                 n_estimators=200,
                                 max_features='auto',
                                 min_samples_leaf=1,
                                 min_samples_split=3,
                                 random_state=42, 
                                 n_jobs=-1)
    model = MultiOutputClassifier(clf)
    
    pipeline = Pipeline([
        ('features', FeatureUnion(
            [('text', Pipeline(
                [('text_field_extractor', basic_utils.TextFieldExtractor('message')), 
                 ('tfidf', TfidfVectorizer(tokenizer=basic_utils.tokenize, min_df=.0025, max_df=0.5, ngram_range=(1,2)))
                 ])),
             ('numerics', FeatureUnion(
                                   [('text_len', Pipeline([('text_len_extractor', basic_utils.NumericFieldExtractor('text_len')), 
                                                       ('text_len_scaler', StandardScaler())
                                                       ])),
                                    ('punt_perc', Pipeline([('punt_perc_extractor', basic_utils.NumericFieldExtractor('punt_perc')), 
                                                       ('punt_perc_scaler', StandardScaler())
                                                       ]))
                                   ])),
             ('starting_verb', basic_utils.PosFieldExtractor('starting_verb_flag'))
            ])),
        ('clf', model)
    ])
    
    return pipeline


def get_classification_reports(Y_test, Y_pred, category_names):
    report_dict = dict()
    reports_dict = dict()
    for category_name in category_names:
        y_test = Y_test[category_name].values
        
        y_pred = Y_pred[category_name].values
        
        report = classification_report(y_test, y_pred)
        report_dict['report'] = report
        
        reports_dict[category_name] = report_dict
    return reports_dict

def display_prediction_report(reports_dict):
    for key in reports_dict.keys():
        report_dict = reports_dict[key]
        print('Scores for ', key, ': ')
        print('---------------------')
        print(report_dict['report'])
        
def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    
    Y_test = Y_test.iloc[:,].reset_index(drop=True)
    Y_pred = pd.DataFrame(data=Y_pred, columns=Y_test.columns)
    
    reports_dict = get_classification_reports(Y_test, Y_pred, category_names)
    display_prediction_report(reports_dict)
    
    
def save_model(model, model_filepath):
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
    except Exception as er:
        print(er)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        entity = load_data(database_filepath)
        X = entity.feature_vector
        Y = entity.target_matrix
        category_names = entity.category_names
        
        X = basic_utils.do_pos_tagging(X)
        
        X_train, X_test, Y_train, Y_test = train_test_split(
                                                            X, 
                                                            Y, 
                                                            stratify=Y,
                                                            test_size=0.2,
                                                            random_state=42)
        
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