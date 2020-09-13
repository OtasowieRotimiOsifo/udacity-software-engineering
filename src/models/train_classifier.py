import sys

import nltk

import re
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


from typing import List
import pickle

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
stopwords = nltk.corpus.stopwords.words('english')
punctuations = list(string.punctuation)
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

class Entity:
    """
    plain object for storing data from a database
    """
    def __init__(self, feature_vector, target_matrix, category_names, df, original='', genre=''):
        self.feature_vector = feature_vector
        self.target_matrix = target_matrix
        self.category_names = category_names
        self.genre = genre
        self.original = original
        self.df = df
    
def remove_empty(df: pd.DataFrame) -> pd.DataFrame:
    """
    remove trailing spaces from texts in selected columns
    Parameters
    ----------
    df : pd.DataFrame
        dataframe with text columns

    Returns
    -------
    df : TYPE
        dataframe with trailind spaces removed from selected columns.

    """
    df['message'].apply(lambda x: x.lstrip())
    
    df['message'].apply(lambda x: x.rstrip())
    return df

def count_punct(text: str) -> float:
    """
    Counts the % number of punctuation marks in input test

    Parameters
    ----------
    text : str
        input text with or without punctuationmarks.

    Returns
    -------
    float
        the % number of punctuation marks rounded to 3 decimal places.

    """
    if (len(text) - text.count(" ") > 0):
        count = sum([1 for ch in text if ch in string.punctuation])
        return round(count/(len(text) - text.count(" ")), 3) * 100
    else:
        return 0
    
def load_data(database_filepath: str) -> Entity:
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
    
    df = remove_empty(df)
    
    df['punt_perc'] = df['message'].apply(lambda x: count_punct(x))
    df['text_len'] = df['message'].apply(lambda x: len(x) - x.count(" "))
    
    category_name_list = df.columns[4:40]
    
    x = [df.message, df.text_len, df.punt_perc]
    headers = ['message', 'text_len', 'punt_perc']
    X = pd.concat(x, axis=1, keys=headers)
    
    Y = df.iloc[:, range(4, 40)]
    
    genre = df.genre
    
    original = df.original
    
    entity = Entity(X, Y, category_name_list,  df, original, genre)
    return entity

def tokenize(text: str) -> str:
    """
    cleans, tokenizes and lemmatizes input text

    Parameters
    ----------
    text : str
        DESCRIPTION.

    Returns
    -------
    str
      tokenized text

    """
    lemmatizer = WordNetLemmatizer()
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    words_list = [word.lower().strip() for word in \
                  word_tokenize(text) if len(word) > 0]
        
    words_list_no_punct = [word for word in words_list \
                           if word not in punctuations]
        
    words_list_no_punct_no_stop = [word for word in words_list_no_punct \
                                   if word not in stopwords]
        
    cleaned_text = [lemmatizer.lemmatize(word).strip() \
                    for word in words_list_no_punct_no_stop]
        
    return cleaned_text

class StartingVerbTagger:
    """
    abstract representation of an object for doing POS tagging 
    """
    def __get_pos_tags_ext(self, text: str) -> List:
        """
        performs POS tagging of input text

        Parameters
        ----------
        text : str
            text containing parts of speech

        Returns
        -------
        List
            a list of POS tags

        """
        pos_tags = list()
        
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            tags = nltk.pos_tag(tokenize(sentence))
            if len(tags) > 0:
                pos_tags.append(tags)
        
        return pos_tags
    
    def __starting_verb_ext(self, text: str) -> int:
        """
        

        Parameters
        ----------
        text : str
            DESCRIPTION.

        Returns
        -------
        int
            1 for starting verb, zero for otherwise
        """
        pos_tags_list = self.__get_pos_tags_ext(text)
        
        for pos_tags in pos_tags_list:
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        
        return 0
    
    def do_tagging(self, df: pd.DataFrame) ->  pd.DataFrame:
        """
        

        Parameters
        ----------
        df : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        df : pd.DataFrame
            dataframe with added column for starting verb flag

        """
        feature_vector_train = df['message'].values
        
        X_tagged = pd.Series(feature_vector_train).apply(self.__starting_verb_ext)
        df['starting_verb_flag'] = X_tagged.values
        
        return df

def do_pos_tagging(X: pd.DataFrame) -> pd.DataFrame:
    """
    

    Parameters
    ----------
    X : pd.DataFrame
        dataframe with text for POS tagging

    Returns
    -------
    X : TYPE
        dataframe with extra column for starting verb flag 

    """
    startingVerbTagger = StartingVerbTagger()
    X = startingVerbTagger.do_tagging(X)
    
    return X

class TextFieldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.field]

class NumericFieldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        return X[[self.field]]

class PosFieldExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        return X[[self.field]]
    
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
                [('text_field_extractor', TextFieldExtractor('message')), 
                 ('tfidf', TfidfVectorizer(tokenizer=tokenize, min_df=.0025, max_df=0.5, ngram_range=(1,2)))
                 ])),
             ('numerics', FeatureUnion(
                                   [('text_len', Pipeline([('text_len_extractor', NumericFieldExtractor('text_len')), 
                                                       ('text_len_scaler', StandardScaler())
                                                       ])),
                                    ('punt_perc', Pipeline([('punt_perc_extractor', NumericFieldExtractor('punt_perc')), 
                                                       ('punt_perc_scaler', StandardScaler())
                                                       ]))
                                   ])),
             ('starting_verb', PosFieldExtractor('starting_verb_flag'))
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
        
        X = do_pos_tagging(X)
        
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