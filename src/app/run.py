import json
import plotly
import pandas as pd
import string
import nltk
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from sqlalchemy import create_engine

from typing import List

nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])
stopwords = nltk.corpus.stopwords.words('english')
punctuations = list(string.punctuation)
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

app = Flask(__name__)

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
#def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()

#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)

#    return clean_tokens

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db') 
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

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
    
def initialize_input_query(query: str)  -> pd.DataFrame:
    query_dict = {'message':query}
    df = pd.DataFrame([query_dict])
    
    df = remove_empty(df)
    
    df['punt_perc'] = df['message'].apply(lambda x: count_punct(x))
    df['text_len'] = df['message'].apply(lambda x: len(x) - x.count(" "))
    
    df = do_pos_tagging(df)
    
    return df
    
    

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
    df = initialize_input_query(query)
    # use model to predict classification for query
    classification_labels = model.predict(df)[0]
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