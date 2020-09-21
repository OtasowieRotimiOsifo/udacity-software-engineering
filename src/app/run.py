import sys
import os

import json
import plotly
import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from plotly.graph_objects import Bar
import plotly.graph_objects as g_o
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

#from typing import List

cwd = os.getcwd()
sys.path.append(cwd)

from basic_utilities import basic_utils
from basic_utilities import analysis_tools



app = Flask(__name__)


#def tokenize(text):
#    tokens = word_tokenize(text)
#    lemmatizer = WordNetLemmatizer()

#    clean_tokens = []
#    for tok in tokens:
#        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#        clean_tokens.append(clean_tok)

#    return clean_tokens

# load data
data_base_file_path = cwd + '/DisasterResponse.db'
engine = create_engine('sqlite:///' + data_base_file_path) 
df_db = pd.read_sql_table('DisasterResponse', engine)

# load model
pickle_file_path = cwd + "/classifier.pkl"
model = joblib.load(pickle_file_path)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = df.groupby('genre').count()['message']
    #genre_names = list(genre_counts.index)
    
    target_categories = df_db.iloc[:, range(4, 40)]
    class_counts = analysis_tools.count_column_values(target_categories)
    #class_counts.dropna(inplace=True)
    #print(class_counts.head())
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    #get_plotly_graph(df: pd.DataFrame, data_info='')
    #graphs = [
    #    {
    #        'data': [
    #            Bar(
    #                #x=genre_names,
    #                y=class_counts
    #            )
    #        ],

    #         'layout': {
    #            'title': 'Distribution of Message Genres',
    #            'yaxis': {
    #                'title': "Count"
    #            },
    #            'xaxis': {
    #                'title': "Genre"
    #            }
    #        }
    #    }
    #]
    
    #fig = analysis_tools.get_plotly_data(class_counts, 'All Data: ')
    #fig = g_o.Figure()
    print(class_counts.iloc[0][0:36].values)
    #data = [Bar(x=class_counts.columns, y=class_counts.iloc[0][0:36].values)]
    
    # layout = {
    #     'title':'All Data: Counts of target classes per category',
    #     'yaxis': {
    #         'title': 'Frequency'
    #         },
    #     'xaxis': {
    #         'title': 'Categories'
    #         }
    # }
    
    
    # graphs = [
    #     {
    #      'data': [
    #          Bar(
    #              x=class_counts.columns, 
    #              y=class_counts.iloc[0][0:36].values
    #         )
    #     ],
         
    #         'layout': {
    #         'title':'All Data: Counts of target classes per category',
    #         'yaxis': {
    #             'title': 'Frequency'
    #             },
    #         'xaxis': {
    #             'title': 'Categories'
    #             }
    #     }
    #   }
    # ]
    
    graphs = [analysis_tools.get_plotly_data(class_counts, 'All Data: ')]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)
   
    

def initialize_input_query(query: str)  -> pd.DataFrame:
    query_dict = {'message':query}
    df = pd.DataFrame([query_dict])
    
    df = basic_utils.remove_empty(df)
    
    df['punt_perc'] = df['message'].apply(lambda x: basic_utils.count_punct(x))
    df['text_len'] = df['message'].apply(lambda x: len(x) - x.count(" "))
    
    df = basic_utils.do_pos_tagging(df)
    
    return df
    
    

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    
    df = initialize_input_query(query)
    # use model to predict classification for query
    classification_labels = model.predict(df)[0]
    classification_results = dict(zip(df_db.columns[4:], classification_labels))
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