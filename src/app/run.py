import sys
import os

import json
import plotly
import pandas as pd


from flask import Flask
from flask import render_template, request
import joblib
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

cwd = os.getcwd()
sys.path.append(cwd)

from basic_utilities import basic_utils
from basic_utilities import analysis_tools



app = Flask(__name__)


# load data
database_filepath = cwd + '/DisasterResponse.db'

entity = basic_utils.load_data(database_filepath)
X = entity.feature_vector
Y = entity.target_matrix

X_train, X_test, Y_train, Y_test = train_test_split(
                                                    X, 
                                                    Y, 
                                                    test_size=0.3,
                                                    random_state=42)

# load model
pickle_file_path = cwd + "/classifier.pkl"
model = joblib.load(pickle_file_path)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    class_counts_all = analysis_tools.count_column_values(Y)
    class_counts_train = analysis_tools.count_column_values(Y_train)
    class_counts_test = analysis_tools.count_column_values(Y_test)
    
    graphs = [
        analysis_tools.get_plotly_data(class_counts_all, 'All Data: '),
        analysis_tools.get_plotly_data(class_counts_train, 'Training Data: '),
        analysis_tools.get_plotly_data(class_counts_test, 'Validation Data: ')
        ]
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
    
    inputPreprocessor = basic_utils.InputPreprocessor()
    df = inputPreprocessor.pre_process_input(query)
    
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