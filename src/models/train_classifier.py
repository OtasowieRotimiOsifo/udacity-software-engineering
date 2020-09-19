import sys
import os

import pandas as pd
from sqlalchemy import create_engine



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pickle

# local packages 
cwd = os.getcwd()
sys.path.append(cwd)


from basic_utilities import basic_utils
from basic_utilities import model_builder
from basic_utilities import analysis_tools


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


        
def evaluate_model(model: Pipeline, X_test: pd.DataFrame, 
                   Y_test: pd.DataFrame) -> None:
    """
    

    Parameters
    ----------
    model : Pipeline
        trained machine learning pipeline to be evaluated against unseen data.
    X_test : pd.DataFrame
        test data to be used for evaluation
    Y_test : pd.DataFrame
        data frame with expected values to be used as reference.

    Returns
    -------
    no return

    """
    
    Y_pred = model.predict(X_test)
    
    Y_test = Y_test.iloc[:,].reset_index(drop=True)
    Y_pred = pd.DataFrame(data=Y_pred, columns=Y_test.columns)
    
    analysis_tools.display_results(Y_test, Y_pred)
    
    
def save_model(model: Pipeline, model_filepath: str) -> None:
    """
    

    Parameters
    ----------
    model : Pipeline
        trained pipeline model.
    model_filepath : str
        file path for saving the trained model

    Returns
    -------
    No return 

    """
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
    
        X = basic_utils.do_pos_tagging(X)
        
        X_train, X_test, Y_train, Y_test = train_test_split(
                                                            X, 
                                                            Y, 
                                                            test_size=0.3,
                                                            random_state=42)
        
        analysis_tools.explore_data_with_plotly(Y, 'All Data: ')
        analysis_tools.explore_data_with_plotly(Y_train, 'Training Data: ')
        analysis_tools.explore_data_with_plotly(Y_test, 'Test Validation Data: ')
        
        print('Building model...')
        modelBuilder = model_builder.ModelBuilder()
        model = modelBuilder.build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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