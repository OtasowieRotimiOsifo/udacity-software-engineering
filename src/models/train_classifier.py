import sys
import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import pickle

# local packages 
cwd = os.getcwd()
sys.path.append(cwd)


from basic_utilities import basic_utils
from basic_utilities import model_builder
from basic_utilities import analysis_tools

        
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
    
    #analysis_tools.display_results(Y_test, Y_pred)
    analysis_tools.display_classification_reports(Y_test, Y_pred)
    
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
        entity = basic_utils.load_data(database_filepath)
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
        #model = modelBuilder.build_model()
        model =  modelBuilder.build_model_from_grid_searchCV(X_train, Y_train)
        
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