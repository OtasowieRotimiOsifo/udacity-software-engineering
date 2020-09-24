import sys
import os

import pandas as pd

from sqlalchemy import create_engine

cwd = os.getcwd()
sys.path.append(cwd)

# local packages 
from basic_utilities import analysis_tools


def read_csv_file(filepath: str) -> pd.DataFrame:
   """
   
   Reads input csv file and returns a data frame
   
   Parameters
    ----------
    messages_filepath : str
        path to a csv file

    Returns
    -------
    pd.DataFrame

    """
   try:
        df = pd.read_csv(filepath)
   except Exception as er:
        print(er)
   return df
        
def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    
    delegates csv file reads and returns a data frame of merged data
    
    Parameters
    ----------
    messages_filepath : str
        path to the messages csv file
    categories_filepath : str
        path to the categories csv file

    Returns
    -------
    tuple of data frames

    """
    try:
        messages = read_csv_file(messages_filepath)
        categories = read_csv_file(categories_filepath)
        
        df = pd.merge(messages, categories, on='id')
        return df
    except Exception as er:
        print(er)
    return None

def set_values_for_categories(categories: pd.DataFrame) -> pd.DataFrame:
    """
    

    Parameters
    ----------
    categories : pd.DataFrame
        dataframe of categories names

    Returns
    -------
    categories_df : TYPE
        modified dataframe with columns, values and categories initialized 

    """
    cols = categories.iloc[0].values
    
    category_colnames = list(map(lambda x: x[0:len(x)-2], cols))
    categories.columns = category_colnames

    values = list(map(lambda x: x[len(x)-1: len(x)], cols))
    
    df_dict = dict()
    for name, value in zip(category_colnames, values):
        col_values = list()
        col_values.append(int(value))
        df_dict[name] = col_values

    for cnt in range(1, len(categories)):
        cols = categories.iloc[cnt].values
        values = list(map(lambda x: x[len(x)-1: len(x)], cols))
        for name, value in zip(category_colnames, values):
            col_values = df_dict[name]
            col_values.append(int(value))
            df_dict.pop(name)
            df_dict[name] = col_values
    
    categories_df = pd.DataFrame(df_dict)
    #categories_df.drop(categories_df.loc[categories_df['related']==2].index, inplace=True)
    
    return categories_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    cleans the data in the inpu for example removal of NAN.
    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing messages amd massage categories.

    Returns
    -------
    pd.DataFrame

    """
    categories = df['categories'].str.split(';', expand=True)
    rows = categories.iloc[0]
    category_colnames = list(map(lambda x: x[0:len(x)-2], rows.values))
    categories.columns = category_colnames
    
    categories = set_values_for_categories(categories)
    
    df.drop(columns=['categories'], inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(keep=False, inplace=True)
    
    df.dropna(inplace=True)
    index_list = df[ df['related'] == 2].index 
    df.drop(index_list, inplace = True)
    
    df.reset_index(drop=True, inplace=True)
    
    return df
    
def save_data(df: pd.DataFrame, database_filename: str) -> None:
    """
    

    Parameters
    ----------
    df : pd.DataFrame
        dataframe containing cleaned data
    database_filename : str
        path to database file name for storing df

    Returns
    -------
    None.

    """
    sql_arg = 'sqlite:///' + database_filename
    engine = create_engine(sql_arg) 
    df.to_sql('DisasterResponse', engine,  if_exists='replace', index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, \
            database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        analysis_tools.explore_target_categories_with_plotly(df, 'All Data: ')
        
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