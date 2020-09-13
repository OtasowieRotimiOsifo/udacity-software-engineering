import sys
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine




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
    except Exception as er:
        print(er)
    return df

def explore_target_categories(df: pd.DataFrame):
    """
    
    counts and plots the number of each target class in each 
    column of categories
    
    Parameters
    ----------
    df : pd.DataFrame
        data frame containing cleaned data

    Returns
    -------
    None.

    """
    target_categories = df.iloc[:, range(4, 40)]
    groups_dict = dict()
    for column in target_categories.columns:
        groups_dict[column] = target_categories[column].value_counts()
    
    class_counts_df = pd.DataFrame(groups_dict)
    
    fig, ax = plt.subplots()
    fig.suptitle("Target Class Counts in Categories", fontsize=12)
    class_counts_df.plot(kind="bar", legend=False, ax=ax)#.grid(axis='x')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Target Category Classes')
    plt.show()

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
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = list(map(lambda x: x[len(x)-1], categories[column].values))
        # convert column from string to numeric
        categories[column] = categories.astype({column:int})
    
    df.drop(columns=['categories'], inplace=True)
    
    df = pd.concat([df, categories], axis=1)
    
    df.drop_duplicates(keep=False, inplace=True)
    
    df.dropna(inplace=True)
    
    df.reset_index(drop=True, inplace=True)
    
    return df
    
def save_data(df: pd.DataFrame, database_filename: str):
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
    df.to_sql('DisasterResponse', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, \
            database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        explore_target_categories(df)
        
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