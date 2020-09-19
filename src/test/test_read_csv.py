import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/disaster_categories.csv')
print(df.iloc[0][0:])
categories = df['categories'].str.split(';', expand=True)
rows = categories.iloc[0].values
print(rows)
category_colnames = list(map(lambda x: x[0:len(x)-2], rows))
categories.columns = category_colnames
print(categories.columns)
print(categories.head())


def set_values_for_categories(categories: pd.DataFrame) -> pd.DataFrame:
    cols = categories.iloc[0].values
    values = list(map(lambda x: x[len(x)-1: len(x)], cols))
    
    df_dict = dict()
    for name, value in zip(category_colnames, values):
        col_values = list()
        col_values.append(int(value))
        df_dict[name] = col_values
    
    print(df_dict)
    for cnt in range(1, len(df)):
        cols = categories.iloc[cnt].values
        values = list(map(lambda x: x[len(x)-1: len(x)], cols))
        for name, value in zip(category_colnames, values):
            col_values = df_dict[name]
            col_values.append(int(value))
            df_dict.pop(name)
            df_dict[name] = col_values
    categories_df = pd.DataFrame(df_dict)
    categories_df.drop(categories_df.loc[categories_df['related']==2].index, inplace=True)
    return categories_df

categories_df = set_values_for_categories(categories)

groups_dict_loc = dict()
for column in categories_df.columns:
    groups_dict_loc[column] = categories_df[column].value_counts()

print(groups_dict_loc)
     
def explore_target_categories(target_categories: pd.DataFrame):
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
    
explore_target_categories(categories_df)
