#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:29:32 2020

@author: otasowie
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from typing import Dict
from typing import List


def get_plotly_data(df: pd.DataFrame, data_info='') -> List:
    """
    returns a plotly bar char object of the values in a data frame

    Parameters
    ----------
    df : pd.DataFrame
        Data frame conting data for plotting
    data_info: str 
        extra information about the data being displayed

    Returns
    -------
    plotly figure object
    """
    
    data = list()
    for column in df.columns:
        data.append(go.Bar(name=column, x=df[column].index, y=df[column].values))
    
    layout = go.Layout(
        title=data_info + 'Counts of target classes per category',
        legend_title='Categories',
        xaxis_title='Target Classes',
        yaxis_title='Counts'
    )
    
    fig = go.Figure(data, layout)
    
    return fig

def get_plotly_graph(df: pd.DataFrame, data_info='') -> go.Figure:
    """
    returns a plotly bar char object of the values in a data frame

    Parameters
    ----------
    df : pd.DataFrame
        Data frame conting data for plotting
    data_info: str 
        extra information about the data being displayed

    Returns
    -------
    plotly figure object
    """
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Bar(name=column, x=df[column].index, y=df[column].values))
    
    layout = go.Layout(
        title=data_info + 'Counts of target classes per category',
        legend_title='Categories',
        xaxis_title='Target Classes',
        yaxis_title='Counts'
    )
    
    fig.update_layout(layout)
    
    return fig
    
def plot_df_data_with_plotly(df: pd.DataFrame, data_info='') -> None:
    """
    Plots bar char of the values in a data frame

    Parameters
    ----------
    df : pd.DataFrame
        Data frame conting data for plotting
    data_info: str 
        extra information about the data being displayed

    Returns
    -------
    no return
    """
    fig = go.Figure()
    for column in df.columns:
        fig.add_trace(go.Bar(name=column, x=df[column].index, y=df[column].values))
    
    layout = go.Layout(
        title=data_info + 'Counts of target classes per category',
        legend_title='Categories',
        xaxis_title='Target Classes',
        yaxis_title='Counts'
    )
    
    fig.update_layout(layout)
    fig.show()
 
def explore_data_with_plotly(df: pd.DataFrame, data_info=''):
    """
    
    counts and plots the number of each target class in each 
    column of the of the input dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        data frame containing cleaned data
    data_info: str 
        extra information about the data being displayed

    Returns
    -------
    None.

    """
  
    class_counts_df  = count_column_values(df)
    
    plot_df_data_with_plotly(class_counts_df, data_info)
    
def explore_data(df: pd.DataFrame):
    """
    
    counts and plots the number of each target class in each 
    column of the of the input dataframe
    
    Parameters
    ----------
    df : pd.DataFrame
        data frame containing cleaned data

    Returns
    -------
    None.

    """
  
    class_counts_df  = count_column_values(df)
    
    plot_df_data(class_counts_df)
    

def explore_target_categories_with_plotly(df: pd.DataFrame, data_info=''):
    """
    
    counts and plots the number of each target class in each 
    column of categories
    
    Parameters
    ----------
    df : pd.DataFrame
        data frame containing cleaned data
    data_info: str 
        extra information about the data being displayed

    Returns
    -------
    None.

    """
    target_categories = df.iloc[:, range(4, 40)]
    
    class_counts_df  = count_column_values(target_categories)
    
    plot_df_data_with_plotly(class_counts_df, data_info)
    
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
    
    class_counts_df  = count_column_values(target_categories)
    
    plot_df_data(class_counts_df)


def count_column_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    counts the total number of values in each column

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe comtaining numerical values.

    Returns
    -------
    TYPE
        Data frame containing the total number of values in each column.

    """
    groups_dict = dict()
    
    for column in df.columns:
        groups_dict[column] = df[column].value_counts()
        
    return pd.DataFrame(groups_dict)

def plot_df_data(df: pd.DataFrame) -> None:
    """
    Plots bar char of the values in a data frame

    Parameters
    ----------
    df : pd.DataFrame
        Data frame conting data for plotting

    Returns
    -------
    no return
    """
    
    fig, ax = plt.subplots()
    fig.suptitle("Target Class Counts in Categories", fontsize=12)
    df.plot(kind="bar", legend=False, ax=ax)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Target Category Classes')
    plt.show()
    
def get_confusion_matrix(expected_df: pd.DataFrame, 
                         predicted_df: pd.DataFrame) -> Dict:
    """
    

    Parameters
    ----------
    expected_df : pd.DataFrame
        expected classification matrix
    predicted_df: pd.DataFrame
        predicted classification matrix
       

    Returns
    -------
    Dict
      Dictionary of mean confusion matrices for each column in the input data frames
        

    """
    
    confusion_matices_dict = dict()
    for column in expected_df.columns:
         expected_values = expected_df[column].values
         predicted_values = predicted_df[column].values
         
         labels = np.unique(expected_values)
         confusion_mat = confusion_matrix(expected_values, 
                                          predicted_values, labels=labels)
         confusion_matices_dict[column] = {'labels': labels, \
                                           'confusion_matrix': confusion_mat}
    return confusion_matices_dict

def get_accuracy(expected_df: pd.DataFrame, \
                 predicted_df: pd.DataFrame) -> Dict:
    """
    

    Parameters
    ----------
    expected_df : pd.DataFrame
        expected classification matrix
    predicted_df: pd.DataFrame
        predicted classification matrix
       

    Returns
    -------
    Dict
      Dictionary of mean accuracy values for each column in 
      the input data frames
        

    """
    
    accuracy_dict = dict()
    for column in expected_df.columns:
         expected_values = expected_df[column].values
         predicted_values = predicted_df[column].values
         
         labels = np.unique(expected_values)
         accuracy = (predicted_values == expected_values).mean()
         accuracy_dict[column] = {'labels': labels, 'accuracy': accuracy}
    return accuracy_dict


def display_results(expected_df: pd.DataFrame, 
                    predicted_df: pd.DataFrame) -> None:
    """
    
    Parameters
    ----------
    expected_df : pd.DataFrame
        expected classification matrix
    predicted_df: pd.DataFrame
        predicted classification matrix
       
    """
    
    accuracy_dict = get_accuracy(expected_df, predicted_df)
    
    confusion_matices_dict = get_confusion_matrix(expected_df, predicted_df) 
    
    for key in accuracy_dict.keys():
        accuracy = accuracy_dict[key]['accuracy']
        labels = accuracy_dict[key]['labels']
        
        confusion_mat = confusion_matices_dict[key]['confusion_matrix']
        
        print('\n Scores for ', key, ': ')
        print('---------------------')
        print("Labels:", labels)
        print("Confusion Matrix:\n", confusion_mat)
        print("Accuracy:", accuracy)
        

def get_classification_reports(Y_test: pd.DataFrame, Y_pred: pd.DataFrame, \
                               category_names: List) -> Dict:
    """

    Parameters
    ----------
    Y_test : pd.DataFrame
        expected classification matrix
    Y_pred: pd.DataFrame
        predicted classification matrix     
    category_names: list of category names
    Returns
    -------
    Dict
      Dictionary of classification report for each column in the input data frames
      
    """
    report_dict = dict()
    reports_dict = dict()
    for category_name in category_names:
        y_test = Y_test[category_name].values
        
        y_pred = Y_pred[category_name].values
        
        report = classification_report(y_test, y_pred)
        report_dict['report'] = report
        
        reports_dict[category_name] = report_dict
    return reports_dict

def display_classification_reports(Y_test: pd.DataFrame, 
                                   Y_pred: pd.DataFrame) -> None:
    """

    Parameters
    ----------
    Y_test : pd.DataFrame
        expected classification matrix
    Y_pred: pd.DataFrame
        predicted classification matrix     
    category_names: list of category names
    Returns
    -------
    
      
    """
    category_names = Y_test.columns
    
    reports_dict = get_classification_reports(Y_test, Y_pred, category_names)
    
    for key in reports_dict.keys():
        report_dict = reports_dict[key]
        
        print('Scores for ', key, ': ')
        print('---------------------')
        print(report_dict['report'])
        

def display_prediction_report(reports_dict: Dict) -> None:
    """
    
    Displays classification reports to standard output.
    Parameters
    ----------
    reports_dict : Dict
        dictionary of classification reports.

    Returns
    -------
    None
        DESCRIPTION.

    """
    for key in reports_dict.keys():
        report_dict = reports_dict[key]
        
        print('Scores for ', key, ': ')
        print('---------------------')
        print(report_dict['report'])