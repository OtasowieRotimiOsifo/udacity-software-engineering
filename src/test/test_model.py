#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:13:02 2020

@author: otasowie
"""
import sys

import pandas as pd
from sqlalchemy import create_engine
import joblib

sys.path.append('../')
from basic_utilities import basic_utils

engine = create_engine('sqlite:///../DisasterResponse.db') 
df = pd.read_sql_table('DisasterResponse', engine)
#print(df['message'][])
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)
print('After loading')
print(genre_names)

# load model
#model_file_path = #'/media/otasowie/Elements/latest-2020-sept-06/latest-latest/udacity-project-3-files/latest/disaster_response_pipeline_project/classifier.pkl'
model_file_path = 'c:\dev\\datascience\\udacity-software-engineering\\src\\classifier.pkl'
model = joblib.load(model_file_path)
#print(model)

def initialize_input_query(query: str)  -> pd.DataFrame:
    query_dict = {'message':query}
    df = pd.DataFrame([query_dict])
    
    df = basic_utils.remove_empty(df)
    
    df['punt_perc'] = df['message'].apply(lambda x: basic_utils.count_punct(x))
    df['text_len'] = df['message'].apply(lambda x: len(x) - x.count(" "))
    
    df = basic_utils.do_pos_tagging(df)
    
    return df

query =  'I am in need of transport' #df['message'][50] #'Weather update - a cold front from Cuba that could pass over Haiti'
#print(df.iloc[50][4:40].values)
df_loc = initialize_input_query(query)
result = model.predict(df_loc)
classification_labels = result[0]
print(classification_labels)