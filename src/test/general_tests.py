#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 09:39:20 2020

@author: otasowie
"""
import os
import sys

import joblib
import plotly.graph_objects as go
#import plotly.express as px

from sqlalchemy import create_engine
#from typing import Dict

import pandas as pd

cwd = os.getcwd()
sys.path.append(cwd)

from basic_utilities import analysis_tools



pickle_file_path = cwd + "/classifier.pkl"

model = joblib.load(pickle_file_path)
print(model)

data_base_file_path = cwd + '/DisasterResponse.db'
engine = create_engine('sqlite:///' + data_base_file_path) 
df = pd.read_sql_table('DisasterResponse', engine)
target_categories = df.iloc[:, range(4, 40)]
class_counts = analysis_tools.count_column_values(target_categories)

fig = go.Figure()
for column in class_counts.columns:
    fig.add_trace(go.Bar(name=column, x=class_counts[column].index, y=class_counts[column].values))

layout = go.Layout(
    title='Counts of target classes per category',
    legend_title='Categories',
    xaxis_title='Target Classes',
    yaxis_title='Counts'
)
fig.update_layout(layout)
#fig.update_xaxes(title_text='Target Classes')
#fig.update_yaxes(title_text='Counts')

fig.show()
#fig = px.bar(class_counts, x=class_counts.index, y=class_counts.values, barmode='group')
#data = list()
#for column in class_counts.columns:
#    data.append(go.Bar(column, x=class_counts[column].index, y=class_counts[column].values))

#fig = go.Figure(data)
#fig.update_layout(barmode='group')
#fig.show()
#fig = px.bar(class_counts, title='Counts of target classes per category', barmode='group')
#fig.update_xaxes(title_text='Target Classes')
#fig.update_yaxes(title_text='Counts')
#print(fig.to_json())
#fig.show()
print(class_counts)

