#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:14:25 2020

@author: otasowie
"""

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler

from basic_utilities import basic_utils

class ModelBuilder:
    def build_model(self) -> Pipeline:
        """
        creates a piline for natural language processing based on tfidf
        
        Returns
        -------
        Pipeline
            pipline contining the steps for training a classifier for nlp.
        
        """
        clf = RandomForestClassifier(
                                     n_estimators=200,
                                     max_features='auto',
                                     min_samples_leaf=1,
                                     min_samples_split=3,
                                     random_state=42, 
                                     n_jobs=-1)
        model = MultiOutputClassifier(clf)
        
        pipeline = Pipeline([
            ('features', FeatureUnion(
                [('text', Pipeline(
                    [('text_field_extractor', basic_utils.TextFieldExtractor('message')), 
                     ('tfidf', TfidfVectorizer(tokenizer=basic_utils.tokenize, min_df=.0025, max_df=0.5, ngram_range=(1,2)))
                     ])),
                 ('numerics', FeatureUnion(
                                       [('text_len', Pipeline([('text_len_extractor', basic_utils.NumericFieldExtractor('text_len')), 
                                                           ('text_len_scaler', StandardScaler())
                                                           ])),
                                        ('punt_perc', Pipeline([('punt_perc_extractor', basic_utils.NumericFieldExtractor('punt_perc')), 
                                                           ('punt_perc_scaler', StandardScaler())
                                                           ]))
                                       ])),
                 ('starting_verb', basic_utils.PosFieldExtractor('starting_verb_flag'))
                ])),
            ('clf', model)
        ])
        
        return pipeline