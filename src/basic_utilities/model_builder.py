#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:14:25 2020

@author: otasowie
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from typing import Dict

from basic_utilities import basic_utils

class ModelBuilder:
    
    def build_model_from_grid_searchCV(self, feature_vector_train: pd.DataFrame, 
                          target_train_values: pd.DataFrame) ->  MultiOutputClassifier: 
        """
        

        Parameters
        ----------
        feature_vector_train : pd.DataFrame
            data frame of features
        target_train_values : pd.DataFrame
            dataframe matrix of target classes

        Returns
        -------
        MultiOutputClassifier
            best classification model from gridsearchCV.

        """
        
        results_dict = self.__do_grid_searchCV(feature_vector_train, 
                                               target_train_values)
        
        return results_dict['r_forest']['best_estimator_']
    
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
                    [('text_field_extractor', 
                      basic_utils.TextFieldExtractor('message')), 
                     ('tfidf', 
                      TfidfVectorizer(tokenizer=basic_utils.tokenize, 
                                      min_df=.0025, max_df=0.5, ngram_range=(1,2)))
                     ])),
                 ('numerics', FeatureUnion(
                                       [('text_len', 
                                         Pipeline([('text_len_extractor', 
                                                    basic_utils.NumericFieldExtractor('text_len')), 
                                                    ('text_len_scaler', StandardScaler())
                                                 ])),
                                        ('punt_perc', 
                                         Pipeline([('punt_perc_extractor', 
                                                    basic_utils.NumericFieldExtractor('punt_perc')), 
                                                    ('punt_perc_scaler', StandardScaler())
                                                 ]))
                                       ])),
                 ('starting_verb', basic_utils.PosFieldExtractor('starting_verb_flag'))
                ])),
            ('clf', model)
        ])
        
        return pipeline
    
    def __build_ml_pipeline(self, clf: MultiOutputClassifier) -> Pipeline:
        """
        builds a machine learning pipeline for doing multioutput classification 
        for natural language processing

        Parameters
        ----------
        clf : MultiOutputClassifier
            classifiers for multi labeled classes and several categores

        Returns
        -------
        Pipeline
            pipeline of steps to be followed in training a classifier for nlp.

        """
        pipeline = Pipeline([
            ('features', FeatureUnion(
                [('text', Pipeline(
                    [('text_field_extractor', 
                      basic_utils.TextFieldExtractor('message')), 
                     #('vect', CountVectorizer(tokenizer=clean_text)),
                     #('tfidf', TfidfTransformer())
                 ('tfidf', TfidfVectorizer(tokenizer=basic_utils.tokenize, 
                                           min_df=.0025, max_df=0.25, 
                                               ngram_range=(1,2)))
                 ])),
                    
                 ('numerics', FeatureUnion(
                                   [('text_len', 
                                     Pipeline([('text_len_extractor', 
                                                basic_utils.NumericFieldExtractor('text_len')), 
                                               ('text_len_scaler', StandardScaler())
                                                       ])),
                                    ('punt_perc', 
                                     Pipeline([('punt_perc_extractor', 
                                               basic_utils.NumericFieldExtractor('punt_perc')), 
                                               ('punt_perc_scaler', StandardScaler())
                                                       ]))
                                   ])),
                ('starting_verb', basic_utils.PosFieldExtractor('starting_verb_flag'))
            ])),
            
            ('clf', clf)
        ])
    
        return pipeline

    def __do_grid_searchCV(self, feature_vector_train: pd.DataFrame, 
                          target_train_values: pd.DataFrame) -> Dict: 
        """
        
        Performs grid search cv and returns the best model and parameters
        Parameters
        ----------
        feature_vector_train : pd.DataFrame
            data frame of features
        target_train_values : pd.DataFrame
            dataframe matrix of target classes

        Returns
        -------
        Dict
            Dictionary of best model and parameters from grid search cv.

        """
        r_forest_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        #e_trees_forest_clf = ExtraTreesClassifier(random_state=42, n_jobs=-1)
        clf_list = [r_forest_clf]
        param_grid = {
            #'features__text__tfidf__ngram_range': ((1, 1), (1, 2)),
            'features__text__tfidf__max_df': (0.5, 0.75, 1.0),
            'clf__estimator__n_estimators': [40, 50, 100, 200],
            'clf__estimator__min_samples_split': [2, 3, 4],
            'features__transformer_weights': (
                {'text': 1, 'starting_verb': 0.5},
                {'text': 0.5, 'starting_verb': 1},
                {'text': 0.8, 'starting_verb': 1},
            )
        }
        results_dict = dict()
        model_names = ['r_forest']
        for model_name, clf in zip( model_names, clf_list):
            model = MultiOutputClassifier(clf)
            grid_search = GridSearchCV(self.__build_ml_pipeline(model), 
                                       param_grid, cv=3)
            
            grid_search.fit(feature_vector_train, target_train_values)
            results_dict[model_name] = \
                {'best_params_':grid_search.best_params_, 
                 'best_score_': grid_search.best_score_, 
                 'best_estimator_': grid_search.best_estimator_, 
                 'gridsearch': grid_search}
                    
        return results_dict 