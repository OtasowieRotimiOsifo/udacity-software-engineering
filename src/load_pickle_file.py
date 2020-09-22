#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 19:57:11 2020

@author: otasowie
"""

import sys
import os
import joblib

cwd = os.getcwd()
sys.path.append(cwd)

from basic_utilities import basic_utils

pickle_file_path = cwd + "/classifier1.pkl"
model = joblib.load(pickle_file_path)
print(model)
