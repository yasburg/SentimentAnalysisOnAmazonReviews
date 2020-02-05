# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:51:11 2019

@author: yasin
"""

import pandas as pd

df = pd.read_csv(open('C:/Users/yasin/Desktop/ML dataset/Project_random_200000.csv','rU'), encoding='utf-8', engine='c')
print(df['overall'].mean())
