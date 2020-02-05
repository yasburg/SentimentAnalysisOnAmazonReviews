# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:00:34 2019

@author: yasin
"""
import pandas as pd
import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('C:/Users/yasin/Desktop/ML dataset/reviews_Movies_and_TV_5.json.gz')
df.to_csv(r'C:/Users/yasin/Desktop/ML dataset/reviews_Movies_and_TV_5.csv')