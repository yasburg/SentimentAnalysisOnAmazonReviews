# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:46:32 2019

@author: yasin
"""

import gzip

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

ratings = []

for review in parse("C:/Users/yasin/Desktop/ML dataset/reviews_Movies_and_TV_5.json.gz"):
  ratings.append(review['overall'])

print (sum(ratings) / len(ratings))