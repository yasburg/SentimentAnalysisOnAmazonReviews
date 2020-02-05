# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 18:47:33 2019

@author: yasin
"""

with open("C:/Users/yasin/Desktop/ML dataset/reviews_Movies_and_TV_5.csv", "rb") as source:
    lines = [line for line in source]

import random
random_choice = random.sample(lines, 200000)

with open("C:/Users/yasin/Desktop/ML dataset/Project_random_200000.csv", "wb") as sink:
    sink.write(b"".join(random_choice))