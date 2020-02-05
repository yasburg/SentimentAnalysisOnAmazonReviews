# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:06:40 2019

@author: yasin
"""

with open("C:/Users/yasin/Desktop/ML dataset/balancedData.csv", "rb") as source:
    lines = [line for line in source]
    
#rating 1
with open("C:/Users/yasin/Desktop/ML dataset/train60_test15.csv", "wb") as sink:
    for i in range(1,7253):        
        sink.write(lines[i])
        
#rating 2 
    for i in range(12089,19341):        
        sink.write(lines[i])

#rating 4 (24176,31428)
    for i in range(24177,31429):        
        sink.write(lines[i])
        
#rating 5 (31428,38680)
    for i in range(36265,43517):        
        sink.write(lines[i])
        
#rating 1 test (7252,9065)
    for i in range(7253,9066):        
        sink.write(lines[i])
        
#rating 2 test (19340,21153)
    for i in range(19341,21154) :        
        sink.write(lines[i])
        
#rating 4 test (31428,33241)
    for i in range(31429,33242):        
        sink.write(lines[i])
        
#rating 5 test (38680,40493)
    for i in range(43517,45330):        
        sink.write(lines[i])

