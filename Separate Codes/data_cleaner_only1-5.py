# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 14:10:08 2019

@author: yasin
"""


import time
start_time = time.time()


# Importing Libraries 
import pandas as pd 

# Import dataset 
dataset = pd.read_csv('C:/Users/yasin/Desktop/ML dataset/train60_test15_only1-5.csv', delimiter = ',') 

# library to clean data 
import re  
#if needed
#nltk.download('stopwords') 
#all part is for tokenizing
#nltk.download('all')
# to remove stopword 
from nltk.corpus import stopwords 
  
# for Stemming propose  
from nltk.stem.porter import PorterStemmer 
  
reviews_test_clean = []  
reviews_train_clean = []

datatrain = 14504

datatest = 3626

with open("C:/Users/yasin/Desktop/ML dataset/train60_test15_only1-5.csv", "rb") as source:
    lines = [line for line in source]
    
with open("C:/Users/yasin/Desktop/ML dataset/train60_test15_only1-5_cleaned.csv", "wb") as sink:
    
    for i in range(0, datatrain):  
          
        # column : "Review", row ith 
        review = re.sub('[^a-zA-Z]', ' ', str(dataset['reviewText'][i]))  
          
        # convert all cases to lower cases 
        review = review.lower()  
          
        # split to array(default delimiter is " ") 
        review = review.split()  
          
        # creating PorterStemmer object to 
        # take main stem of each word 
        ps = PorterStemmer()  
          
        # loop for stemming each word 
        # in string array at ith row     
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  
                      
        # rejoin all string array elements 
        # to create back into a string 
        review = ' '.join(review)   
        
        # append each string to create 
        # array of clean text  
        reviews_train_clean.append(review)  
        sink.write(str.encode(review+','+str(dataset['overall'][i])+"\n"))
    
    
    print("reviews_train_clean :", reviews_train_clean)
    
    for i in range(datatrain, datatrain+datatest):  
          
        # column : "Review", row ith 
        review = re.sub('[^a-zA-Z]', ' ', str(dataset['reviewText'][i]))  
          
        # convert all cases to lower cases 
        review = review.lower()  
          
        # split to array(default delimiter is " ") 
        review = review.split()  
          
        # creating PorterStemmer object to 
        # take main stem of each word 
        ps = PorterStemmer()  
          
        # loop for stemming each word 
        # in string array at ith row     
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  
                      
        # rejoin all string array elements 
        # to create back into a string 
        review = ' '.join(review)   
        
        # append each string to create 
        # array of clean text  
        reviews_test_clean.append(review) 
        sink.write(str.encode(review+','+str(dataset['overall'][i])+"\n"))
    
    
    print("reviews_test_clean :", reviews_test_clean)

elapsed_time = time.time() - start_time
print("time:",elapsed_time)