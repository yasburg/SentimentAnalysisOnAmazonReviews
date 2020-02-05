# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 22:00:06 2019

@author: yasin
"""
import time
start_time = time.time()

# Importing Libraries 
import pandas as pd 

# Import dataset 
dataset = pd.read_csv('C:/Users/yasin/Desktop/ML dataset/train60_test15.csv', delimiter = ',') 

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

datatrain = 29008

datatest = 7252

for i in range(0, datatrain):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['summary'][i]))  
      
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


print("reviews_train_clean :", reviews_train_clean)

for i in range(datatrain, datatrain+datatest):  
      
    # column : "Review", row ith 
    review = re.sub('[^a-zA-Z]', ' ', str(dataset['summary'][i]))  
      
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


print("reviews_test_clean :", reviews_test_clean)


# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)
# X contains corpus (dependent variable) 
#X = cv.fit_transform(reviews_train_clean).toarray() 

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [0 if i < (datatrain//2)+1 else 1 for i in range(1,datatrain+1)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.80)

for c in [0.01, 0.05, 0.25, 0.5, 0.60, 0.65, 0.75, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, lr.predict(X_val))))
    

    
newtarget = []
for i in range(datatrain,datatrain+(datatest)):
    if(int(dataset['overall'][i]) <3):
        newtarget.append(0)
    else:
        newtarget.append(1)

final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
print ("Final Accuracy: %s" % accuracy_score(newtarget, final_model.predict(X_test)))
# Final Accuracy: 0.86734

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
elapsed_time = time.time() - start_time
print("time:",elapsed_time)
