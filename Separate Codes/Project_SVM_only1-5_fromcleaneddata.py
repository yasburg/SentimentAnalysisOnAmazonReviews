# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:00:44 2019

@author: yasin
"""

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
dataset = pd.read_csv('C:/Users/yasin/Desktop/ML dataset/train60_test15_only1-5_cleaned.csv', delimiter = ',') 


reviews_test_clean = []  
reviews_train_clean = []

datatrain = 14504

datatest = 3626

for i in range(0, datatrain):  
    
    review = str(dataset['cleaned_review'][i])
    # append each string to create 
    # array of clean text  
    reviews_train_clean.append(review) 


#print("reviews_train_clean :", reviews_train_clean)

for i in range(datatrain, datatrain+datatest):  
    
    review = str(dataset['cleaned_review'][i])
    # append each string to create 
    # array of clean text  
    reviews_test_clean.append(review) 


#print("reviews_test_clean :", reviews_test_clean)


# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)
# X contains corpus (dependent variable) 
#X = cv.fit_transform(reviews_train_clean).toarray() 

print("\nLogistic Regression:\n")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [0 if i < (datatrain//2)+1 else 1 for i in range(1,datatrain+1)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.80)

#This part is to fint best parameter
#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#    
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %% %s" 
#           % (c, str(("%.3f" % ((accuracy_score(y_val, lr.predict(X_val)))*100)))))
    

newtarget = []
for i in range(datatrain,datatrain+(datatest)):
    if(int(dataset['overall'][i]) <3):
        newtarget.append(0)
    else:
        newtarget.append(1)

final_model = LogisticRegression(C=0.15)
final_model.fit(X, target)
print ("Final Accuracy: %% %s\n"  % str(("%.3f" % ((accuracy_score(newtarget, final_model.predict(X_test)))*100))))
# Final Accuracy: 0.86734

feature_to_coef = {
    word: coef for word, coef in zip(
        cv.get_feature_names(), final_model.coef_[0]
    )
}

print("Positive words:")
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
print("\nNegative words:")
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
    
print("\nLogistic Regression with Bigrams:\n")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.80
)

#This part is to fint best parameter
#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#    
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for %%C=%s: %s" 
#            % (c, str(("%.3f" % ((accuracy_score(y_val, lr.predict(X_val))*100))))))

newtarget = []
for i in range(datatrain,datatrain+(datatest)):
    if(int(dataset['overall'][i]) <3):
        newtarget.append(0)
    else:
        newtarget.append(1)

final_ngram = LogisticRegression(C=0.6)
final_ngram.fit(X, target)
print ("Final Accuracy: %% %s\n" 
       % str(("%.3f" % ((accuracy_score(newtarget, final_ngram.predict(X_test)))*100))))

# Final Accuracy: 
print("\nSVM with Bigrams:\n")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.80)

#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#for c in [0.005,0.015,0.014,0.16]:
#    
#    svm = LinearSVC(C=c)
#    svm.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, str("%.3f" % ((accuracy_score(y_val, svm.predict(X_val)))*100))))
#    


newtarget = []
for i in range(datatrain,datatrain+(datatest)):
    if(int(dataset['overall'][i]) <3):
        newtarget.append(0)
    else:
        newtarget.append(1)

final_svm_ngram = LinearSVC(C=0.015)
final_svm_ngram.fit(X, target)
print ("Final Accuracy: %s" 
       % str("%.3f" % ((accuracy_score(newtarget, final_svm_ngram.predict(X_test)))*100)))

    
elapsed_time = time.time() - start_time
print("time:",elapsed_time)

# Final Accuracy: 0.8974
#########################################################################################
#import numpy as np
#import string
#from scipy.sparse import spmatrix, coo_matrix
#
#from sklearn.base import BaseEstimator
#from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
#from sklearn.svm import LinearSVC
#
#
#__all__ = ['NBSVM']
#
#class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):
#
#    def __init__(self, alpha=1, C=0.15, beta=0.25, fit_intercept=False):
#        self.alpha = alpha
#        self.C = C
#        self.beta = beta
#        self.fit_intercept = fit_intercept
#
#    def fit(self, X, y):
#        self.classes_ = np.unique(y)
#        if len(self.classes_) == 2:
#            coef_, intercept_ = self._fit_binary(X, y)
#            self.coef_ = coef_
#            self.intercept_ = intercept_
#        else:
#            coef_, intercept_ = zip(*[
#                self._fit_binary(X, y == class_)
#                for class_ in self.classes_
#            ])
#            self.coef_ = np.concatenate(coef_)
#            self.intercept_ = np.array(intercept_).flatten()
#        return self
#
#    def _fit_binary(self, X, y):
#        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
#        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
#        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
#        b = np.log((y == 1).sum()) - np.log((y == 0).sum())
#
#        if isinstance(X, spmatrix):
#            indices = np.arange(len(r))
#            r_sparse = coo_matrix(
#                (r, (indices, indices)),
#                shape=(len(r), len(r))
#            )
#            X_scaled = X * r_sparse
#        else:
#            X_scaled = X * r
#
#        lsvc = LinearSVC(
#            C=self.C,
#            fit_intercept=self.fit_intercept,
#            max_iter=10000
#        ).fit(X_scaled, y)
#
#        mean_mag =  np.abs(lsvc.coef_).mean()
#
#        coef_ = (1 - self.beta) * mean_mag * r + \
#                self.beta * (r * lsvc.coef_)
#
#        intercept_ = (1 - self.beta) * mean_mag * b + \
#                     self.beta * lsvc.intercept_
#
#        return coef_, intercept_
#    
#token_pattern = r'\w+|[%s]' % string.punctuation
#ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3),token_pattern=token_pattern)
#ngram_vectorizer.fit(reviews_train_clean)
#X = ngram_vectorizer.transform(reviews_train_clean)
#X_test = ngram_vectorizer.transform(reviews_test_clean)
#
#X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.80)
#
#newtarget = []
#for i in range(datatrain,datatrain+(datatest)):
#    if(int(dataset['overall'][i]) <3):
#        newtarget.append(0)
#    else:
#        newtarget.append(1)
#
#print("Fitting Model")
#import numpy
#numpy.set_printoptions(threshold=numpy.nan)
#mnbsvm = NBSVM()
#mnbsvm.fit(X_train, X_val)
#print('Test Accuracy: %s' % mnbsvm.score(y_train,y_val))

