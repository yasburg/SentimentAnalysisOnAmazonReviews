# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:49:17 2019

@author: yasin
"""
# -*- coding: utf-8 -*-

import time
start_time = time.time()
randomforesttime = time.time()

# Importing Libraries 
import pandas as pd 
import numpy as np 
# Import dataset 
dataset = pd.read_csv('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/BalancedStemmed60k.csv', delimiter = ',') 

reviews_test_clean = []  
reviews_train_clean = []
reviews_scores = []

#these all_... are for the 1-5 train set 1-2-4-5 test set but it didnt work
all_test_clean = []
all_test_scores = []

#ADDING RATING 1 TO TRAIN DATA
for i in range(1, 7253):  
    review = str(dataset['stemmed'][i])
    # append each string to create 
    # array of clean text
    reviews_train_clean.append(review) 
    reviewscore = str(dataset['overall'][i])
    reviews_scores.append(reviewscore)
##ADDING RATING 2 TO TRAIN DATA
#for i in range(12089, 19341):  
#    review = str(dataset['stemmed'][i])
#    # append each string to create 
#    # array of clean text
#    reviews_train_clean.append(review) 
#    reviewscore = str(dataset['overall'][i])
#    reviews_scores.append(reviewscore)
##ADDING RATING 4 TO TRAIN DATA
#for i in range(36265, 43517):  
#    review = str(dataset['stemmed'][i])
#    # append each string to create 
#    # array of clean text
#    reviews_train_clean.append(review) 
#    reviewscore = str(dataset['overall'][i])
#    reviews_scores.append(reviewscore)
#ADDING RATING 5 TO TRAIN DATA
for i in range(48353, 55605):  
    review = str(dataset['stemmed'][i])
    # append each string to create 
    # array of clean text
    reviews_train_clean.append(review) 
    reviewscore = str(dataset['overall'][i])
    reviews_scores.append(reviewscore)

#ADDING RATING 1 TO TEST DATA
for i in range(7253, 9066):  
    review = str(dataset['stemmed'][i])
    # append each string to create 
    # array of clean text  
    reviews_test_clean.append(review) 
    all_test_clean.append(review)
    reviewscore = str(dataset['overall'][i])
    reviews_scores.append(reviewscore)
#    
##ADDING RATING 2 TO TEST DATA
#for i in range(19341, 21154):  
#    review = str(dataset['stemmed'][i])
#    # append each string to create 
#    # array of clean text  
#    reviews_test_clean.append(review) 
#    all_test_clean.append(review)
#    reviewscore = str(dataset['overall'][i])
#    reviews_scores.append(reviewscore)
#    all_test_scores.append(reviewscore) 
#    
##ADDING RATING 3 TO TEST DATA
#for i in range(43517, 45330):  
#    review = str(dataset['stemmed'][i])
#    # append each string to create 
#    # array of clean text  
#    reviews_test_clean.append(review) 
#    all_test_clean.append(review)
#    reviewscore = str(dataset['overall'][i])
#    reviews_scores.append(reviewscore)
#    all_test_scores.append(reviewscore) 
#    
#ADDING RATING 5 TO TEST DATA
for i in range(55605, 57418):  
    review = str(dataset['stemmed'][i])
    # append each string to create 
    # array of clean text  
    reviews_test_clean.append(review) 
    all_test_clean.append(review)
    reviewscore = str(dataset['overall'][i])
    reviews_scores.append(reviewscore)
    all_test_scores.append(reviewscore) 


############################################################################## 
    
numberoftrees = 20
forrandomforest = reviews_train_clean + reviews_test_clean
# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 
# To extract max *** feature. 
# "max_features" is attribute to 
# experiment with to get better results 
cv = CountVectorizer() 
# X contains corpus (dependent variable) 
X = cv.fit_transform(forrandomforest).toarray() 
# y contains answers if review 
# is positive or negative 

y = reviews_scores
#print("y:",y)
# Splitting the dataset into 
# the Training set and Test set 
from sklearn.model_selection import train_test_split 
# experiment with "test_size" 
# to get better results 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25) 
# Fitting Random Forest Classification 
# to the Training set 
from sklearn.ensemble import RandomForestClassifier 
# n_estimators can be said as number of 
# trees, experiment with n_estimators 
# to get better results 
model = RandomForestClassifier(n_estimators = numberoftrees, criterion = 'entropy') 	
model.fit(X_train, y_train) 
# Predicting the Test set results
y_pred = model.predict(X_test)
RFtime = time.time() - randomforesttime
print("Random Forest took", "%.1f" %RFtime,"seconds")
cross_val_time = time.time()
from sklearn.model_selection import cross_val_score
print("-------------------------------------------")
print('With Cross Validation(10):')
scores = cross_val_score(model, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Cross validation took:",time.time()-cross_val_time)
# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix 
np.set_printoptions(threshold=np.inf)
cm = confusion_matrix(y_test, y_pred) 
print("Confusion matrix:")
print(cm)
correctcount = cm[0][0] + cm[1][1]
print("accuracy:",(correctcount/(len(forrandomforest)*0.25))*100)
##############################################################################

logistictime = time.time()
datatrain = 14504
datatest = 3626
# Creating the Bag of Words model 
from sklearn.feature_extraction.text import CountVectorizer 

cv = CountVectorizer(binary=True)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)
# X contains corpus (dependent variable) 
#X = cv.fit_transform(reviews_train_clean).toarray() 
print("-------------------------------------------")
print("Binary Logistic Regression with Unigrams:")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = [0 if i < (datatrain//2)+1 else 1 for i in range(1,datatrain+1)]

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

#This part is to fint best parameter
#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#    
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %% %s" 
#           % (c, str(("%.3f" % ((accuracy_score(y_val, lr.predict(X_val)))*100)))))
    

newtarget = []
for i in range(7253, 9066):
    if(int(dataset['overall'][i]) <3):
        newtarget.append(0)
    else:
        newtarget.append(1)
for i in range(55605, 57418):
    if(int(dataset['overall'][i]) <3):
        newtarget.append(0)
    else:
        newtarget.append(1)
final_model = LogisticRegression(C=0.15)
final_model.fit(X, target)
cross_val_time = time.time()
print('With Cross Validation(10):')
scores = cross_val_score(final_model, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Cross validation took:",time.time()-cross_val_time)
print('Normal Validation:')
print ("Final Accuracy: %% %s"  % str(("%.3f" % ((accuracy_score(newtarget, final_model.predict(X_test)))*100))))
# Final Accuracy: 0.86734
print("-------------------------------------------")
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
print("-------------------------------------------")
print("Binary Logistic Regression with Bigrams:")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)
#This part is to fint best parameter
#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#    lr = LogisticRegression(C=c)
#    lr.fit(X_train, y_train)
#    print ("Accuracy for %%C=%s: %s" 
#            % (c, str(("%.3f" % ((accuracy_score(y_val, lr.predict(X_val))*100))))))

final_ngram = LogisticRegression(C=0.6)
final_ngram.fit(X, target)
cross_val_time = time.time()
print('With Cross Validation(10):')
scores = cross_val_score(final_model, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Cross validation took:",time.time()-cross_val_time)
print('Normal Validation:')
print ("Final Accuracy: %% %s" % str(("%.3f" % ((accuracy_score(newtarget, final_ngram.predict(X_test)))*100))))
logisticctime = time.time() - logistictime
print("Binary Logistic took", "%.1f" %logisticctime,"seconds")
# Final Accuracy: 
##############################################################################
svmutime = time.time()
print("-------------------------------------------")
print("SVM with Unigrams:")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True)
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

final_svm_ngram = LinearSVC(C=0.015)
final_svm_ngram.fit(X, target)
cross_val_time = time.time()
print('With Cross Validation(10):')
scores = cross_val_score(final_svm_ngram, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Cross validation took:",time.time()-cross_val_time)
print('Normal Validation:')
print ("Final Accuracy: %s" % str("%.3f" % ((accuracy_score(newtarget, final_svm_ngram.predict(X_test)))*100)))
svmu = time.time() - svmutime
print("SVM unigram took", "%.1f" %svmu,"seconds")

##############################################################################
svmtime = time.time()
print("-------------------------------------------")
print("SVM with Bigrams:")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2))
ngram_vectorizer.fit(reviews_train_clean)
X = ngram_vectorizer.transform(reviews_train_clean)
X_test = ngram_vectorizer.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(X, target, train_size = 0.75)

#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#for c in [0.01,0.05,0.1,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95, 1]:
#    svm = LinearSVC(C=c)
#    svm.fit(X_train, y_train)
#    print ("Accuracy for C=%s: %s" 
#           % (c, str("%.3f" % ((accuracy_score(y_val, svm.predict(X_val)))*100))))
    
final_svm_ngram = LinearSVC(C=0.015)
final_svm_ngram.fit(X, target)
cross_val_time = time.time()
print('With Cross Validation(10):')
scores = cross_val_score(final_svm_ngram, X_train, y_train, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Cross validation took:",time.time()-cross_val_time)
print('Normal Validation:')
print ("Final Accuracy: %s" % str("%.3f" % ((accuracy_score(newtarget, final_svm_ngram.predict(X_test)))*100)))
svmm = time.time() - svmtime
print("SVM bigram took", "%.1f" %svmm,"seconds")
print("-------------------------------------------")
# Final Accuracy: 0.8974
##############################################################################

nbsvmm = time.time()
from scipy.sparse import spmatrix, coo_matrix

from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC

__all__ = ['NBSVM']

class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self, alpha=1, C=1, beta=0.25, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=10000
        ).fit(X_scaled, y)

        mean_mag =  np.abs(lsvc.coef_).mean()

        coef_ = (1 - self.beta) * mean_mag * r + \
                self.beta * (r * lsvc.coef_)

        intercept_ = (1 - self.beta) * mean_mag * b + \
                     self.beta * lsvc.intercept_

        return coef_, intercept_
import glob
import os
import string

from sklearn.feature_extraction.text import CountVectorizer



def load_imdb():
    print("Vectorizing Training Text")

    train_pos = glob.glob(os.path.join('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/train/pos', '*.txt'))
    train_neg = glob.glob(os.path.join('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/train/neg', '*.txt'))

    token_pattern = r'\w+|[%s]' % string.punctuation

    vectorizer = CountVectorizer('filename', ngram_range=(1, 3),
                                 token_pattern=token_pattern,
                                 binary=True)
    X_train = vectorizer.fit_transform(train_pos+train_neg)
    y_train = np.array([1]*len(train_pos)+[0]*len(train_neg))

    print("Vocabulary Size: %s" % len(vectorizer.vocabulary_))
    print("Vectorizing Testing Text")

    test_pos = glob.glob(os.path.join('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/test/pos', '*.txt'))
    test_neg = glob.glob(os.path.join('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/test/neg', '*.txt'))

    X_test = vectorizer.transform(test_pos + test_neg)
    y_test = np.array([1]*len(test_pos)+[0]*len(test_neg))

    return X_train, y_train, X_test, y_test

def main():

    X_train, y_train, X_test, y_test = load_imdb()

    print("Fitting Model")

    mnbsvm = NBSVM()
    mnbsvm.fit(X_train, y_train)
    nbsvmm = time.time() - svmtime
    print("NBSVM took", "%.1f" %nbsvmm,"seconds")
    cross_val_time = time.time()
    print('With Cross Validation(10):')
    scores = cross_val_score(mnbsvm, X_train, y_train, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Cross validation took:",time.time()-cross_val_time)
    print('Normal Validation:')
    print('Test Accuracy: %s' % mnbsvm.score(X_test, y_test))

if __name__ == '__main__':
    main()
    
elapsed_time = time.time() - start_time
print("total time:","%.1f" %elapsed_time,"seconds")
