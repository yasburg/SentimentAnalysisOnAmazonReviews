# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:53:02 2019

@author: yasin
"""

import time
start_time = time.time()

import numpy as np

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

    train_pos = glob.glob(os.path.join('C:/Users/yasin/Desktop/ML dataset/train/pos', '*.txt'))
    train_neg = glob.glob(os.path.join('C:/Users/yasin/Desktop/ML dataset/train/neg', '*.txt'))

    token_pattern = r'\w+|[%s]' % string.punctuation

    vectorizer = CountVectorizer('filename', ngram_range=(1, 3),
                                 token_pattern=token_pattern,
                                 binary=True)
    X_train = vectorizer.fit_transform(train_pos+train_neg)
    y_train = np.array([1]*len(train_pos)+[0]*len(train_neg))

    print("Vocabulary Size: %s" % len(vectorizer.vocabulary_))
    print("Vectorizing Testing Text")

    test_pos = glob.glob(os.path.join('C:/Users/yasin/Desktop/ML dataset/test/pos', '*.txt'))
    test_neg = glob.glob(os.path.join('C:/Users/yasin/Desktop/ML dataset/test/neg', '*.txt'))

    X_test = vectorizer.transform(test_pos + test_neg)
    y_test = np.array([1]*len(test_pos)+[0]*len(test_neg))

    return X_train, y_train, X_test, y_test

def main():

    X_train, y_train, X_test, y_test = load_imdb()

    print("Fitting Model")

    mnbsvm = NBSVM()
    mnbsvm.fit(X_train, y_train)
    print('Test Accuracy: %s' % mnbsvm.score(X_test, y_test))

if __name__ == '__main__':
    main()

elapsed_time = time.time() - start_time
print("time:",elapsed_time)


