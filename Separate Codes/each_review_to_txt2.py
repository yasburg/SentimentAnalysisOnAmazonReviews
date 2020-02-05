# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:31:17 2019

@author: yasin
"""
import time
start_time = time.time()

# Import dataset 
import csv
mycsv = csv.reader(open('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/BalancedStemmed60k.csv'))
import pandas as pd 

# Import dataset 
dataset = pd.read_csv('C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/BalancedStemmed60k.csv', delimiter = ',') 

negfilelocation = "C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/train/neg/"
posfilelocation = "C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/train/pos/"

for i in range(0, 9066):  
    txtname = str(i) + "_" + str(dataset['overall'][i])
    f= open(negfilelocation + txtname +".txt","w+")
    f.write(str(dataset['stemmed'][i]))
    f.close()
    
for i in range(48352, 57418): 
    txtname = str(i-48352) + "_" + str(dataset['overall'][i])
    f= open(posfilelocation + txtname +".txt","w+")
    f.write(str(dataset['stemmed'][i]))
    f.close()
    
negfilelocation = "C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/test/neg/"
posfilelocation = "C:/Users/yasin/Desktop/Machine Learning/Project/Dataset/test/pos/"

for i in range(9066, 12088):  
    txtname = str(i-9066) + "_" + str(dataset['overall'][i])
    f= open(negfilelocation + txtname +".txt","w+")
    f.write(str(dataset['stemmed'][i]))
    f.close()
    
for i in range(57418, 60440):  
    txtname = str(i-57418) + "_" + str(dataset['overall'][i])
    f= open(posfilelocation + txtname +".txt","w+")
    f.write(str(dataset['stemmed'][i]))
    f.close()
    
elapsed_time = time.time() - start_time
print("time:",elapsed_time)

#rowidcounter = 0
#counter = 1
#for row in mycsv:
#    if row[7] == 'overall':
#        #skip first row
#        counter = counter+1
#        rowidcounter +=1
#    elif(int(row[7])==1):
#        txtname = str(rowidcounter) + "_" + str(row[7])
#        f= open(negfilelocation + txtname +".txt","w+")
#        f.write(str(row[8]))
#        f.close()
#        rowidcounter +=1
#    elif(int(row[7])==5):
#        txtname = str(rowidcounter) + "_" + str(row[7]) 
#        f= open(posfilelocation + txtname +".txt","w+")
#        f.write(str(row[8]))
#        f.close()
#        rowidcounter +=1
    

#
#from openpyxl import *
#import os    
#
#p = 'C:/Users/yasin/Desktop/ML dataset/balancedData.csv'
#files = [_ for _ in os.listdir(p) if _.endswith('.xlsx')]
#
#for f in files:
#
#     wb = load_workbook(os.path.join(p, f))
#     ws = wb['name_of_sheet']
#     for row in ws.rows:
#         with open(row[2].value+'.txt', 'w') as outfile:
#              outfile.write(row[0].value)