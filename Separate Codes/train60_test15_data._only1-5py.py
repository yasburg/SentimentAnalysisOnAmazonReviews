# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 12:59:17 2019

@author: yasin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 23:06:40 2019

@author: yasin
"""

import time
start_time = time.time()


with open("C:/Users/yasin/Desktop/ML dataset/balancedData.csv", "rb") as source:
    lines = [line for line in source]
    
#rating 1
with open("C:/Users/yasin/Desktop/ML dataset/train60_test15_only1-5.csv", "wb") as sink:
    for i in range(1,7253):        
        sink.write(lines[i])
        
#rating 5 (31428,38680)
    for i in range(48353,55605):        
        sink.write(lines[i])
        
#rating 1 test (7252,9065)
    for i in range(7253,9066):        
        sink.write(lines[i])
        
#rating 5 test (38680,40493)
    for i in range(55605,57418):        
        sink.write(lines[i])

elapsed_time = time.time() - start_time
print("time:",elapsed_time)
