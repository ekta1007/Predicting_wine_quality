#knnClassifier.py
import random, csv
import numpy #as np
from sklearn import neighbors
import itertools

"""
Key observation - data is skewed with this distribution
quality	count(1)
6	2836
5	2138
7	1079
4	216
8	193
3	30
9	5
"""

# define custom_metric: sum of squared distance from true vs predicted label.
# Idea( format True label /Predicted label) 9 /10 is different from 7/4 (with former being close estimation than the latter)
def dist_metric(true,predicted):
    diff_score=0
    #for i in range(0,len(true)):
    diff_score=[(int(true[i])-int(predicted[i]))**2 for i in range(0,len(true))]
    diff_score=float(sum(diff_score))/float(len(true))
    return diff_score

## main allogithm - sample creation etc.
data=[]
with open("D:/Desktop/wine_combined.csv", "r") as source:  
    header=source.readline() 
    for line in source :
        data.append(line.replace('\n','').split(',')[0:13]) # can use [0:header.count(',')+1] to abstact the # of column fields, here [0:13]

# partioning the datset into train & test
numpy.random.shuffle(data) # to enable random sampling later
train_sample_len=int(len(data)*0.7) # 70% train sample
training, test = data[0:train_sample_len], data[train_sample_len:]
train_data=[training[i][0:12] for i in range(0,train_sample_len)]  
train_label=[training[i][12:] for i in range(0,train_sample_len)]

test_data=[test[i][0:12] for i in range(0,(len(data)-train_sample_len))] # test data
test_label_true=[test[i][12] for i in range(0,(len(data)-train_sample_len))] # true labels of test


# run the knn classifier now
knn = neighbors.KNeighborsClassifier()
knn=neighbors.KNeighborsClassifier(n_neighbors=23,algorithm='auto',weights='distance',p=12)
knn.fit(train_data, train_label)
test_label_predicted=knn.predict(test_data)
knn.predict_proba(test_data)
knn.score(test_data,test_label_true) # 0.55179487179487174 % mean accuracy on the given test data and true labels
#knn.score(test_data,test_label_predicted) , will give 100% as expected !
# custom metric 
diff_score=dist_metric(test_label_true,test_label_predicted) #0.6887179487179487
