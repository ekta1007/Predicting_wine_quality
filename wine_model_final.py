# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd , random 
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from datetime import datetime 
from sklearn.metrics import classification_report
import math
import matplotlib.pyplot as plt
import statsmodels.api as st
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import pylab as pl

"""
@author : Ekta Grover, ekta1007@gmail.com
@ Last Modified : 29th Jan, 2014
"""

# Tried ROC averaging from this paper, but it did not fine tune for my use case (time constraints) 
#A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classiﬁcation Problems -DAVID J. HAND & ROBERT J. TILL

#Validation Scheme :  instead of trying k fold cross validation, I ran the model several times over with straight-randomization - the primary reason was k-fold validation from sklearn
#was very black-boxish and randomizing over passes pretty straight-forward. In the long run, I did see the numbers as reported below.
# Parts of code based on documentation/samples from sklearn and StackOverflow
# I also tried GradientBoostingRegressor, but the performance/fit wasn't as good, for the simple reason that it had many more parameters in there


"""
Inputs and Usage
file_handle - change this over wrt White/Red Wine(binned data with long tails) & files with outliers removed completely
sampling_ratio is a number between 0 and 1, for example 0.7 means 70%
color defines what data-model we used (Red/White wine)

"""

startTime = datetime.now()
#with optional arguments - We can either use reservoir sampling, or use the "random" and selecting indexed rows as in ReadAndPrepare_DataFrame belo


# Also tried transforming/scaling the data in equi-ranges [0,1],only improve performance marginally
def transform(x):
    new_x=(x-min(x))/(max(x)-min(x))
    return new_x

def ReadAndPrepare_DataFrame(sampling_ratio,*args, **kwargs):
    file_handle_csv = kwargs.get('file_handle_csv', None)
    df = kwargs.get('df', None)
     
    #(file_handle, df_returned, sampling_ratio): 
    if file_handle_csv:
        df = pd.read_csv(file_handle_csv,skiprows=0, sep=',')
    """
    #Transforming the data- there were only marginal gains in this approach - so skiping this
    for i in header[0:11] :
        df[i] = df[i].astype(float)
        df[i]=transform(df[i])
    """
    header=list(df.columns.values) # or df.columns
    X = df[df.columns - [header[-1]]] # header[-1] = ['quality'] - this is to make the code genric enough
    Y = df[header[-1]] # df['quality']
    rows = random.sample(df.index, int(len(df)*sampling_ratio)) # indexing the rows that will be picked in the train set
    x_train, y_train = X.ix[rows],Y.ix[rows] # Fetching the data frame using indexes
    x_test,y_test  = X.drop(rows),Y.drop(rows)
    return df,header,x_train, y_train, x_test,y_test

def ReadAndPrepare_from_ReservoirSampledData(df,sampling_ratio):
    # df is already shuffled at this point - so we just read in
    header=list(df.columns)
    x_data, y_data = df[df.columns - [header[-1]]], df[header[-1]]
    return x_data, y_data # can re-use this as test & train with different sampling ratios
    

def LinearSVC_custom(header,x_train, y_train,x_test,y_test,color):
    clf = LinearSVC(C=1, penalty="l1", dual=False) #,verbose=1) ,tol=0.0001,fit_intercept=True, intercept_scaling=1)
    clf.fit(x_train.values, y_train.values)
    x_tranformed= clf.fit_transform(x_train.values, y_train.values) # transformed X to its most important features
    clf.predict(x_test.values)
    print "Goodness of fit using the LinearSVC is %f \n \n  " %clf.score(x_test.values, y_test.values) # Goodness of fit
    #clf.coef_  # estimate set of coeffs - This will actually store the coeffs as "0" for the vars we wont be using, so it does the trick of fetching the corresponding indices
    important_features=[]
    m=clf.coef_[0]
    index=0
    for i in m:
        if i == 0:
            pass
        else : #not zero, meaning this atribute defines the transformed dataset from the orignal linear combination data-set
            important_features.append(index)
        index=index+1
    features=[header[i] for i in important_features]
   # returning the set of important features with the corresponding "model" (color of wine is the model)
    print "The important features for %s color are : %s  \n \n " %(color, str(features).replace("'",'').replace("[",'').replace("]",''))
    return features


def train_UsingExtraTreesClassifier(df,header,x_train, y_train,x_test,y_test) :

    # training
    clf = ExtraTreesClassifier(n_estimators=200,random_state=0,criterion='gini',bootstrap=True,oob_score=1,compute_importances=True)
    # Also tried entropy for the information gain but 'gini' seemed to give marginally better fit, bith in sample & out of sample
    clf.fit(x_train, y_train)
    #estimation of goodness of fit
    print "Estimation of goodness of fit using the ExtraTreesClassifier is : %f  \n" % clf.score(x_test,y_test)
    print "Estimation of out of bag score  using the ExtraTreesClassifier is : %f \n \n  " % clf.oob_score_
    # getting paramters back, if needed
    clf.get_params()
    # get the vector of predicted prob back
    y_test_predicted= clf.predict(x_test)
    X = df[df.columns - [header[-1]]]

    feature_importance = clf.feature_importances_
    # On a scale of 10 - make importances relative to max importance and plot them
    feature_importance = 10.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance) #Returns the indices that would sort an array.
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    return y_test_predicted

def regress_MNLogit(df,header,x_train,y_train,x_test,y_test,color):
    # df.describe()
    #y_train=[int(i) for i in y_train]
    m_logit = st.MNLogit(y_train, x_train)
    m_logit_fit = m_logit.fit()
    print m_logit_fit.summary()
    # get params back
    #print np.exp(result.params)
    # odds ratios and 95% CI
    params = m_logit_fit.params
    #marginal effects
    m_logit_margeff = m_logit_fit.get_margeff()
    print " Marginal effects of model with all variables \n \n "
    print m_logit_margeff.summary()
    print "AIC of the preliminary model trained using MNLlogit is %f \n  " %m_logit_fit.aic 
    print "BIC of the preliminary model trained using MNLlogit is %f \n " %m_logit_fit.bic 
    # RED
    if color=='red':
        new_cols = ['volatile acidity',  'residual sugar', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol']
    #WHITE
    elif color=='white':
        new_cols = ['volatile acidity',  'residual sugar', 'free sulfur dioxide', 'fixed acidity', 'density', 'sulphates', 'alcohol']
    data = df[new_cols]
    # new fitted
    m_logit_pass1 = st.MNLogit(y_train, x_train[new_cols])
    m_logit_fit_pass1 = m_logit_pass1.fit()
    print m_logit_fit_pass1.summary()
    m_logit_margeff_pass1 = m_logit_fit_pass1.get_margeff()
    print " Marginal effects of reduced model \n \n  "
    print m_logit_margeff_pass1.summary()
    print " Odds Ratio of reduced model \n \n  "
    print np.exp(m_logit_fit.params) # odds ratios 
    print "AIC of the reduced model is %f " %m_logit_fit_pass1.aic 
    print "BIC of the reduced model is %f " %m_logit_fit_pass1.bic
    # fit & predict the results back
    return m_logit_fit_pass1,data

def metrics(df,header,y_test,y_test_predicted):
    labels =list(np.sort(list(set(df[header[-1]])))) # Recall that df[header[-1] is indeed df['quality']
    target_names=['class'+str(i) for i in labels]
    print " Printing Classification report \n \n "
    print classification_report(y_test,y_test_predicted, target_names=target_names)
    cm = confusion_matrix(y_test, y_test_predicted,labels)
    print(cm)
    # Plot confusion matrix 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    pl.title('Confusion matrix of the classifier [Numbers represent Wine Quality labels] \n')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    pl.xlabel('Predicted Wine Quality ')
    pl.ylabel('True Wine Quality ')
    pl.show()

def reservoir_sampling(file_handle,sampling_ratio):
    # To reproduce the results - use random.seed(12345) , or any number
    # reservoir_sampling is modified from Wiki
    sample_lines = []
    data=[]
    with open(file_handle, 'r') as source:
        header=source.readline()
        header=header.strip()
        ncols=header.count(',')+1
        nrows=0
        for line in source :
            data.append(line.replace('\n','').split(',')[0:ncols]) # can use [0:header.count(',')+1] to abstact the # of column fields, here [0:13]
            nrows=nrows+1
        sample_count=nrows*sampling_ratio
    # Generate the reservoir
    for index, line in enumerate(data):
        if index < sample_count:
            sample_lines.append(line)
        else:                  
            # Randomly replace elements in the reservoir with a decreasing probability         
            # Choose an integer between 0 and index (inclusive)               
            r = random.randint(0, index)               
            if r < sample_count:                       
                sample_lines[r] = line
    df1=pd.DataFrame(sample_lines)
    df1.columns=header.split(',')
    return df1

def plot_ROC(df,y_test_predicted,y_test,labels,color):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predicted,pos_label=labels)
    roc_auc = auc(fpr, tpr)
    print("Area under the ROC curve for color %s , label %d : %f" % (color,labels,roc_auc))
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.0])
    pl.ylim([0.0, 1.0])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC curve for %s Wine Quality label %d ' %(color,labels))
    pl.legend(loc="lower right")
    pl.show()



                        
# Finally, Initializing all functions
#RED/White - replace color & run this code for each of the variations
color = 'white' # 'white'/Red
sampling_ratio=0.7
if color=='red':
    file_handle="D:/Desktop/AdNear/wine_red_binned2.txt"
    file_handle_csv="D:/Desktop/AdNear/wine_red_binned2.csv"
elif color=='white':
    file_handle="D:/Desktop/AdNear/wine_white_binned2.txt"
    file_handle_csv="D:/Desktop/AdNear/wine_white_binned2.csv"
print "Initial parameters \n "
print " Color : %s , Sampling Ratio  : %f , File handle : %s, Corresponding csv File handle : %s \n \n  " %(color,sampling_ratio,file_handle,file_handle_csv) # Init



# READING DATA SETS & SAMPLING

# TECHNIQUE - RESERVOIR SAMPLING
df_train=reservoir_sampling(file_handle,sampling_ratio) # Alternate sampling with replacement as opposed to ReadAndPrepare_DataFrame which is without replacement
df_test=reservoir_sampling(file_handle,(1-sampling_ratio))
x_train, y_train= ReadAndPrepare_from_ReservoirSampledData(df_train,sampling_ratio) #train 
x_test, y_test = ReadAndPrepare_from_ReservoirSampledData(df_test,(1-sampling_ratio)) # test - hence with 1-sampling ratio - using full data-set
header=list(df_train.columns) # using df_train or df_test, really doesnt matter - all we want is column names
#print type(x_train), type(y_train),type(x_test), type(y_test)

#TECHNIQUE - REGULAR RANDOM NUMBER SAMPLING
df_returned = None # to use the regular sampling instead of Reservoir sampling
df,header,x_train, y_train, x_test,y_test=ReadAndPrepare_DataFrame(0.7,file_handle_csv=file_handle_csv,df=df_returned)

# Classification Methods , Metrics & plots 
features=LinearSVC_custom(header,x_train, y_train,x_test,y_test,color)
y_test_predicted=train_UsingExtraTreesClassifier(df,header,x_train, y_train,x_test,y_test)
result,data=regress_MNLogit(df,header,x_train,y_train,x_test,y_test,color) 
metrics(df,header,y_test,y_test_predicted)

# plot ROC curves per label, per wine color
labels =list(np.sort(list(set(df['quality']))))
for i in labels:
    plot_ROC(df,y_test_predicted,y_test,i,color)

print" Total time taken is "
print (datetime.now()-startTime)
