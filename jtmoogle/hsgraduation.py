##
## This is prepared for Udacity Machine Learning Engineer Nanodegree online class
## Author: jtmoogle @github.com All Rights Reserved
## Date:Feb-Apr 2018
## 
## This file contains APIs for HSGraduation 
## 1. load_gradcensus - load jtmoogle/data/GRADUATION_WITH_CENSUS.csv
## 2. Classification
##    (1) plot_cls_gradcensus - illustrate classification graduation census data
##    (2) preproc_cls_data - prepare classification data cleaning 
##    (3) cls_feature_sel - classification feature selection
##    (4) compare_cls_featranking - compared classification feature ranking
##    (5) cls_stats - classification statistic summary
##    (6) create_cls_sample - create sample for Training and Testing datasets
##    (7) cls_acc_featimportance - calculated F score, accurancey, feature importance
##    (8) cls_visual_benchmark - benchmark result and visualization
##    (9) handy functions: cls_ftest - F test result for classification
##        cls_pca - PCA result for classification
## 3. Regression
##    (1) plot_rgs_gradcensus - illustrate regression graduation census data
##    (2) preproc_rgs_data - prepare Regression data cleaning
##    (3) rgs_feature_sel - regression feature selection
##    (4) compare_rgs_featranking - compared regression feature ranking
##    (5) rgs_stats - regression statistic summary
##    (6) create_rgs_sample - create sample for Training and Testing datasets
##    (7) rgs_visual_benchmark - benchmark result and visualization
##    (8) handy functions: rgs_r2 - r2 score for regression
## 
## contact info: jtmoogle@gmail.com for question, or suggestion
## 

import os.path
os.chdir('I:/_githup/capstone-report/')  # this source is at jtmoogle

#os.chdir('/_githup/capstone/')  # this source is at jtmoogle
#print("Currnet wdir={}   ".format(os.getcwd()))

from IPython.display import display
from collections import Counter
from jtmoogle.helper import MyHelper
from matplotlib import pyplot as plt
import numpy as np       
from scipy import stats
from sklearn import metrics
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor)
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.linear_model import (LinearRegression, LogisticRegression, RidgeClassifier, Perceptron, Ridge, Lasso, RandomizedLasso)
#from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm.classes import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.extmath import density
from time import time
import json, codecs
import os.path
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore", category=Warning)


class HSGraduation(object):
    """ Class of 2011-2012 High School Graduation Rate and 2010 Census
    """
    def __init__(self, datapath='jtmoogle/data/GRADUATION_WITH_CENSUS.csv',
                 cols_todrop = ['Unnamed: 0', 'Unnamed: 0.1', 'leaid11', 'leanm11', 'FIPST'],
                 dbx = False) :
        # default constant values
        self.random_state = 99
        self.n_splits = 5   # for kFold self.n_splits
        self.test_size = 0.3 # for sample split
        self.n_estimators = 10 
        self.n_neighbors = 3

        self.datapath = datapath
        self.cols_todrop = cols_todrop
        self.rawdata = {}
        self.selcol_regex = 'Inc|INC|_COHORT_|pct_|avg_|_House_|_AREA_|ALL_|Success' 
        # criteria to drop columns
        self.dropcol_regex = 'ALL_COHORT_1112|MOE_|_FRMS_|Mail|Percentage|County|State|Tract|District|GIDTR|Tract|Flag|Response|Delete|Vacant|BILQ|Diff|Leave|Plumb'
        self.cls_target_col = 'Success_Pass_90'  # target for classification
        self.cls_cmpl_ds={}        # data for learn & test models
        self.cls_feature = {}
        self.cls_feature_cols ={}
        self.cls_feature_data = {}
        self.cls_target = {}
        self.cls_target_data = {}
        self.cls_X_test = {}
        self.cls_X_train = {}
        self.cls_y_test = {}
        self.cls_y_train = {}
        self.rgs_target_col = 'ALL_RATE_1112'    # target for regression
        self.rgs_cmpl_ds={}       # data for learn & test models
        self.rgs_feature = {}
        self.rgs_feature_cols ={}
        self.rgs_feature_data = {}
        self.rgs_target = {}
        self.rgs_target_data = {}
        self.rgs_X_test = {}
        self.rgs_X_train = {}
        self.rgs_y_test = {}
        self.rgs_y_train = {}
        self.dbx =dbx
    
    #1----- - Load dataset, and report data size
    def load_gradcensus(self): 
        # read data from file  ftype: 1=csv 2=excel
        print("Currnet wdir={}   Loading {} ".format(os.getcwd(), self.datapath))
        ds = MyHelper.load_dataset(self.datapath) 
        ds.drop(self.cols_todrop, axis = 1, inplace = True)   
        self.rawdata = ds
        return ds

#--------------------
# Classification
#--------------------    
    #2 Classification - feature variables
    #3 Classification - target variable
    #4 Classification - visualization
    def plot_cls_gradcensus(self):
        ''' Generate classification plots of graduation census data
        '''
        #2 Classification - feature variables
        rgs_target_col = self.rgs_target_col
        cls_target_col = self.cls_target_col
        
        cls_cmpl_ds = self.rawdata.copy() 
        cls_cmpl_ds[cls_target_col] = (self.rawdata[rgs_target_col] >= 90.0) * 1   # boolean True/False to 1/0
                    
        #print('Classification dataset feature variables')
        MyHelper.stats(cls_cmpl_ds)
        
        #3 Classification - target variable        
        MyHelper.stats(cls_cmpl_ds[[cls_target_col]], 1, 3)
        
        print('Classification dataset target variable (0=NOT PASSED  1=PASSED) \nvalues count=\n{} '.format( 
                cls_cmpl_ds[cls_target_col].value_counts() ) )
        
        #f, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(y = cls_target_col, data=cls_cmpl_ds, palette=sns.color_palette("Set1", n_colors=8, desat=.5)) #palette="RdBu")
        #plt.title( "US High Schools District Graduation" )
        plt.grid(True)
        plt.ylabel( 'Graduation rate over 90%' )
        plt.yticks( [0,1], ['0=NOT PASSED', '1=PASSED'])
        plt.xlabel('Count of school districts')
        plt.subplots_adjust(bottom=0.25, left=0.25)
        #plt.subplots_adjust(left=0.15, right=0.15, top=0.05, bottom=0.25)
        plt.savefig('saved/plot_cls_gradcensus.1.png')
        plt.show()
        
        #4 Classification - visualization
        f, ax2 = plt.subplots(figsize=(12, 16))
        #plt.title( "US High Schools graduation rate over 90% by state name" )
        alist = cls_cmpl_ds.pivot_table(index='STNAM', columns='Success_Pass_90', values='ALL_RATE_1112', aggfunc='count') 
        display(alist)
        MyHelper.save2file('saved/cls_pivot_count_st_success90.txt', alist )        
        
        ax2= sns.countplot( y='STNAM', hue= cls_target_col, data=cls_cmpl_ds, palette=sns.color_palette("Set1", n_colors=8, desat=.5))
        ax2.legend(labels=['0=NOT PASSED', '1=PASSED'])
        plt.grid(True)        
        plt.subplots_adjust(left=0.25, bottom=0.15)
        plt.xlabel('Count of school districts')
        plt.ylabel('US States')
        plt.savefig('saved/plot_cls_gradcensus.2.png')
        plt.show()        
        self.cls_cmpl_ds = cls_cmpl_ds
        return

    #5 Pre-process data: clean dataset 
    def preprocdata(self, dataset, target_col) : #  sel_regex, drop_regex )
        ''' preprocess data
        1. Removed non-relevant variables toward target variable.  Dropped the columns whose feature names contained the partial/full text of *MOE_, _FRMS_, Mail, Percentage, County, State, Tract, District, GIDTR, Tract, Flag, Response, Delete, Vacant, BILQ, Diff, Leave, Plumb.*
        2. Included columns whose feature names contained partial/full text of *Inc, INC, _COHORT_, pct_, avg_, _House_, _AREA_, ALL_, Success*
        3. Selected columns whose data types were float64 or int32/int64.  
        4. Dropped rows if column has NaN/nullable values
        5. Extracted target data based on the target column name
        6. Removed non-relevant variables such as the target column: (1) Success_Pass_90 for classification (2) ALL_RATE_1112 for regression
        7. Identified missing value and imputed NaN with zero by filling with zero value

        input: dataset, target columns
        output: cls_feature_cols, cls_feature_data, cls_target_data
        '''
        #  select reg expression, drop columns
        sel_regex = self.selcol_regex
        drop_regex = self.dropcol_regex
        
        fulldata = dataset.copy(deep=True)
        
        print('1. Drop colums regex={}'.format(drop_regex))  
        fulldata = fulldata.drop(dataset.filter(regex=drop_regex), axis = 1) #drop column
        #print( 'colums={}'.format(fulldata.columns.tolist()))
        if (self.dbx): display( fulldata.columns.tolist())
        
        print('2. Select colums regex={}'.format(sel_regex))            
        fulldata = fulldata.filter(regex=sel_regex, axis=1) 
        if (self.dbx): print( 'colums={}'.format(fulldata.columns))
    
        print('3. Filter only datatype float64, int32/int64')        
        fulldata = fulldata.select_dtypes(include=['float64', 'int32', 'int64'])
       
        print('4. Drop rows if col has NaN value')  
        fulldata = fulldata.dropna(subset=[target_col]) 
        fulldata = fulldata.dropna( thresh=10 ) # if count(Nan)>= 10
        
        print('5. Get target data for targe column ') 
        targetdata = fulldata[[target_col]]
        
        print('6. Drop targe column ') 
        try:
            featuredata = fulldata.drop(columns=[target_col]) 
            featuredata = featuredata.drop(columns=[self.rgs_target_col]) 
            featuredata = featuredata.drop(columns=[self.cls_target_col]) 
        except:
            None
            
        print('7. Fill in missing data with zero - impute NaN with zero')  
        featuredata.fillna(0,  inplace=True) # impute with zero.  NOT delete featuredata = featuredata.dropna(axis=1, how='any') 
        featurecols = featuredata.columns  # get feature column names
        if (self.dbx): 
            MyHelper.stats(featuredata)
            MyHelper.stats(targetdata)        
        return featurecols, featuredata, targetdata

    #6 Classification - process for features data
    # criteria to select columns
    def preproc_cls_data(self ): # cls_cmpl_ds): 
        
        self.cls_feature_cols, self.cls_feature_data, self.cls_target_data = self.preprocdata(
                self.cls_cmpl_ds, self.cls_target_col )
        print( 'feature columns={}'.format(self.cls_feature_cols))
        print('cls_feature_data')

        # transform to int for classification 
        self.cls_feature_data = self.cls_feature_data.astype(int)
        self.cls_target_data = self.cls_target_data.astype(int)

        if (self.dbx): 
            MyHelper.stats(self.cls_feature_data, 3, 1)
        else:
            MyHelper.stats(self.cls_feature_data)

        print('cls_target_data')
        if (self.dbx): 
            MyHelper.stats(self.cls_target_data, 3,1)
        else: 
            MyHelper.stats(self.cls_target_data)
        return
    
    # 7 Classification - feature selection
    def cls_feature_sel(self, sel_feat_nbr=20):         
        '''Classification feature selection: implement the recursive feature elimination method of feature ranking via the use of classifier model
        default selected feature numbers to the first 20 features which had the hightest ranks
        '''
        savefname='saved/cls_feature_sel.pkl'
        redofit=True
        
        # create a base classifier used to evaluate a subset of attributes
        X= self.cls_feature_data
        Y= self.cls_target_data
        
        if (not redofit) and (os.path.exists(savefname)):
            print( '{} exist.   Classification Feature Select data loaded from a file'.format(savefname))
            rfe = MyHelper.load_pklfile(savefname)
        else:        
            model = self.get_cls_estimator()
        
            # create the RFE model and select 20 attributes
            rfe = RFE(model, sel_feat_nbr)   # default to select the first 20th ranked features
            rfe = rfe.fit(X, Y)
            print( 'Save Classification Feature Select data to a file {}'.format(savefname))
            MyHelper.mdl2pklfile( savefname, rfe)
        
        # summarize the selection of the attributes
        print('Number of Features: {}'.format(rfe.n_features_))
        print('Selected Features Indicator: {}'.format(rfe.support_))
        print('Feature Ranking: {}'.format(rfe.ranking_))
        
        zlist = zip(rfe.ranking_, self.cls_feature_cols)
        alist = sorted(zlist)        
        MyHelper.save2file('saved/cls_feature_sel_ranking.txt', alist )
        
        selresult=list()
        
        for i in range(len(alist)): 
            k, v = alist[i]
            if (k==1): selresult.append(v)
        
        cls_selresult = selresult
        print(cls_selresult)
        print('Selected Features/columns: {}'.format(cls_selresult))
        self.cls_feature_data = self.cls_feature_data[cls_selresult]
        self.cls_feature_cols = cls_selresult
        MyHelper.stats(self.cls_feature_data )        
        return

    # 8 Classification - reduced to 20 features - statistic summary 
    # Cited: Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical 
    # modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
    # http://www.statsmodels.org/stable/index.html
    def cls_stats(self):
        ''' classification  statistic summary of Ordinary Least Squares '''
        stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
       
        X= self.cls_feature_data
        y= self.cls_target_data
        # OLS - Ordinary Least Squares 
        results = smf.OLS(y, X).fit()
        print(results.summary())
        #print('Parameters: ')
        #print(results.params )
        #print('Standard errors: ')
        #print(results.bse)
        #print('R2: ')
        #print(results.rsquared)
        #print('Predicted values: ', results.predict())
        MyHelper.smry2file('saved/cls_ols_statssummary.txt', results )
        MyHelper.smry2file('saved/cls_ols_statssummary.csv', results )


    #9 classification: create shuffle and split data: training 70%, testing 30%
    def create_cls_sample(self) : #cls_feature_data, cls_target_data) :
        '''classification: create shuffle and split data: training 70%, testing 30%
        '''
        np.random.seed( self.random_state )
        
        # Split the feature and target data into 70% for training and 30% for testing sets
        #cls_X_train, cls_X_test, cls_y_train, cls_y_test = \
        self.cls_X_train, self.cls_X_test, self.cls_y_train, self.cls_y_test = \
        train_test_split( self.cls_feature_data, self.cls_target_data, test_size = self.test_size, random_state = self.random_state)
        
        # Success
        print("cls Training and testing split was successful.  Split using target variable={}".format(self.cls_target_data.columns.tolist()))
        print("Count of training set is {} ({:.2f}%)  testing set is {}  ({:.2f}%)  in total {}.".format(
                self.cls_X_train.shape[0], 100 * self.cls_X_train.shape[0]/self.cls_feature_data.shape[0], 
                self.cls_X_test.shape[0], 100 * self.cls_X_test.shape[0]/self.cls_feature_data.shape[0], 
                self.cls_feature_data.shape[0] ))
        
        MyHelper.stats(self.cls_X_train, 2)
        MyHelper.stats(self.cls_y_train, 2)
        #return self.cls_X_train, self.cls_X_test, self.cls_y_train, self.cls_y_test

    # 10 classification- Benchmark function
    # cited: Classification of text documents using sparse features
    # http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html
    # 1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
    # 2. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.
    #    
    def cls_benchmark(self, clf, topnbr=0):
        ''' classification- Benchmark function calculate accuracy, coef, 
        classification report, density, confusion matrix
        '''        
        print('_' * 80)
        print("Training: ")
        print(clf)
                
        X_train = self.cls_X_train # self.cls_feature_data
        y_train = self.cls_y_train # self.cls_target_data
        X_test = self.cls_X_test
        y_test = self.cls_y_test

        t0 = time()
        clf.fit(X_train, y_train)
        feature_names = X_train.columns
        target_names = y_train.columns    
        print("fit -> clf.score: {0:8.3f}".format( clf.score(X_test, y_test)))   
        
        train_time = time() - t0
        print("--- train time: {0:8.3f}s".format(train_time))
    
        t0 = time()
        pred = clf.predict(X_test)
        
        test_time = time() - t0
        print("--- test time:  {0:8.3f}s".format(test_time))
    
        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   {0:8.3f}".format(score))
    
        if hasattr(clf, 'coef_'):
            print("dimensionality: {0:d}".format(np.count_nonzero(clf.coef_))) 
            print("coef_: {}".format(clf.coef_))
            print("density: {0:12.3f}".format(density(clf.coef_)))
    
            if feature_names is not None:
                totalcnt = np.count_nonzero(clf.coef_)
                if topnbr==0: topnbr = totalcnt
                print("dimensionality/count(non zero of coef_): {} ".format( totalcnt ))
                print("density: {0:12.3f}".format( density(clf.coef_)))
                if feature_names is not None:
                    for i, label in enumerate(target_names):
                        top = np.argsort(clf.coef_[i])[ -1*topnbr :]
                        print('Target feature: {}'.format(label)) 
                        print("Top {} Features".format( topnbr))
                        display( sorted(zip(clf.coef_[i][top], feature_names[top]), reverse=True))
                        #display( list(zip(feature_names[top10], cls.coef_[i][top10])))
                print()
    
            print("classification report:")
            print(metrics.classification_report(y_test, pred,
                              target_names=target_names))
    
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))
    
        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
    
    def trim(self, s):
        """Trim string to fit on terminal (assuming 80-column display)"""
        return s if len(s) <= 80 else s[:77] + "..."


#11.2 Classification - implement the model and get benchmark
# PhD Jason Browlee described in https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
# 5 of the most common metrics for evaluating predictions on classification machine learning problems:
# 
# 1. Classification Accuracy - the number of correct predictions made as a ratio of all predictions made.
# -> an equal number of observations in each class and that all predictions and prediction errors are equally important, which is often not the case
# 2. Logarithmic Loss/logloss - performance metric for evaluating the predictions of probabilities of membership to a given class.
# -> Smaller logloss is better with 0 representing a perfect logloss
# 3. Area Under ROC Curve/AUC - performance metric for binary classification problems
# The AUC represents a model’s ability to discriminate between positive and negative classes. An area of 1.0 represents a model that made all predictions perfectly. An area of 0.5 represents a model as good as random.
# -> ROC can be broken down into sensitivity and specificity. 
#   Sensitivity is the true positive rate also called the recall. It is the number instances from the positive (first) class that actually predicted correctly.
#   Specificity is also called the true negative rate. Is the number of instances from the negative class (second) class that were actually predicted correctly.
# 4. Confusion Matrix - prsentation of the accuracy of a model with two or more classes.
# The table presents predictions on the x-axis and accuracy outcomes on the y-axis. 
# Predictions for 0 that were actually 0 appear in the cell for prediction=0 and actual=0, whereas predictions for 0 that were actually 1 appear in the cell for prediction = 0 and actual=1. 
# 5. Classification Report - displays the precision, recall, f1-score and support for each class.
# all of the input variables are numeric 
# 

    def cls_benchmark2(self, cls, topnbr=0):    #    def cls_benchmark2(cls, X_train, X_test, y_train, y_test, X, Y, topnbr=0): 
        ''' Classification Benchmark using 10-fold cross validation.  
        Metric include accuracy, ROC, Confusion Matrix, Precision, Recall, F1 score
        '''
        X_train = self.cls_X_train # self.cls_feature_data
        y_train = self.cls_y_train # self.cls_target_data
        X_test = self.cls_X_test
        y_test = self.cls_y_test
        X=X_train
        Y=y_train

        print('_' * 80)
        print("Training: ")
        print(cls)
        
        t0 = time()
        cls.fit(X_train, y_train)
        feature_names = X_train.columns
        target_names = y_train.columns
        print("fit -> cls.score: {0:8.3f}".format( cls.score(X_test, y_test)))   
                
        train_time = time() - t0
        print("--- train time: {0:8.3f}s".format(train_time))
    
        t0 = time()
        pred = cls.predict(X_test)
    
        test_time = time() - t0
        print("--- test time:  {0:8.3f}s".format(test_time))
        
        if hasattr(cls, 'coef_'):
            totalcnt = np.count_nonzero(cls.coef_)
            if topnbr==0: topnbr = totalcnt
            print("dimensionality/count(non zero of coef_): {} ".format( totalcnt ))
            print("density: {0:12.3f}".format( density(cls.coef_)))
    
            if feature_names is not None:
                for i, label in enumerate(target_names):
                    top = np.argsort(cls.coef_[i])[ -1*topnbr :]
                    print('coef_-->\nTarget feature: {}'.format(label)) 
                    print("Top {} Features".format( topnbr))
                    display( sorted(zip(cls.coef_[i][top], feature_names[top]), reverse=True))
                    #display( list(zip(feature_names[top10], cls.coef_[i][top10])))
            print()
            
        print("confusion matrix:")
        mtrx = metrics.confusion_matrix(y_test, pred) 
        print(mtrx)
    
        print("classification report:")
        rpt = metrics.classification_report(y_test, pred, target_names=target_names)
        print(rpt)
            
        if hasattr(cls, 'support_vectors_'):
            #print("support_vectors_: {} ".format(cls.support_vectors_))
            print("count support_vectors_: {} ".format( np.count_nonzero(cls.support_vectors_)))
    
    # Note: cross_val_score function report the performance, and looking for ascending order, the largest score the best
    # A 10-fold cross-validation test harness, the most likely scenario in various different algorithm evaluation metrics
    # Regression: all input variables are numeric
        kfold = model_selection.KFold(n_splits= self.n_splits, random_state = self.random_state)
        model = cls
     
        # Evaluate the models using crossvalidation
        print("cross_val_score ------> ")
        scoring = 'accuracy'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(" Accuracy:\tmean={0:8.3f} std={1:8.3f}".format(cv_results.mean(), cv_results.std()))
        accuracy_score = cv_results.mean()
        
        scoring = 'roc_auc'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(" AUC:\tmean={0:8.3f} std={1:8.3f}".format(cv_results.mean(), cv_results.std()))
        roc_auc = cv_results.mean()
        
        scoring = 'precision'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(" precision:\tmean={0:8.3f} std={1:8.3f}".format(cv_results.mean(), cv_results.std()))
        precision = cv_results.mean()
        
        scoring = 'average_precision'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(" avg precision:\tmean={0:8.3f} std={1:8.3f}".format(cv_results.mean(), cv_results.std()))
        avg_precision = cv_results.mean()
        
        scoring = 'recall'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(" recall:\tmean={0:8.3f} std={1:8.3f}".format(cv_results.mean(), cv_results.std()))
        recall = cv_results.mean()
        
        scoring = 'f1'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print(" f1:\tmean={0:8.3f} std={1:8.3f}".format(cv_results.mean(), cv_results.std()))
        f1_score = cv_results.mean()

        print("Scores ------>")
        print("accuracy\tROC_AUC\tprecision\tavgprecision\trecall\tf1")
        print("{0:8.3f}\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}\t{4:8.3f}\t{5:8.3f}".format(
                accuracy_score, roc_auc, precision, avg_precision, recall, f1_score))
        
#        print("metric ------>")
#        print("accuracy\tROC_AUC\tprecision\tavgprecision\trecall\tf1")
#        print("{0:8.3f}\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}\t{4:8.3f}\t{5:8.3f}".format(
#        metrics.accuracy_score(y_test, pred), 
#        metrics.roc_auc_score(y_test, pred), 
#        metrics.precision_score(y_test, pred), 
#        metrics.average_precision_score(y_test, pred),
#        metrics.recall_score(y_test, pred), 
#        metrics.f1_score(y_test, pred) ))
        
        print()
        cls_descr = str(cls).split('(')[0]
        return cls_descr, train_time, test_time, accuracy_score, roc_auc, precision, avg_precision, recall, f1_score

    #11.3 Classification - implement the model and get benchmark    
    def cls_visual_benchmark(self): #X_train, X_test, y_train, y_test, X, Y) :
        ''' Create visual comparison plot for classification benchmark result
        '''
        cls_results2 = []
        
        params = {'n_estimators': 100, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': self.random_state,
                           'min_samples_split': 5}
        for cls, name in (
            (GaussianNB(), "Gaussian Naive Bayes(NB)"),    
            (LogisticRegression(), "Logistic Regression(Logistic)"),    
            (RandomForestClassifier(max_depth=5, n_estimators=self.n_estimators), "Random Forest(RF)"),
            (KNeighborsClassifier(n_neighbors=self.n_neighbors), "K-Nearest Neighbors(KNN)"), 
            (DecisionTreeClassifier(max_depth=5), "Decision Tree(DT)"),
            (RidgeClassifier(), "Ridge Classifier(Ridge)"),
        #    (SVC(kernel="linear", C=0.025), "SVC Linear"),   
            (Perceptron(n_iter=50, alpha=0.1, penalty=None), "Perceptron"),
            (GradientBoostingClassifier(**params), "Gradient Boosting Classifier(GB)"),
        ):
        
            print('=' * 80)
            print(name)
            #cls_results2.append( self.cls_benchmark2(cls, X_train, X_test, y_train, y_test, X, Y))
            rs = self.cls_benchmark2(cls)
            cls_results2.append( rs )

        idx = np.arange(len(cls_results2))
        #print(cls_results2)

        pd_result = pd.DataFrame(data=cls_results2, \
                columns=['cls_names', 'train_time', 'test_time', 'accuracy_score', \
                    'roc_auc', 'precision', 'avg_precision', 'recall', 'f1_score'])
        # f1 score the higher the better fit
        pd_result = pd_result.sort_values( by='f1_score', ascending=False)    
        pd_result['train_time'] = pd_result['train_time'] / np.max(pd_result['train_time'])
        pd_result['test_time'] = pd_result['test_time'] / np.max(pd_result['test_time'])
        
        print("Benchmark Summary for Classification (sorted by f1 score Desc (higher to lower))")
        print('- ' * 30)
        print(pd_result)
        MyHelper.save2file('saved/cls_visual_benchmark.txt', pd_result )   
        MyHelper.save2csvfile( "saved/cls_visual_benchmark.csv", pd_result)
            
        #12 classification - benchmark visualization    
        # visual plots illustrate the accuracy, training time (normalized) and test time (normalized) of each classifier.
        plt.figure(figsize=(14, 15))
        plt.grid(True, which='both', axis='both')
        #plt.title("Classification Benchmark Metric by Cross Validation Scoring")
        plt.xlabel( 'Classification Benchmark F1 and Accuracy Score' )
        plt.barh(idx , pd_result['train_time'], .1, label="training time", color='orange')
        plt.barh(idx + .13, pd_result['test_time'], .1, label="test time", color='purple')
        plt.barh(idx + .30, pd_result['accuracy_score'], .18, label="accuray score", color='blue')
        plt.barh(idx + .51, pd_result['f1_score'], .18, label="f1 score", color='lightgreen')
        plt.subplots_adjust(left=0.30)
        
        plt.yticks(())
        plt.legend(loc=4)  # lower right
        
        for i, c in zip(idx, pd_result['cls_names']):
            plt.text(-.3, i, c)

        plt.savefig('saved/cls_visual_benchmark.f1.png')        
        plt.show()    


# additional tryout
    def cls_ftest(self):    
        ''' Get classification f-test score, estimated mutual information between each feature and the target.
        '''        
       # create a base classifier used to evaluate a subset of attributes
        X= self.cls_feature_data
        Y= self.cls_target_data
         
        f_test, _ = f_regression(X, Y)
        f_test /= np.max(f_test)
        mi = mutual_info_regression(X, Y)
        mi /= np.max(mi)
        print('classification f test -->')
        for i in range(len(X.columns)):
            print(" {}\tF-test=\t{:.2f}\t mi=\t{:.2f}".format(X.columns.values[i], f_test[i], mi[i]))
        
        plt.figure(figsize=(8, 10))
        plt.grid(True)
        plt.plot( f_test, range(len(X.columns)), "red", label="f-test")
        plt.plot( mi, range(len(X.columns)), "blue", label="mutual_info_regression")
        plt.xlabel( 'Classification Ftest and MI Scores' )
        plt.yticks(range(len(X.columns)), X.columns.values)
        plt.legend(loc='best') 
        plt.savefig('saved/cls_ftest.png') 
        plt.show()

   
    def get_cls_estimator(self):
        '''get classification estimator for feature importance
        '''
        return RandomForestClassifier(max_depth= 8, min_samples_leaf=5, criterion="entropy", random_state= self.random_state)
        #return RandomForestClassifier(max_depth=5, n_estimators=self.n_estimators)
        #return ExtraTreesClassifier()
        
    def cls_acc_featimportance(self, samplesize=0):            
        ''' Get Classifer accurancy score and feature importance the highest; last row is lowest
        '''        
        if samplesize == 0: samplesize = self.cls_feature_data.shape[0]
        data= self.cls_feature_data[0:samplesize]
        target_feature= self.cls_target_col
        target_data = self.cls_target_data[0:samplesize]
        acc_scores = {}  
        featimpt = {}
        
        cnt=0
        print("The mean Accuracy score for features are sorted the highest to lowest ")
        for trgt_feat in data.columns.values:
        # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
            trgt_data = data[trgt_feat]
            new_data = data.copy(deep=True)
            new_data.drop( [trgt_feat], axis= 1 , inplace = True) #drop column

            # Split the data into training and testing sets using the given feature as the target
            X_train, X_test, y_train, y_test = train_test_split (
                new_data, trgt_data, test_size = self.test_size, random_state = self.random_state)
             
            # transform to int for classification 
            X_train = X_train.astype(int)
            y_train = y_train.astype(int)
            X_test = X_test.astype(int)
            y_test = y_test.astype(int)
            
            feature_cols = X_train.columns.values
            colcnt = X_train.shape[1]
            
            # Create a decision tree classifer and fit it to the training set
            classifer = self.get_cls_estimator()       
            classifer.fit( X_train, y_train)
            # this is the mean accuracy of the prediction.
            acc_scores[trgt_feat] = classifer.score(X_test, y_test)
        
        # accuracy score            
        akeys = sorted(acc_scores, key= acc_scores.get, reverse=True)        
        print(akeys)
        best_score = 0.0
        best_feature = ''
        
        cnt=0        
        print('Sorting Mean Accuracy score, the highest to lowest')
        for feature in akeys: 
            if cnt==0: 
                best_feature = feature
                best_score = acc_scores[feature]
            cnt += 1
            print("#{}\t{:20}\tAccuracy= \t{:+.4f}".format(cnt, feature, acc_scores[feature]))                

        print( '\n--> {} has the highest mean accuracy score {:.4f}  '.format( best_feature, best_score) )        
        MyHelper.save2file('saved/cls_accuracy.txt', acc_scores)
        
        print('Feature importance --> ')
        alist = sorted(zip(feature_cols, classifer.feature_importances_))
        display(alist)
        
        h = max(11, 0.25 * colcnt)
        
        plt.figure(figsize=(12, h))                
        plt.grid(True)
        plt.ylabel( 'Classification Features' ) 
        plt.xlabel( 'Relative Importance' )
        cnt=0
        for i in classifer.estimators_ :
            cnt += 1
            plt.plot( i.feature_importances_ , range( len(i.feature_importances_ ) ), "red", label="n_estimators={}".format(cnt))           
        plt.plot( classifer.feature_importances_, range(len(classifer.feature_importances_)), "blue", label="average", linewidth=2)    
        plt.yticks(range( len(classifer.feature_importances_) ), feature_cols)
        plt.legend(loc='best') 
        plt.subplots_adjust(left=0.20)
        
        plt.savefig('saved/cls_featimportance.png')        
        plt.show()

        print('Sorting feature importance, the hight to lowest ')
        alist = sorted(zip(classifer.feature_importances_, feature_cols), reverse=True )
        display(alist)
        savefname='saved/cls_featimportance.txt'        
        MyHelper.save2file(savefname, alist )        

        cfimpt = classifer.feature_importances_
        cnt = 0
        # Sort feature importances in descending order
        indices = np.argsort(cfimpt)[::-1]
        for i in indices: 
            cnt += 1
            featnames = feature_cols[i]
            featimpt[featnames] = cfimpt[i]
            print('#{}  iloc={}\t{} \timportance=\t{}'.format(cnt, i, featnames,cfimpt[i] ))

        # Sort feature importances in descending order
        print('Visualize Features in Correlation Matrix->')

        # get correlations
        good_data, good_targetdata = self.feat_scaling(data, target_data, 'saved/cls_acc_featimportance.scaling.png', 'cls')
        if samplesize==0:
            self.cls_feature_data = good_data
            self.cls_target_data = good_targetdata
        
        return

    def feat_scaling(self, data, targetdata, savedfname, type):
        
        good_data, good_targetdata = self.outlier_detection(data, targetdata, type)
        
        corr= np.round( good_data.corr(), 2)
        display( 'Before log-transformed, log_data Mean=Average & Median=50% -> ', 
            np.round( good_data.describe().loc[['mean','50%']], 2))
        display(corr)

        # Scale the data using the natural logarithm
        log_data = good_data.copy(deep=True) 
        log_data = np.log( log_data )
        display( 'After log-transformed, log_data Mean=Average & Median=50% -> ', 
            np.round(log_data.describe().loc[['mean','50%']], 2))
        MyHelper.stats(log_data )

        # get correlations
        print( 'log_data in Correlation Matrix -> ')
        log_corr= np.round( log_data.corr(), 2)
        display( log_corr )        
        # Scale the sample data using the natural logarithm
        
        #print ('Visualize log_data in Scatter Matrix -> ')
        print('Visualize comparing data and log_data in Correlation Matrix')
        colcnt = good_data.shape[1]
        h = max( 12, 0.5 * colcnt) 
        
        fig, (ax1, ax2)= plt.subplots(nrows=2, ncols=1,  figsize=( h , 1.5 * h ), sharex=False, sharey=False) 
        ax1.set_title( 'data ' )
        ax2.set_title( 'log_data ' )
        # symmetric visualization: dispaly one side only
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask, 1)] = True
        with sns.axes_style("white"):
            sns.heatmap( corr, annot=True, linewidth=0.3, fmt='-.1f', cmap='PuOr', mask=mask, square=True, ax=ax1)
            sns.heatmap( log_corr, annot=True, linewidth=0.3, fmt='-.1f', cmap='PuOr', mask=mask, square=True, ax=ax2)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        plt.subplots_adjust(bottom=0.1, left=0.30)
        
        plt.savefig(savedfname)        
        plt.show()
        plt.close()
        
#        good_data = self.outlier_detection(data, type)
        self.pca(good_data, type)
        
        return good_data, good_targetdata
        
    def cls_feature_re_sel(self):         
        '''Classification feature re-selection using logistics regression model to rank,
        select the first 20 features the hightest ranks
        '''        
        selcols=6
        alist = self.cls_featimportance()
        selresult=list()        
        # Sort feature importances in descending order
        indices = np.argsort(cfimpt)[::-1]
        # Rearrange feature names so they match the sorted feature importances
        cnt=0
        for i in range(max(selcols, len(alist))): 
            k, v = alist[i]
            selresult.append(v)
        
        cls_selresult = selresult
        print('cls_feature_re_sel Features/columns: {}'.format(cls_selresult))
        self.cls_feature_data = self.cls_feature_data[cls_selresult]
        self.cls_feature_cols = cls_selresult
        MyHelper.stats(self.cls_feature_data )
        return
    
    def outlier_detection(self, ldata, tdata, type):
        '''Using Tukey's Method [35] to identfy outliers and removed outliers
        '''
        ldata.reset_index(drop=True)
        tdata.reset_index(drop=True)
        
        outliers  = []
        # For each feature find the data points with extreme high or low values
        for feat in ldata.keys():            
            # Calculate Q1 (25th percentile of the data) for the given feature
            Q1 = np.percentile( ldata[feat], 25)            
            # Calculate Q3 (75th percentile of the data) for the given feature
            Q3 = np.percentile( ldata[feat], 75)            
            # Compute interquartile range  IQR=Q3-Q1
            IQR = Q3 - Q1            
            # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
            # anything outside this range is an outlier
            step = 1.5 * IQR
            
            # Display the outliers
            feat_outlier = ldata[~((ldata[feat] >= Q1 - step) & (ldata[feat] <= Q3 + step))]
            print("\nData points considered outliers for the feature --> '{}'" \
                  "\n   Q1={:.4f}\tQ3= {:4f}\tstep= 1.5*(Q3-Q1) = {:.4f}\t\t Feeature Outlier cnt= {}".format( \
                  feat, Q1, Q3, step, len(feat_outlier) ))
            outliers.extend( list(feat_outlier.index.values) )
        
        # OPTIONAL: Select the indices for data points you wish to remove
        # if counter for the outlier found in which features, 
        # if counter is > 1, consider duplicate feature.  Otherwise, expect 1 for each feature
        duplicated_feature_outliers = (Counter(outliers) - Counter(set(outliers))).keys() 
        print( '\n\ndata size={} max_idx={}  Outliers for all features ={} ' \
        '\nNote: Also found duplicate outliers in multiple features ={} '.format( \
                ldata.shape[0], max(ldata.index.values), len(outliers) , len(duplicated_feature_outliers) ))
                            
        # Remove the outliers, if any were specified                    
        duplist = list( filter(lambda x : x < ldata.shape[0], duplicated_feature_outliers) )
        good_data = ldata.drop(ldata.index[ duplist ]).reset_index(drop = True)
        good_targetdata = tdata.drop(tdata.index[ duplist ]).reset_index(drop = True)
            
        print('\nRemoved duplicated outlier data -> good datasize={}   target datasize={}  \n'.format( good_data.shape, good_targetdata.shape ))

        return good_data, good_targetdata
    
    def pca(self, good_data, type) :
        ''' Apply PCA by fitting the good data with the same number of dimensions as features
        and Get classification PCA Explained Variance
        '''        
        # feature extraction
        #X= good_data #self.cls_feature_data
        good_data_colsize = good_data.shape[1]
        print ( 'Extracting the top {} features from {} data points\n\n'.format( good_data_colsize,  good_data.shape[0])) 
        pca = PCA(n_components= good_data_colsize, svd_solver='auto')
        pca.fit( good_data )
              
        # summarize components
        print (' PCA Explained variance ratio={}'.format( np.round(pca.explained_variance_ratio_, 2)))
        print( pca.components_)

        '''
        Create a DataFrame of the PCA results
        Includes dimension feature weights and explained variance
        Visualizes the PCA results
        '''
            
        # Dimension indexing
        dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
            
        # PCA components
        components = pd.DataFrame(np.round(pca.components_, 2), columns = good_data.keys())
        components.index = dimensions
            
        # PCA explained variance
        ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
        variance_ratios = pd.DataFrame(np.round(ratios, 2), columns = ['Explained Variance'])
        variance_ratios.index = dimensions
            
        # Create a bar plot visualization
        h = max(10, 0.2*good_data_colsize) 
        fig, ax = plt.subplots(figsize = ( 1.5 * h, h ))
            
        # Plot the feature weights as a function of the components
        components.plot(ax = ax, kind = 'bar');
        ax.set_ylabel("Feature Weights")
        ax.set_xticklabels(dimensions, rotation=90)            
            
        # Display the explained variance ratios
        for i, ev in enumerate(pca.explained_variance_ratio_):
        	ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n  %.4f"%(ev))

        plt.legend(loc='best') 
            
        # Return a concatenated DataFrame
        pd.concat([variance_ratios, components], axis = 1)
        
        plt.savefig('saved/{}_pca.png'.format(type))    
        plt.show()
        plt.close()
        
        return
 

    def rank2ict(self, ranks, names, order=1):
        minmax = MinMaxScaler(copy=True, feature_range=(0, 1))
        ranks = minmax.fit_transform( ranks.T ).T[0]
        ranks = map(lambda x: np.round(x, 2), ranks.T)
        return dict(zip(names, ranks ))
    
    def compare_cls_featranking(self, minRanking=0.15 ): 
        '''Compare all feature ranking result and plot the highest rankings if greater than min ranking value
        the result and plot will be saved to files
        '''

        X = self.cls_feature_data
        Y = self.cls_target_data
        col_names = self.cls_feature_cols 
        ranks = {}
        
        logit = LogisticRegression()
        logit.fit(X, Y)
        ranks["logit"] = self.rank2ict(np.abs(logit.coef_), col_names)
        

        ridge = RidgeClassifier()
        ridge.fit(X, Y)
        ranks["Ridge"] = self.rank2ict(np.abs(ridge.coef_), col_names)
        
        dt = DecisionTreeClassifier(max_depth=5)
        dt.fit(X, Y)
        ranks["DT"] = self.rank2ict( np.array([dt.feature_importances_] ), col_names)     
        
        rf = RandomForestClassifier(max_depth=5, n_estimators=self.n_estimators)
        rf.fit(X,Y)
        ranks["RF"] = self.rank2ict( np.array([rf.feature_importances_]) , col_names)
                
        feat_names=col_names
       
        # average mean of ranking score
        r = {}
        est_list = ranks.keys()
        for f in feat_names: 
            r[f] = np.round(np.mean( [ranks[est][f] for est in est_list]), 2)
        
        est_list  = sorted(ranks.keys())
        ranks["Mean"] = r
        est_list.append("Mean")

        ds_ranks = pd.DataFrame(ranks)
        MyHelper.save2csvfile( "saved/cls_ds_allranking_ds.csv", ds_ranks)
        
        print('Features\t\t {}'.format( '\t'.join(map(str, est_list))))
        
        for f in feat_names: 
            print('{}\t\t{}'.format(f, '\t'.join(map(str, [ranks[est][f] for est in est_list] )))) 
        
        
        # Put the mean scores into a Pandas dataframe
        meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])        
        # Sort the dataframe
        meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
        MyHelper.save2csvfile( "saved/cls_ds_meanranking.csv", meanplot)    
        
        sdata = meanplot[meanplot['Mean Ranking'] >= minRanking ] 
        print(sdata)

        h=max(6, 0.5*len(sdata))
        # Let's plot the ranking of the features
        sns.factorplot(x="Mean Ranking", y="Feature", data = sdata, kind="bar", size=h, aspect=1, palette='coolwarm')
        plt.savefig('saved/cls_compare_ranking.png')
                
        plt.show() 
        
        # extract selected features
        selresult=list(sdata['Feature'])                
        print('Selected Features/columns: {}'.format(selresult))
        self.cls_feature_data = self.cls_feature_data[selresult]
        self.cls_feature_cols = selresult
        return
               
#--------------------
# Regression
#--------------------
    #12 Regression - feature variables
    #13 Regression - target variables
    #14 Regression - target variable   
    #15 Regression - visualization        
    def plot_rgs_gradcensus(self):
        ''' Generate regression plots based on graduation census data
        '''
        #12 Regression - feature variables
        rgs_target_col = self.rgs_target_col
        cls_target_col = self.cls_target_col
        
        rgs_cmpl_ds = self.rawdata.copy() 

        print('Regression dataset feature variables')
        MyHelper.stats(rgs_cmpl_ds)                
        
        MyHelper.stats(rgs_cmpl_ds[[rgs_target_col]], 1, 3)
        #print('Regression target variable= {}'.format( rgs_target_col ))
        #15 Regression - visualization
        f, ax = plt.subplots(figsize=(10, 8))
        #plt.title("US High Schools graduation {} distribution".format(rgs_target_col))
        plt.grid(True)
        rgs_cmpl_ds = rgs_cmpl_ds.dropna(subset=[rgs_target_col]) 
        display(rgs_cmpl_ds.head())
        sns.distplot(rgs_cmpl_ds[rgs_target_col], 
                       rug=True, rug_kws={"color":"g"},
                       kde_kws= {"color": "b", "lw": 2, "label": "Kernel Density Estimate"},
                       norm_hist=True, vertical=False,
                       axlabel = 'Density Distribution with Rug of {}'.format(rgs_target_col), 
                       label='Histogram and Kernal Density Estimate with default binsize' ) 
        self.rgs_cmpl_ds = rgs_cmpl_ds
        plt.savefig('saved/plot_rgs_gradcensus.png')   
        plt.show()        
        return

    #16 Regression - process for features data
    # criteria to select columns
    def preproc_rgs_data(self): # rgs_cmpl_ds): 
        ''' Pre process regression data
        '''
        
        self.rgs_feature_cols, self.rgs_feature_data, self.rgs_target_data = self.preprocdata(
                self.rgs_cmpl_ds, self.rgs_target_col )
        #rgs_feature_cols, rgs_feature_data, rgs_target_data = preprocdata( 
        #rgs_cmpl_ds, rgs_target_col, selcol_regex, dropcol_regex ) 
        
        print( 'feature columns={}'.format(self.rgs_feature_cols))
        print('rgs_feature_data')
        if (self.dbx): 
            MyHelper.stats(self.rgs_feature_data, 3, 1)
        else:
            MyHelper.stats(self.rgs_feature_data)
            print('rgs_target_data')
        if (self.dbx): 
            MyHelper.stats(self.rgs_target_data, 3, 1)
        else: 
            MyHelper.stats(self.rgs_target_data)
        return #return self.rgs_feature_cols, self.rgs_feature_data, self.rgs_target_data 

    #17 Resssion - Stepwise selection
    # cited: Does scikit-learn have forward selection/stepwise regression algorithm?
    # https://datascience.stackexchange.com/questions/937/does-scikit-learn-have-forward-selection-stepwise-regression-algorithm
    # 
    def stepwise_selection(self, X, y, initial_list=[], threshold_in=0.01, threshold_out = 0.05):
        """ Perform a forward-backward feature selection based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
        Returns: list of selected features 
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        """
        included = list(initial_list)
        while True:
            changed=False
            # forward step
            excluded = list(set(X.columns)-set(included)) 
            new_pval = pd.Series(index=excluded)
            
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                #print(model.summary())
                new_pval[new_column] = model.pvalues[new_column]
            best_pval = new_pval.min()
            if best_pval < threshold_in:
                best_feature = new_pval.argmin()
                included.append(best_feature)
                changed=True
                print( 'Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
    
            # backward step
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit() 
            #print(model.summary())
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
            if worst_pval > threshold_out:
                changed=True
                worst_feature = pvalues.argmax()
                included.remove(worst_feature)
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
            if not changed:
                break
        return included

    #18 regression feature selection - Stepwise selection continue
    def rgs_feature_sel(self):        
        ''' Regression feature selection using stepwise selection
        '''
        savefname='saved/rgs_feature_sel.txt'
        redofit=True
        
        rgs_X = pd.DataFrame( self.rgs_feature_data, columns= self.rgs_feature_cols)
        rgs_y = self.rgs_target_data
        
        if (not redofit) and (os.path.exists(savefname)):
            print( '{} exist.  Regression Feature Selection loaded from a file'.format(savefname))
            with open(savefname) as data_file:
                rgs_selresult = json.load(data_file)
        else: 
            rgs_selresult = self.stepwise_selection(rgs_X, rgs_y)
            with open(savefname, 'wb') as f:
                print( 'Save Regression Feature Selection to a file {}'.format(savefname))
                json.dump(rgs_selresult, codecs.getwriter('utf-8')(f), ensure_ascii=False)
        
        print('Number of Features: {}'.format(len(rgs_selresult)))
        print('Regression Selected Features/columns: {}'.format(rgs_selresult))
        self.rgs_feature_data = self.rgs_feature_data[rgs_selresult]
        self.rgs_feature_cols = rgs_selresult
        MyHelper.stats(self.rgs_feature_data )
        return
        
# 19 Regression - reduced to 34 features - statistic summary 
# Cited: Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical 
# modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.
# http://www.statsmodels.org/stable/index.html
    def rgs_stats(self):
        ''' Regression statistic summary 
        '''
        stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
       
        X= self.rgs_feature_data
        y= self.rgs_target_data

        rgsresults = smf.OLS(y, X).fit()
        print(rgsresults.summary())
        MyHelper.smry2file('saved/rgs_ols_statssummary.csv', rgsresults ) 
        MyHelper.smry2file('saved/rgs_ols_statssummary.txt', rgsresults ) 
        return
    
# 20 Regression - Benchmark function
# cited: Comparison of kernel ridge regression and SVR¶
# http://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html
# 1. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
# 2. API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.
#
# PhD Jason Bronlee described in https://machinelearningmastery.com/metrics-evaluate-machine-learning-algorithms-python/
# The most common metrics for evaluating predictions on regression machine learning problems:
# 1. Mean Absolute Error/MAE - sum of absolute differences btw predctions and actual values.
# ->How wrong the predictions were.  A value of 0 indicates no error or perfect predictions
# 2. Mean Squared Error/MSE - the mean absolute error in that it provides a gross idea of the magnitude of error.
# ->Taking the square root of the mean squared error is called "Root Mean Squared Error/RMSE"
# 3. R Sqaured/R^2 - indicating the goodness of fit of a set of predictions to the actual values. 
# -> it is called coefficient of determination in statistics
#
    def rgs_benchmark(self, rgs, topnbr=0):
        ''' Regression benchmark using 10-fold cross validation.  
        Metric include variance scor,  mean absolute error , mean square error, 
        root mean square error, median absolute error, r2 score
        '''

        print('_' * 80)        
        print("Training: ")
        print(rgs)    
        
        X_train = self.rgs_X_train # self.rgs_feature_data
        y_train = self.rgs_y_train # self.rgs_target_data
        X_test = self.rgs_X_test
        y_test = self.rgs_y_test
        X=X_train
        Y=y_train
        
        t0 = time()
        rgs.fit(X_train, y_train)
        feature_names = X_train.columns
        target_names = y_train.columns
        
        print("fit -> rgs.score: {0:8.3f}".format( rgs.score(X_test, y_test)))   
        
        train_time = time() - t0
        print("--- train time: {0:8.3f}s".format(train_time))
    
        t0 = time()
        pred = rgs.predict(X_test)
    
        test_time = time() - t0
        print("--- test time:  {0:8.3f}s".format(test_time))
        
        if hasattr(rgs, 'coef_'):
            totalcnt = np.count_nonzero(rgs.coef_)
            print("dimensionality: {0:d}".format(np.count_nonzero(totalcnt))) 
            print("coef_: {}".format(rgs.coef_))
            print("density: {0:12.3f}".format(density(rgs.coef_)))
    
            if feature_names is not None:
                if topnbr==0: topnbr = totalcnt
                print("dimensionality/count(non zero of coef_): {} ".format( totalcnt ))
                print("density: {0:12.3f}".format( density(rgs.coef_)))
                if feature_names is not None:
                    for i, label in enumerate(target_names):
                        top = np.argsort(rgs.coef_[i])[ -1*topnbr :]
                        print('coef_---->\nTarget feature: {}'.format(label)) 
                        print("Top {} Features".format( topnbr))
                        display( sorted(zip(rgs.coef_[i][top], feature_names[top]), reverse=True))
                print()
            
        if hasattr(rgs, 'support_vectors_'):
            print("support_vectors_: mean={0:12.3f}  std={1:12.3f} ".format(np.mean(rgs.support_vectors_), np.std(rgs.support_vectors_)))
    
    # Note: cross_val_score function report the performance, and looking for ascending order, the largest score the best
    # A 10-fold cross-validation test harness, the most likely scenario in various different algorithm evaluation metrics
    # Regression: all input variables are numeric
        kfold = model_selection.KFold(n_splits= self.n_splits, random_state= self.random_state)
        model = rgs
     
        # Evaluate the models using crossvalidation
        print("cross_val_score ------> ")
        scoring = 'explained_variance'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print("ExplainedVariance: mean={0:12.3f} std={1:8.3f}".format(np.abs(cv_results.mean()), cv_results.std()))
        explained_variance_score = np.abs(cv_results.mean())
        
        scoring = 'neg_mean_absolute_error'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print("MeanAbsError/MAE: mean={0:12.3f} std={1:8.3f}".format(np.abs(cv_results.mean()), cv_results.std()))
        mean_absolute_error = np.abs(cv_results.mean())
        
        scoring = 'neg_mean_squared_error'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print("MeanSqrErr/MSE: mean={0:12.3f} std={1:8.3f}".format(np.abs(cv_results.mean()), cv_results.std()))
        mean_squared_error = np.abs(cv_results.mean())
        root_mean_squared_error = np.sqrt( np.abs(mean_squared_error))
        print("Accuracy / RMSE= {0:12.3f}".format(root_mean_squared_error))
       # score= np.sqrt(metrics.mean_squared_error(y_test, pred))
    
        scoring = 'neg_median_absolute_error'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print("MedianAbsErr/Median SE: mean={0:12.3f} std={1:8.3f}".format(np.abs(cv_results.mean()), cv_results.std()))
        median_absolute_error = np.abs(cv_results.mean())
        
        scoring = 'r2'
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
        print("R^2: mean={0:12.3f} std={1:8.3f}".format(np.abs(cv_results.mean()), cv_results.std()))
        r2_score = np.abs(cv_results.mean())

        print("Scores ------>")
        print("varianceScore\tmeanAbsErr\tMeanSqrErr\tRMSE \tmedianAbsErr\tr2Score")
        print("{0:8.3f}\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}\t{4:8.3f}\t{4:8.3f}\t".format(
         explained_variance_score, mean_absolute_error, mean_squared_error, \
         root_mean_squared_error, median_absolute_error, r2_score ))
                
        print("metric------>")
        print("varianceScore\tmeanAbsErr\tMeanSqrErr\tRMSE \tmedianAbsErr\tr2Score")
        print("{0:8.3f}\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}\t{4:8.3f}\t{4:8.3f}\t".format(
        metrics.explained_variance_score(y_test, pred), 
        metrics.mean_absolute_error(y_test, pred), 
        metrics.mean_squared_error(y_test, pred), 
        np.sqrt(metrics.mean_squared_error(y_test, pred)), 
        metrics.median_absolute_error(y_test, pred), 
        metrics.r2_score(y_test, pred)
        ))  
    
        print()
        rgs_descr = str(rgs).split('(')[0]
        
        return rgs_descr, train_time, test_time, explained_variance_score, \
            mean_absolute_error, mean_squared_error, \
            root_mean_squared_error, median_absolute_error, r2_score 
         
    
    #21-------- shuffle and split data: training 70%, testing 30%
    def create_rgs_sample(self) : #rgs_feature_data, rgs_target_data) :
        np.random.seed( self.random_state )

        # Split the feature and target data into 70% for training and 30% for testing sets
        self.rgs_X_train, self.rgs_X_test, self.rgs_y_train, self.rgs_y_test = \
        train_test_split( self.rgs_feature_data, self.rgs_target_data, test_size = self.test_size, random_state = self.random_state)
        
        # Success
        print("rgs Training and testing split was successful.  Split using target variable={}".format(self.rgs_target_data.columns.tolist()))
        print("Count of training set is {} ({:.2f}%)  testing set is {}  ({:.2f}%)  in total {}.".format(
                self.rgs_X_train.shape[0], 100 * self.rgs_X_train.shape[0]/ self.rgs_feature_data.shape[0], 
                self.rgs_X_test.shape[0], 100 * self.rgs_X_test.shape[0]/ self.rgs_feature_data.shape[0], 
                self.rgs_feature_data.shape[0] ))
        
        MyHelper.stats(self.rgs_X_train, 2)
        MyHelper.stats(self.rgs_y_train, 2)
        return  #return self.rgs_X_train, self.rgs_X_test, self.rgs_y_train, self.rgs_y_test

    #22 Regression - Implement the model and get Benchmark 
    #22 Regression - benchmark visualization        
    def rgs_visual_benchmark(self): #X_train, X_test, y_train, y_test, X, Y) :
        ''' Regression visual comparison plot for classification benchmark result
        '''
        X_train = self.rgs_X_train
        X= np.array(X_train)
        y_train = self.rgs_y_train
        Y= np.array(y_train)
        X_test = self.rgs_X_test
        y_test = self.rgs_y_test
        
        rgs_results = []
        
        # Regression 
        for rgs, name in (
                (LinearRegression(), "Linear Regression(Linear)"),    
                (RandomForestRegressor(max_depth=30, random_state= self.random_state), "Random Forest(RF)"),
                (KNeighborsRegressor(n_neighbors=self.n_neighbors), "k-Nearest Neighbors(KNN)"),
                (DecisionTreeRegressor(), "Decision Tree(DT)"),
                (Ridge(alpha=0.05, normalize=True), "Ridge"),    
        #        (SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.25, fit_intercept=True), "Stochastic Gradient Descent"),
                (SVR(kernel='rbf'), "Support Vector Regression(SVR)"),
        ): 
            print('=' * 80)
            print(name)
            #rgs_results.append(self.rgs_benchmark(rgs))
            rs = self.rgs_benchmark(rgs)
            rgs_results.append(rs)
            
        idx = np.arange(len(rgs_results))
        #print(rgs_results)
        pd_result = pd.DataFrame(data=rgs_results, \
                columns=['rgs_names', 'train_time', 'test_time', 'explained_variance_score', \
                    'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error', \
                    'median_absolute_error', 'r2_score'])
        # RMSE the lower  the better fit
        pd_result = pd_result.sort_values( by='root_mean_squared_error', ascending=True)    
        pd_result['train_time'] = pd_result['train_time'] / np.max(pd_result['train_time'])
        pd_result['test_time'] = pd_result['test_time'] / np.max(pd_result['test_time'])
        pd_result['root_mean_squared_error'] = pd_result['root_mean_squared_error']
        MyHelper.save2file( "saved/rgs_visual_benchmark.txt", pd_result) 

        MyHelper.save2csvfile( "saved/rgs_visual_benchmark.csv", pd_result)

        print("Benchmark Summary for Regression (sorted by RMSE Asc (low to high))")
        print('- ' * 30)
        print(pd_result)
        
        plt.figure(figsize=( 10, 8))
        plt.grid(True, which='both', axis='both')        
        #plt.title("Regression Benchmark by Cross Validation")
        plt.xlabel( 'Regression Benchmark by RMSE' )
        
        plt.barh(idx , pd_result['train_time'], .1, label="training time", color='orange')
        plt.barh(idx + .13, pd_result['test_time'], .1, label="test time", color='purple')
        plt.barh(idx + .30, pd_result['root_mean_squared_error'] / np.max( pd_result['root_mean_squared_error'] ), .18 , label="RMSE", color='blue')
        
        plt.yticks(())
        plt.legend(loc=4)  # lower right
        
        for i, c in zip(idx, pd_result['rgs_names']):
            plt.text( -0.5, i, c)
        plt.subplots_adjust(left=0.35)
        
        plt.savefig('saved/rgs_visual_benchmark.rmse.png')           
        plt.show()        

        plt.figure(figsize=( 8, 6))
        plt.xlabel( 'Regression Benchmark by R2 score' )
        plt.barh(idx , pd_result['r2_score'], label="r2 score", color='lightgreen')
        plt.yticks(())
        plt.legend(loc=4)  # lower right
        for i, c in zip(idx, pd_result['rgs_names']):
            plt.text( -0.3, i, c)
        plt.subplots_adjust(left=0.35)
        
        plt.savefig('saved/rgs_visual_benchmark.r2score.png')           
        plt.show()        
        
        return

    def get_rgs_estimator(self):
        '''get Regression estimator for feature importance
        '''
        #return DecisionTreeRegressor(random_state = 99, max_depth=3, min_samples_leaf=5, criterion 'mse' #mean squared error)
        return RandomForestRegressor(max_depth= 3 , min_samples_leaf=5, random_state= self.random_state)
    
    def rgs_r2_featimportance(self, samplesize=0, minval=0.0): 
        ''' Get Regression estimator to fit, sort R2 scores , and sort feature importance 
        '''
        if samplesize == 0: samplesize = self.rgs_feature_data.shape[0]
        data= self.rgs_feature_data[0:samplesize]
        target_feature= self.rgs_target_col
        target_data = self.rgs_target_data[0:samplesize]
        r2_scores = {}  
        
        cnt=0
        # find feature relative 
        print("The R^2 score for features are sorted the highest to lowest ")
        for trgt_feat in data.columns.values:
            trgt_data = data[trgt_feat]
            # Make a copy of the DataFrame, using the 'drop' function to drop the given feature
            new_data = data.copy(deep=True)
            new_data.drop( [trgt_feat], axis= 1 , inplace = True) #drop column
            
            # Split the data into training and testing sets using the given feature as the target
            X_train, X_test, y_train, y_test = train_test_split (
                new_data, trgt_data, test_size = self.test_size, random_state = self.random_state)
                
            feature_cols = X_train.columns.values
            colcnt = X_train.shape[1]
    
            # Show the results of the split
            # Create a decision tree regressor and fit it to the training set
            regressor = self.get_rgs_estimator()            
            regressor.fit( X_train, y_train)
            # this is the coefficient of determination R^2 of the prediction.
            r2_scores[trgt_feat] = regressor.score(X_test, y_test)
            # alternative way to calculate R2 score
            # prediction = regressor.predict(X_test)
            # r2_scores[trgt_feat] = r2_score(y_test, prediction)   
        
        akeys = sorted(r2_scores, key=r2_scores.get, reverse=True)
                
        best_score = 0.0
        best_feature = ''
        cnt=0
        
        print('Sorting R2 score, the highest to lowest')
        for feature in akeys: 
            if cnt==0: 
                best_feature = feature
                best_score = r2_scores[feature]
            cnt += 1
            print("# {}\t{:20}\tR2= \t{:+.4f} ".format(cnt, feature, r2_scores[feature]))                
        print( '\n--> {} has the highest R2 score {:.4f}  '.format( best_feature, best_score) )
        MyHelper.save2file('saved/rgs_r2.txt', r2_scores)
        
        print('Feature importance --> ')
        alist = sorted(zip(data.columns.values, regressor.feature_importances_))
        display(alist)

        h = max(12, 0.3 * colcnt)            
        plt.figure(figsize=(15, h))                
        plt.grid(True)
        plt.ylabel( 'Regression Features' ) 
        plt.xlabel( 'Relative Importance' )
        cnt=0
        for i in regressor.estimators_ :
            cnt += 1
            plt.plot( i.feature_importances_ , range( len(i.feature_importances_ ) ), "red", label="n_estimators={}".format(cnt))           
        plt.plot( regressor.feature_importances_, range(len(regressor.feature_importances_)), "blue", label="average", linewidth=2)    
        plt.yticks(range( len(regressor.feature_importances_) ), feature_cols)
        plt.legend(loc='best')  
        plt.subplots_adjust(bottom=0.40, left=0.40)
        
        plt.savefig('saved/rgs_featimportance.png')        
        plt.show()

        print('Sorting feature importance, the hight to lowest ')
        alist = sorted(zip(regressor.feature_importances_, feature_cols), reverse=True )
        display(alist)
        savefname='saved/rgs_featimportance.txt'        
        MyHelper.save2file(savefname, alist )
        
        # Sort feature importances in descending order
        print('Visualize Features in Correlation Matrix->')
        # get correlations
        good_data, good_targetdata = self.feat_scaling(data, target_data, 'saved/rgs_corr_feat.scaling.png', 'rgs')
        if samplesize==0:
            self.rgs_feature_data = good_data
            self.rgs_target_data = good_targetdata
            
        #self.rgs_feature_re_sel(alist, minval)
        return

    def rgs_feature_re_sel(self, alist, minval=0.0):         
        '''Regression feature re-selection using logistics regression model to rank,
        select the first nth features the hightest ranks
        '''        
        selcols=len(alist)
        selresult=list()        
        for i in range(selcols): 
            k, v = alist[i]
            if k > minval:
                selresult.append(v)        
        rgs_selresult = selresult
        print('rgs_feature_re_sel Features/columns: {}'.format(rgs_selresult))
        self.rgs_feature_data = self.rgs_feature_data[rgs_selresult]
        self.rgs_feature_cols = rgs_selresult
        MyHelper.stats(self.rgs_feature_data )
        self.rgs_stats()
        
        # Sort feature importances in descending order
        print('Visualize Features in Correlation Matrix->')
        # get correlations
        self.feat_scaling(self.rgs_feature_data, 'saved/rgs_corr_featimportance.scaling.2.png', 'rgs')
               
        return selresult
    
    def compare_rgs_featranking(self, minRanking=0.15 ): 
        '''Compare all feature ranking result and plot the highest rankings if greater than min ranking value
        the result and plot will be saved to files
        '''

        X = np.array(self.rgs_feature_data)
        Y = np.array(self.rgs_target_data)
        col_names = self.rgs_feature_cols 
        ranks = {}
         
        lr = LinearRegression(normalize=True)
        lr.fit(X, Y)
        ranks["Linear"] =  self.rank2ict( np.abs(lr.coef_), col_names)
        
        ridge = Ridge(alpha=0.05, normalize=True)
        ridge.fit(X, Y)
        ranks["Ridge"] = self.rank2ict(np.abs(ridge.coef_), col_names)
        
#        lasso = Lasso(alpha=.05)
#        lasso.fit(X, Y)
#        ranks["Lasso"] = rank2ict( np.abs([lasso.coef_]), col_names)
        
        dt = DecisionTreeRegressor()
        dt.fit(X, Y)
        ranks["DT"] = self.rank2ict( np.array([dt.feature_importances_] ), col_names)     
        
        rf = RandomForestRegressor(max_depth=30, random_state= self.random_state) 
        #rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
        rf.fit(X,Y)
        ranks["RF"] = self.rank2ict( np.array([rf.feature_importances_]) , col_names)

        f, pval  = f_regression(X, Y, center=True)
        ranks["Corr"] = self.rank2ict(np.array([f]), col_names)
                
        feat_names=col_names
       
        # average mean of ranking score
        r = {}
        est_list = ranks.keys()
        for f in feat_names: 
            r[f] = np.round(np.mean( [ranks[est][f] for est in est_list]), 2)
        
        est_list  = sorted(ranks.keys())
        ranks["Mean"] = r
        est_list.append("Mean")

        ds_ranks = pd.DataFrame(ranks)
        MyHelper.save2csvfile( "saved/rgs_ds_allranking_ds.csv", ds_ranks)
        
        print('Features\t\t {}'.format( '\t'.join(map(str, est_list))))
        
        for f in feat_names: 
            print('{}\t\t{}'.format(f, '\t'.join(map(str, [ranks[est][f] for est in est_list] )))) 
        
        
        # Put the mean scores into a Pandas dataframe
        meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])        
        # Sort the dataframe
        meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
        MyHelper.save2csvfile( "saved/rgs_ds_meanranking.csv", meanplot)    
        
        sdata = meanplot[meanplot['Mean Ranking'] >= minRanking ] 
        print(sdata)

        h=max(6, 0.5*len(sdata))
        # Let's plot the ranking of the features
        sns.factorplot(x="Mean Ranking", y="Feature", data = sdata, kind="bar", size=h, aspect=1, palette='coolwarm')
        plt.savefig('saved/rgs_compare_ranking.png')
                
        plt.show() 
        
        # extract selected features
        selresult=list(sdata['Feature'])                
        print('Selected Features/columns: {}'.format(selresult))
        self.rgs_feature_data = self.rgs_feature_data[selresult]
        self.rgs_feature_cols = selresult
        return
    