# -*- coding: utf-8 -*-
##
## This is prepared for Udacity Machine Learning Engineer Nanodegree online class
## Author: jtmoogle @github.com All Rights Reserved
## Date:Feb-Apr 2018
## 

import os.path
os.chdir('I:/_githup/capstone-report/')  # this source is at jtmoogle
#print("Currnet wdir={}   ".format(os.getcwd()))

from jtmoogle.helper import MyHelper
from jtmoogle.hsgraduation import HSGraduation

def runme( cls=2, rgs=0) :
    
    MyHelper.printversion()
    
    # create high school graduation class
    hs = HSGraduation()
    
    # Invoke methods for Classification
    if cls==1:  
        print("Classification: complete dataset")    
        rawdata = hs.load_gradcensus()
        MyHelper.stats(rawdata, 1, 0)
        hs.plot_cls_gradcensus()
        hs.preproc_cls_data()
        hs.cls_feature_sel()
        hs.cls_stats()
        hs.cls_acc_featimportance()
        hs.create_cls_sample()
        hs.cls_visual_benchmark( )
        #hs.cls_ftest()
        #hs.cls_pca()
    elif cls==2 : 
        print("Classification: 300 sample dataset")    
        rawdata = hs.load_gradcensus()
        hs.plot_cls_gradcensus()
        hs.preproc_cls_data()
        hs.cls_feature_sel()
        hs.cls_stats()
        hs.cls_acc_featimportance(300)
        hs.create_cls_sample()
        hs.cls_visual_benchmark( )    
    elif cls==3:  # all methods
        rawdata = hs.load_gradcensus()
        #MyHelper.stats(rawdata, 1, 0)
        hs.plot_cls_gradcensus()
        hs.preproc_cls_data()
        hs.cls_feature_sel()        
        hs.compare_cls_featranking(minRanking=0.2)
        hs.cls_stats()
        hs.create_cls_sample()
        hs.cls_acc_featimportance()
        hs.cls_visual_benchmark( )
    else:
        print("Classification: not running")    
    
    
    # Invoke methods for Regression
    if rgs==1: 
        print("Regression: complete dataset")        
        rawdata = hs.load_gradcensus()
        hs.plot_rgs_gradcensus()
        hs.preproc_rgs_data()
        hs.rgs_feature_sel()
        hs.rgs_stats()
        hs.rgs_r2_featimportance()
        hs.create_rgs_sample()
        hs.rgs_visual_benchmark( )
    elif rgs==2:
        print("Regression: 300 sample dataset")    
        rawdata = hs.load_gradcensus()
        hs.plot_rgs_gradcensus()
        hs.preproc_rgs_data()
        hs.rgs_feature_sel()
        hs.rgs_stats()
        hs.rgs_r2_featimportance(300)
        hs.create_rgs_sample()
        hs.rgs_visual_benchmark( )
    elif rgs==3:  # all methods
        rawdata = hs.load_gradcensus()
        hs.plot_rgs_gradcensus()
        hs.preproc_rgs_data()
        hs.rgs_feature_sel()
        hs.compare_rgs_featranking(minRanking=0.2)
        hs.rgs_stats()
        hs.create_rgs_sample()
        hs.rgs_r2_featimportance()
        hs.rgs_visual_benchmark( )
    else:
        print("Regression: not running")   

    return