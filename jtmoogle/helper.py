##
## This is prepared for Udacity Machine Learning Engineer Nanodegree online class
## Author: jtmoogle @github.com All Rights Reserved
## Date:Feb-Apr 2018
## 
## This file contains 
## 1. MyLogger - logging to a file or console output for purpose of debug, info, error 
## 2. MyHelper - common functions to load dataset, print out statistic summary, save to result to a file
## 
## contact info: jtmoogle@gmail.com for question, or suggestion
## 
import os.path
os.chdir('I:/_githup/capstone-report/')  # this source is at jtmoogle

#os.chdir('/_githup/capstone/')  # this source is at jtmoogle
#print("Currnet wdir={}   ".format(os.getcwd()))

import gc
import IPython
from IPython.display import display
import json, codecs
import logging  
from matplotlib import pyplot as plt
import numpy as np
import os.path
import pandas as pd
from platform import python_version
import seaborn as sns
import sklearn as sk
from sklearn.externals import joblib
import sys 
import tensorflow as tf
import time
import warnings
warnings.filterwarnings("ignore", category=Warning)

class MyLogger(object):
    """Logger for output ton console or to file
    """
    def __init__(self, logname, lvl=logging.DEBUG):
        # global logger to console and file
        self.logger = logging.getLogger(logname)
        self.logger.setLevel(lvl)
        
        self.console = logging.StreamHandler()
        self.formatter = self.myformatter()
        self.console.setFormatter(self.formatter)
        self.console.setLevel(lvl)
        self.logger.addHandler(self.console)
        
        # file handler
        ldir=os.path.realpath(".")
        fname = 'exec_log.' + time.strftime('%Y-%m-%d-%H') + ".txt"
        filename = os.path.join(ldir, 'log', fname)
        
        #notnew = os.path.exists(filename)        
        self.fhandler = logging.FileHandler(filename=filename, mode='a')
        print(" Logging to {}".format(filename))
        self.fhandler.setFormatter(self.formatter)
        self.fhandler.setLevel(lvl)
        self.logger.addHandler(self.fhandler)  

    def myformatter(self): return logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt="%H:%M:%S")
    
    #def getlogger(): return self.logger
    # logging preference
    def inf(self, msg): return (self.logger.info(msg))
    def dbx(self, msg): return (self.logger.debug(msg))
    def err(self, msg): return (self.logger.error(msg))
    def now(self): return (time.time())    
    def timetaken(self, starttime): 
        '''
        inputs: logging level, starttime
        output: elapsed time 
        '''  
        elapsed_time = self.now() - starttime
        return("Time taken at {0:.2f} seconds".format(elapsed_time))
    
class MyHelper(object):
    '''helper contain common fucntions to clean memory, load dataset, generate statistic report
    '''
    def __init__(self):
        None
        
    @staticmethod
    def printversion():       
        # Software, API version
        print( '-'*30 )
        print( '--> IPython version: {}'.format(IPython.__version__ ))
        print( '--> numpy version: {}'.format(np.__version__))
        print( '--> pandas version: {}'.format(pd.__version__))
        print( '--> python version: {}'.format(python_version()))
        print( '--> scikit-learn version: {}'.format( sk.__version__))
        print( '--> sys version: {} '.format(sys.version ))
        print( '--> tensorflow version: {} '.format(tf.__version__ ))
        print( '-'*30 )
        
    @staticmethod
    def clean_mem():
        '''    release unreferenced memory with gc.collect()    '''
        gc.collect()

    # define function to load train, test, and validation datasets
    @staticmethod
    def load_dataset(nativepath, convert2Excel=False ):
       '''read data from the file, load data to dataset, and return the dataset
       input: nativepath - data filename with extension.
       output: data frame of data read from the file
       '''
       try:
           if (os.path.exists(nativepath)):
               # if this is csv file, and look for '.ds.csv' which has been tranformed previously 
               if (os.path.splitext(nativepath)[1]).lower() in [".csv"]:  
                   csvpath = '{}.ds.{}'.format(nativepath, 'csv')                
                   if (os.path.exists(csvpath)):
                       print('Load dataset path={}'.format(nativepath))
                       ds = pd.read_csv(csvpath, encoding = "ISO-8859-1", dtype={"LEAID": str },  quotechar='"')
                       return ds
                   # transoform money string by removing $, from string, and save back as csv file
                   ds = pd.read_csv(nativepath, encoding = "ISO-8859-1", dtype={"LEAID": str },  quotechar='"')
                   print('Convert dollar string to value, save to {}'.format(csvpath))            
                   dcols = {"Med_HHD_Inc_ACS_08_12", "Med_HHD_Inc_ACSMOE_08_12", "Aggregate_HH_INC_ACS_08_12",  "Aggregate_HH_INC_ACSMOE_08_12", "Med_House_value_ACS_08_12",  "Med_House_value_ACSMOE_08_12", "Aggr_House_Value_ACS_08_12", "Aggr_House_Value_ACSMOE_08_12"}         
                   for col in dcols:  
                       ds[col] = ds[col].str.replace('$', '').str.replace(',','')
                   ds.to_csv(csvpath , sep=',')                      
                   # if conver to Excell, save to file named <path>.ds.exls 
                   if convert2Excel: 
                       path = '{}.ds.{}'.format(nativepath, 'xlsx')                
                       print('  save to excel file={}'.format(path))
                       writer = pd.ExcelWriter(path)
                       ds.to_excel(writer, index = False)
                       writer.save()
                   print('Load dataset path={}'.format(csvpath))
                   ds = pd.read_csv(csvpath, encoding = "ISO-8859-1", dtype={"LEAID": str },  quotechar='"')
                   return ds
                       
               # expect to load excel file format    
               path = nativepath
               print('Load dataset path={}'.format(path)) 
               # excel files
               ds = pd.read_excel( path, encoding = "ISO-8859-1" )
               return ds
       except Exception as e:
           print('Failed to load data file: {} '.format(e))
           return False

    # basic statistics output
    @staticmethod
    def stats(dataset, infotype= 1, detailtype=1 ):
        '''Genreate basic statistics report
        '''
        if (infotype > 0): display( 'Statistics: dataset has {} (rows) samples with  {} (columns) features each'.format(dataset.shape[0], dataset.shape[1]))
        if (infotype > 1): display( dataset.head(3) )
        if (infotype > 2): display( 'corr() -->', dataset.corr())        
        #if (infotype > 3): display( 'cov() -->', dataset.cov())
        #if (detailtype > 0):  
            #print('Data column={}'.format(dataset.columns))   
            #print( 'Data column={}'.format(dataset.columns.tolist())) 
        if (detailtype == 0): 
            print( 'Unique count for Scholl district={} \nState={} \tcount={} '.format(
                    len(dataset['District.ID'].unique()), len(dataset['State'].unique()),
                    len(dataset['County'].unique())   ))
            print( 'COHORT average count')
            display(dataset.mean()[0:10])
            print( 'COHORT total count')
            display(dataset.sum()[1:11])
            print( 'Geography and Population Total for\n  Tot_Population_CEN_2010\t{:,d}\n  RURAL_POP_CEN_2010\t {:,d}\n  URBANIZED_AREA_POP_CEN_2010\t{:,d}\n  URBAN_CLUSTER_POP_CEN_2010\t{:,d} '.format(
                    sum(dataset['Tot_Population_CEN_2010'].fillna(0)), 
                    sum(dataset['RURAL_POP_CEN_2010'].fillna(0)), 
                    sum(dataset['URBANIZED_AREA_POP_CEN_2010'].fillna(0)), 
                    sum(dataset['URBAN_CLUSTER_POP_CEN_2010'].fillna(0)) ))            
            print( 'Gender Total for\n  Males_CEN_2010\t{:,d}\n  Females_CEN_2010\t{:,d} '.format(
                    sum(dataset['Males_CEN_2010'].fillna(0)), 
                    sum(dataset['Females_CEN_2010'].fillna(0)) ))
            print( 'Age Total for\n   Pop_under_5_CEN_2010\t {:,d}\n   Pop_5_17_CEN_2010\t {:,d}\n   Pop_18_24_CEN_2010\t {:,d}\n   Pop_25_44_CEN_2010\t {:,d}\n   Pop_45_64_CEN_2010\t {:,d}\n   Pop_65plus_CEN_2010\t{:,d}'.format(
                    sum(dataset['Pop_under_5_CEN_2010'].fillna(0)), 
                    sum(dataset['Pop_5_17_CEN_2010'].fillna(0)), 
                    sum(dataset['Pop_18_24_CEN_2010'].fillna(0)),
                    sum(dataset['Pop_25_44_CEN_2010'].fillna(0)), 
                    sum(dataset['Pop_45_64_CEN_2010'].fillna(0)),
                    sum(dataset['Pop_65plus_CEN_2010'].fillna(0))  ))
            print( 'English Language Speaks Total for\n  ENG_VW_ACS_08_12\t {:,.0f}\n  ENG_VW_SPAN_ACS_08_12\t {:,.0f}\n  ENG_VW_INDO_EURO_ACS_08_12\t {:,.0f}\n  ENG_VW_API_ACS_08_12\t {:,.0f}\n  ENG_VW_OTHER_ACS_08_12\t {:,.0f}'.format(
                    sum(dataset['ENG_VW_ACS_08_12'].fillna(0)) ,
                    sum(dataset['ENG_VW_SPAN_ACS_08_12'].fillna(0)),
                    sum(dataset['ENG_VW_INDO_EURO_ACS_08_12'].fillna(0)), 
                    sum(dataset['ENG_VW_API_ACS_08_12'].fillna(0)),
                    sum(dataset['ENG_VW_OTHER_ACS_08_12'].fillna(0)) ))
            print( 'Family Education Total for\n   Not_HS_Grad_ACS_08_12\t {:,.2f}\n   College_ACS_08_12\t{:,.2f}'.format(
                    sum(dataset['Not_HS_Grad_ACS_08_12'].fillna(0)), 
                    sum(dataset['College_ACS_08_12'].fillna(0))  ))
            print( 'Family Background Total for\n  Pov_Univ_ACS_08_12\t {:,.0f}\n  Prs_Blw_Pov_Lev_ACS_08_12\t {:,.0f}\n  Civ_labor_16plus_ACS_08_12\t {:,.0f}\n  Civ_emp_16plus_ACS_08_12\t {:,.0f}\n  Civ_unemp_16plus_ACS_08_12\t {:,.0f}\n  Civ_labor_16_24_ACS_08_12\t{:,.0f}'.format(
                    sum(dataset['Pov_Univ_ACS_08_12'].fillna(0)),
                    sum(dataset['Prs_Blw_Pov_Lev_ACS_08_12'].fillna(0)), 
                    sum(dataset['Civ_labor_16plus_ACS_08_12'].fillna(0)),
                    sum(dataset['Civ_emp_16plus_ACS_08_12'].fillna(0)),
                    sum(dataset['Civ_unemp_16plus_ACS_08_12'].fillna(0)),
                    sum(dataset['Civ_labor_16_24_ACS_08_12'].fillna(0)),
                    sum(dataset['Civ_labor_16plus_ACS_08_12'].fillna(0))  ))
            print( 'Income Total for\n  PUB_ASST_INC_ACS_08_12\t {:,.0f}\n  Med_HHD_Inc_ACS_08_12\t {:,.0f}\n  Aggregate_HH_INC_ACS_08_12\t {:,.0f}'.format(
                    sum(dataset['PUB_ASST_INC_ACS_08_12'].fillna(0)),
                    sum(dataset['Med_HHD_Inc_ACS_08_12'].fillna(0)), 
                    sum(dataset['Aggregate_HH_INC_ACS_08_12'].fillna(0)) ))
            print( 'Other Total for\n  Born_US_ACS_08_12\t {:,.0f}\n  Born_foreign_ACS_08_12\t {:,.0f}\n  US_Cit_Nat_ACS_08_12\t {:,.0f}\n  NON_US_Cit_ACS_08_12\t {:,.0f}\n  MrdCple_Fmly_HHD_CEN_2010\t {:,.0f}\n  Not_MrdCple_HHD_CEN_2010\t {:,.0f}\n  Female_No_HB_CEN_2010\t{:,.0f}\n  NonFamily_HHD_CEN_2010\t{:,.0f}'.format(
                    sum(dataset['Born_US_ACS_08_12'].fillna(0)),
                    sum(dataset['Born_foreign_ACS_08_12'].fillna(0)), 
                    sum(dataset['US_Cit_Nat_ACS_08_12'].fillna(0)),
                    sum(dataset['NON_US_Cit_ACS_08_12'].fillna(0)),
                    sum(dataset['MrdCple_Fmly_HHD_CEN_2010'].fillna(0)),
                    sum(dataset['Not_MrdCple_HHD_CEN_2010'].fillna(0)),
                    sum(dataset['Female_No_HB_CEN_2010'].fillna(0)),  
                    sum(dataset['NonFamily_HHD_CEN_2010'].fillna(0))   ))

        if (detailtype > 1): print(dataset.dtypes)
        if (detailtype > 2): 
            smry= dataset.describe()
            display('-Data Summary->', np.round( smry, 2))
           # display( '  Mean=Average & Median=50% percentile -> ', np.round(smry.loc[['mean','50%']],2))                    
        if (detailtype > 3):
            str_list = [] # empty list to contain columns with strings (words)
            for colname, colvalue in dataset.iteritems():
                if type(colvalue[1]) == str:
                     str_list.append(colname)
            # Get to the numeric columns by inversion            
            num_list = dataset.columns.difference(str_list) 
            # Create Dataframe containing only numerical features
            dsnum = dataset[num_list]
            f, ax = plt.subplots(figsize=(16, 12))
            plt.title('Pearson Correlation of features')
            # Draw the heatmap using seaborn
            #sns.heatmap(house_num.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="PuBuGn", linecolor='k', annot=True)
            sns.heatmap(dsnum.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="cubehelix", linecolor='k', annot=True)

    @staticmethod
    def smry2file(savefname, mdl_result): 
        with open(savefname, 'w') as f:
            print( 'Save to a file {}'.format(savefname))
            if savefname.lower().endswith(('.csv')) :
                f.write(mdl_result.summary().as_csv())
            if savefname.lower().endswith(('.txt')) :
                f.write(mdl_result.summary().as_text())        
            f.close()


    @staticmethod
    def save2dsfile(savefname, mdl_result):         
        with open(savefname, 'w+') as f:
            print( 'Save to a file {}'.format(savefname))
            print(mdl_result, file=f)
            path = '{}.ds.{}'.format(savefname, 'csv') 
            mdl_result.to_csv(path, sep=',')
            f.close()

    @staticmethod
    def save2file(savefname, mdl_result):         
        ''' save the model result to a txt file 
        '''        
        with open(savefname, 'w+') as f:
            print( 'Save to a file {}'.format(savefname))
            print(mdl_result, file=f)
            f.close()

    @staticmethod
    def save2csvfile(savefname, mdl_result):        
        ''' save the model result to a csv file 
        '''        
        print( 'Save to a file {}'.format(savefname))
        mdl_result.to_csv(savefname, sep=',')

    @staticmethod
    def save2jsnfile(savefname, mdl_result):        
        ''' save the model result to a jason file 
        '''
        with open(savefname, 'wb') as f:
            print( 'Save to a file {}'.format(savefname))
            json.dump(mdl_result, codecs.getwriter('utf-8')(f), ensure_ascii=False)
            f.close()

    @staticmethod
    def mdl2pklfile(savefname, mdl_result):      
        ''' save the model result to a pkl file
        '''
        print( 'Save model to a file {}'.format(savefname))
        joblib.dump(mdl_result, savefname)   # pickle format
        return
            
    @staticmethod
    def load_pklfile(fname):         
        ''' read pkl file and return model result
        '''
        print( 'Read fromm pkl file {}'.format(fname))
        mdl_result = joblib.load(fname)   # pickle format
        return mdl_result

    @staticmethod
    def load_jsnfile(fname):         
        ''' read jason file, and return dataset
        '''
        print( 'Read fromm a file {}'.format(fname))
        with open(fname) as f:
            ds = json.load(f)
            f.close()
        return ds
    
    @staticmethod
    def load_csvfile(fname):      
        ''' read csv file return dataset
        '''
        print( 'Read from a file {}'.format(fname))
        ds = pd.read_csv(fname)            
        return ds
