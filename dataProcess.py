'''
Created on Feb 26, 2015

@author: Thusitha Chandrapala
'''

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.sparse as sp
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')

def clean(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower()
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower()
        

class DataReader(object):
    '''
    Read and process the csv data to a suitable format
    '''


    def __init__(self):
        self.rawdata = []
        self.exdata_train = []
        self.exdata_test = []  
        self.trdata = []
        self.testdata = []
        self.imputer = []
        
    def readData(self, filename ):
        self.rawdata = pd.read_csv(filename)
    
        
    def factorInd(self,tags):
        #change the labeling into [0,1,2...] Must change to one hot encoding to get best results. (later)
        for itm in tags:
            self.rawdata[itm],lab = self.rawdata[itm].factorize() 
    
    def oneHotSelected(self,oneHotInd):
        df = pd.concat([pd.get_dummies(self.rawdata[col]) for col in oneHotInd], axis=1, keys=self.rawdata.columns)
        self.rawdata = self.rawdata.drop( oneHotInd, axis = 1 )
        self.rawdata = pd.merge(self.rawdata,df, left_index = True, right_index = True )
        
    def remFeatures(self, remLabels):
        self.rawdata = self.rawdata.drop( remLabels, axis = 1 )

    def selFeatures(self,valIdx):
        #Assumes [0] is the project id and all others fare features. so valID has the same number of items as the original data set
        idx = valIdx.nonzero()       
        #testdata
        dataobj = (self.testdata.values[:,idx]).squeeze()    
        self.exdata_test  = dataobj.astype(np.float32)
        #trainingdata
        dataobj = (self.trdata.values[:,idx]).squeeze()    
        self.exdata_train  = dataobj.astype(np.float32)
        
    def imputeMissing(self):
            self.imputer = DataFrameImputer()
            self.rawdata = self.imputer.fit_transform(self.rawdata)           
        
    def fillMissngMean(self, trainOnly=True):#fill missing data with column mean
        if self.exdata_train == []:
            return
        #get the mean from training data
        col_mean = stats.nanmean(self.exdata_train,axis=0)
        if trainOnly==True:
            #fill in training data gaps
            inds = np.where(np.isnan(self.exdata_train))       
            self.exdata_train[inds]=np.take(col_mean,inds[1])
            return
        #fill in testing data gaps 
        inds = np.where(np.isnan(self.exdata_test))       
        self.exdata_test[inds]=np.take(col_mean,inds[1])        
    
    def getLabels(self):
        #get the labels as a numpy array
        dataobj = (self.rawdata.values).squeeze()    
        self.exdata_train  = dataobj.astype(np.float32)
        return self.exdata_train
            
        
    def fillFactorErrors(self):
        #factorizing missing values causes negative 1 indices. This will go through each row and replace by the first column value
        for row in self.exdata_train:
            row[row<0] = row[0]
              
     
     
    def getcharacters(self):
        #get the number of characters in the essay
        nitems = self.rawdata.shape[0]
        nchars = np.zeros([nitems,1],int)
        for i in range(nitems):
            es = self.rawdata['essay'][i]
            if pd.isnull(es):
                nchars[i] = 0
            else:
                nchars[i]=len(es)
            
        self.rawdata['elength'] = nchars    
        

            
        
        
        
    def processEssay(self, testidx, trainidx):
        #process essay
        self.rawdata['essay'] = self.rawdata['essay'].apply(clean)
        self.trdata = self.rawdata['essay'].ix[trainidx]
        self.testdata = self.rawdata['essay'].ix[testidx]
        trainessay = np.array(self.trdata.fillna('Missing'))
        testessay = np.array(self.testdata.fillna('Missing'))
        tfidfEs = TfidfVectorizer(min_df=4,  max_features=500)
        tfidfEs.fit(trainessay)
        #=======================================================================
        # #process need statement
        # self.rawdata['need_statement'] = self.rawdata['need_statement'].apply(clean)
        # self.trdata = self.rawdata['need_statement'].ix[trainidx]
        # self.testdata = self.rawdata['need_statement'].ix[testidx]
        # trainneedst = np.array(self.trdata.fillna('Missing'))
        # testneedst= np.array(self.testdata.fillna('Missing'))
        # tfidfNs = TfidfVectorizer(min_df=3,  max_features=20)
        # tfidfNs.fit(trainneedst)
        #  
        # #process short desc
        # self.rawdata['short_description'] = self.rawdata['short_description'].apply(clean)
        # self.trdata = self.rawdata['short_description'].ix[trainidx]
        # self.testdata = self.rawdata['short_description'].ix[testidx]
        # trainshortd = np.array(self.trdata.fillna('Missing'))
        # testshortd= np.array(self.testdata.fillna('Missing'))
        # tfidfSd = TfidfVectorizer(min_df=3,  max_features=20)
        # tfidfSd.fit(trainshortd)
        # 
        # self.exdata_train = sp.hstack((tfidfEs.transform(trainessay),tfidfNs.transform(trainneedst),tfidfSd.transform(trainshortd) ))
        # self.exdata_test =  sp.hstack((tfidfEs.transform(testessay),tfidfNs.transform(testneedst),tfidfSd.transform(testshortd) ))
        #=======================================================================
        self.exdata_train = tfidfEs.transform(trainessay) #only use the essay
        self.exdata_test =  tfidfEs.transform(testessay)
        
        
        
    def calcDateAdj(self):
        self.testdata['date_posted'] = pd.to_datetime(self.testdata['date_posted'])
        b = self.testdata.date_posted.max().value
        a = self.testdata.date_posted.min().value
        vals = np.zeros([self.testdata.shape[0],])
        
        for idx, itm in enumerate(self.testdata.date_posted):
            val = itm.value
            fac = 1- 0.5/(b-a)*(b-val)
            vals[idx] = fac
        return vals

    def getMonth(self):
        self.rawdata['date_posted'] = pd.to_datetime(self.rawdata['date_posted'])
        months = np.zeros([self.rawdata.shape[0],])
        for idx, itm in enumerate(self.rawdata.date_posted):
            months[idx] = itm.month
        self.rawdata['month_posted'] = months - 1  
        
        
         
class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)         
    

class DataWriter(object): 
    def __init__(self):
        self.savedf = []
        
    def saveData(self, filename, dataMat , colnames):
        #dataMat is  a list of columns that need to be saved 
        #colnames in a list of column names
        headings = colnames
        xdata_arr = np.array([dataMat[0],dataMat[1]]).T
        savedf = pd.DataFrame(xdata_arr,columns=headings)
        self.savedf =  savedf.convert_objects(convert_numeric=True)           
        self.savedf = self.savedf.set_index('projectid')
        f = file(filename,"wb")    
        self.savedf.to_csv(path_or_buf=f)
        f.close()
    
       
    