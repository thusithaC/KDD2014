'''
Created on Feb 28, 2015

@author: Thusitha Chandrapala
'''
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np

class MLModel(object):
    '''
    The machine learning model
    '''


    def __init__(self):
        self.data_train = [] #[n_samples, n_features]
        self.data_train_ess = []
        self.labels_train = [] #[n_samples, n_outputs]
        self.data_val = [] #probably we will not need a validation set since sklearn seem to handle stuff by itself
        self.labels_val = []
        self.data_test = []
        self.data_test_ess = []
        self.est_labels_test = []
        self.scaler = []   
        self.mlmodel = []
        self.mlmodel2 = []   
        self.mlmodel3 = []  
        self.all_results = []
        self.xtra = []
    
    def preprocessTr(self,method=0):
        '''
        Preprocess training data
        method=0 for zero mean, unit var and method=1 for min-max scaling [0,1]
        '''
        if self.data_train == []:
            return
            if method == 0:
                self.scaler = preprocessing.StandardScaler()
            else:
                self.scaler = preprocessing.MinMaxScaler()
    
            self.data_train = self.scaler.fit_transform(self.data_train)

    def preprocessTs(self,testData=False):
        '''
        testdata=False will treat the validation data only
        '''            
        if not testData:
            self.data_val = self.scaler.transform(self.data_val)
        else:
            self.data_test = self.scaler.transform(self.data_test)

    def trainModel(self,modelType=0):       
        self.mlmodel = [RandomForestClassifier(n_estimators = 10) for i in range(8)]
        #self.mlmodel = [AdaBoostClassifier() for i in range(8)]      
        #self.mlmodel = list[svm.SVC()]*7
        for i in range(0,7):  
            self.mlmodel[i].fit(self.data_train,self.labels_train[:,i+1])
    
    def trainModelSingle(self):
        #self.mlmodel = LogisticRegression()
        #self.mlmodel = AdaBoostClassifier()
        #self.mlmodel = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        self.mlmodel = RandomForestClassifier(n_estimators = 200, n_jobs=3,verbose =1)    
        self.mlmodel.fit(self.data_train,self.labels_train[:,0])     
        
    def trainModelEss(self):
        #self.mlmodel = LogisticRegression()
        self.mlmodel = AdaBoostClassifier()
        #self.mlmodel = AdaBoostClassifier()
        #self.mlmodel = RandomForestClassifier(n_estimators = 200, n_jobs=3,verbose =1)    
        self.mlmodel.fit(self.data_train_ess,self.labels_train[:,0])       

    def trainModelComb(self):
        self.mlmodel2 = LogisticRegression()
        self.mlmodel = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        #self.mlmodel = RandomForestClassifier(n_estimators = 200, n_jobs=3,verbose =1)    
        self.mlmodel.fit(self.data_train,self.labels_train[:,0])
        self.mlmodel2.fit(self.data_train_ess,self.labels_train[:,0])
    
    def trainModelCombNaive(self):
        #self.mlmodel = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4) 
        self.mlmodel = LogisticRegression()    
        self.mlmodel.fit(np.hstack((self.data_train,self.data_train_ess)),self.labels_train[:,0])

    def predictCombNaive(self):   
        set_result1 = self.mlmodel.predict_proba(np.hstack((self.data_test,self.data_test_ess))) 
        self.est_labels_test = set_result1[:,1]
        return self.est_labels_test
        
    def trainModelComb2(self):
        self.mlmodel2 = LogisticRegression()
        self.mlmodel2.fit(self.data_train_ess,self.labels_train[:,0])
        set_result2 = self.mlmodel2.predict_proba(self.data_train_ess)
        self.data_train = np.column_stack((self.data_train,set_result2[:,1]))
        
        #self.mlmodel = AdaBoostClassifier()
        self.mlmodel = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        #self.mlmodel = RandomForestClassifier(n_estimators = 200, n_jobs=3,verbose =1)    
        self.mlmodel.fit(self.data_train,self.labels_train[:,0])

    def trainModelComb4(self):
        ntrain = self.data_train.shape[0]
        self.xtra = 5
        est_prob = np.zeros([ntrain,self.xtra+1]) #for original data, essay and others, which would be fed to a second gb 
        
        self.mlmodel2 = [LogisticRegression() for i in range(self.xtra)]
        for i in range(self.xtra-1):  
            self.mlmodel2[i].fit(self.data_train,self.labels_train[:,i+1])
            set_result =  self.mlmodel2[i].predict_proba(self.data_train)
            est_prob[:,i] = set_result[:,1]
                    
        self.mlmodel2[self.xtra-1].fit(self.data_train_ess,self.labels_train[:,0])
        set_result2 = self.mlmodel2[self.xtra-1].predict_proba(self.data_train_ess)
        est_prob[:,self.xtra-1] = set_result2[:,1]
        
        #self.data_train = np.hstack((self.data_train,est_prob))
        
        #self.mlmodel = AdaBoostClassifier()
        self.mlmodel = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        #self.mlmodel = RandomForestClassifier(n_estimators = 200, n_jobs=3,verbose =1)    
        self.mlmodel.fit(self.data_train,self.labels_train[:,0])
        set_result3 = self.mlmodel.predict_proba(self.data_train)
        est_prob[:,self.xtra] = set_result3[:,1]
        
        #2nd layer GB
        self.mlmodel3 = GradientBoostingClassifier(learning_rate=0.1)
        self.mlmodel3.fit(est_prob,self.labels_train[:,0])
        
    def predictComb4(self):
        ntest = self.data_test.shape[0]
        est_prob = np.zeros([ntest,self.xtra+1])
        for i in range(self.xtra-1):  
            set_result = self.mlmodel2[i].predict_proba(self.data_test)
            est_prob[:,i] = set_result[:,1]   

        set_result = self.mlmodel2[self.xtra-1].predict_proba(self.data_test_ess)
        est_prob[:,self.xtra-1] = set_result[:,1]

        #self.data_test = np.hstack((self.data_test,est_prob))
        
        set_result1 = self.mlmodel.predict_proba(self.data_test)
        est_prob[:,self.xtra] = set_result1[:,1]
        
        #second layer model
        set_result3 = self.mlmodel3.predict_proba(est_prob)
        self.est_labels_test = set_result3[:,1]
        return self.est_labels_test  
    
    def trainModelComb3(self):
        ntrain = self.data_train.shape[0]
        self.xtra = 5
        est_prob = np.zeros([ntrain,self.xtra])
        
        self.mlmodel2 = [LogisticRegression() for i in range(self.xtra)]
        for i in range(self.xtra-1):  
            self.mlmodel2[i].fit(self.data_train,self.labels_train[:,i+1])
            set_result =  self.mlmodel2[i].predict_proba(self.data_train)
            est_prob[:,i] = set_result[:,1]
                    
        self.mlmodel2[self.xtra-1].fit(self.data_train_ess,self.labels_train[:,0])
        set_result2 = self.mlmodel2[self.xtra-1].predict_proba(self.data_train_ess)
        est_prob[:,self.xtra-1] = set_result2[:,1]
        
        self.data_train = np.hstack((self.data_train,est_prob))
        
        #self.mlmodel = AdaBoostClassifier()
        self.mlmodel = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        #self.mlmodel = RandomForestClassifier(n_estimators = 200, n_jobs=3,verbose =1)    
        self.mlmodel.fit(self.data_train,self.labels_train[:,0])
        
    def predictComb3(self):
        ntest = self.data_test.shape[0]
        est_prob = np.zeros([ntest,self.xtra])
        for i in range(self.xtra-1):  
            set_result = self.mlmodel2[i].predict_proba(self.data_test)
            est_prob[:,i] = set_result[:,1]   

        set_result = self.mlmodel2[self.xtra-1].predict_proba(self.data_test_ess)
        est_prob[:,self.xtra-1] = set_result[:,1]

        self.data_test = np.hstack((self.data_test,est_prob))
        set_result1 = self.mlmodel.predict_proba(self.data_test)
        
        self.est_labels_test = set_result1[:,1]
        return self.est_labels_test       
        
    def predictComb2(self):   
        set_result = self.mlmodel2.predict_proba(self.data_test_ess)
        self.data_test = np.column_stack((self.data_test,set_result[:,1]))
        set_result1 = self.mlmodel.predict_proba(self.data_test)
        
        self.est_labels_test = set_result1[:,1]
        return self.est_labels_test        
        
    def predictComb(self):   
        set_result1 = self.mlmodel.predict_proba(self.data_test)
        set_result2 = self.mlmodel2.predict_proba(self.data_test_ess)
        self.est_labels_test = 0.5*set_result1[:,1]+ 0.5*set_result2[:,1]
        return self.est_labels_test
                
    def predictSingle(self):   
        set_result = self.mlmodel.predict_proba(self.data_test)
        self.est_labels_test = set_result[:,1]
        return set_result
    
    def predictSingleEss(self):   
        set_result = self.mlmodel.predict_proba(self.data_test_ess)
        self.est_labels_test = set_result[:,1]
        return set_result
        
    def predict(self):
        '''
        testdata=False will treat the validation data only
        '''  
        #est_labels = self.mlmodel.predict(self.data_test).astype(bool)
        ntest = self.data_test.shape[0]
        set_result = np.zeros([ntest,])
        est_labels = np.zeros([ntest,7])
        for i in range(0,7):
            set_result =  self.mlmodel[i].predict_proba(self.data_test) 
            est_labels[:,i] = set_result[:,1]
        
        index = 0
        for row in est_labels:
            set_result[index] = (row[0]*row[1]*row[2]*row[3]) * ( row[4] + row[5] + row[6])  
            index+=1 
        self.est_labels_test = set_result 
        self.all_results = est_labels  
        return self.est_labels_test
        
    def splitData(self, trainData, trainLabels):
        cross_validation.train_test_split(trainData,trainLabels)
        
        
    def doXValidation(self,nxvals, targetNum):
        #model = RandomForestClassifier(n_estimators = 10)
        model = GradientBoostingClassifier(learning_rate=0.2,subsample=0.4)
        return cross_validation.cross_val_score(model, self.data_train, self.labels_train[:,0], cv=nxvals)
    

    