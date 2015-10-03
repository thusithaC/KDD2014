'''
Created on Feb 26, 2015
Will try to separately process the essay and other feature and combine later. 
@author: Thusitha Thusitha Chandrapala
'''
import dataProcess as dp
import learnModel as lm
import numpy as np
from scipy import sparse
import cPickle as pickle


PROCESSESSAY = 1
ONEHOTSELECTED = 0


def findTeacherHistory(df_feat, df_targets):
    '''
    Find the exciting projects, and get their ids. use the ids to find the respective teachers and construct a set of good teachers. construct a feature vector, is_good_teacher for the training and test data combined. 
        '''
    dd = df_targets.loc[(df_targets["is_exciting"] == 1)]
    exciting_id = list(dd.index)
    dg = df_feat.ix[exciting_id]
    good_teachers = set(dg.teacher_acctid)
    nitems = df_feat.shape[0]
    is_good_teacher = np.zeros(nitems,)
    for idx, itm in enumerate(df_feat.teacher_acctid):
        is_good_teacher[idx] = float(itm in good_teachers)
    df_feat['is_good_teacher'] = is_good_teacher
    return df_feat 
        


def readProcessData():
    #working with the essays
    if PROCESSESSAY==1:
        de = dp.DataReader()
        de.readData("C:\Work\Workspace\KDD2014Project\data\essays.csv")
        de.getcharacters()
    
    
    # training results
    do = dp.DataReader()
    do.readData("C:\Work\Workspace\KDD2014Project\data\outcomes.csv")
    do.rawdata = do.rawdata.set_index('projectid') #set the index to be the project id
    train_index = do.rawdata.index.values.tolist() #index of training data 
    #have to fill in the blanks of the dataset based on the main outcome.
    
    #get the labels as a numpy array and fill in the nans
    #Index([u'is_exciting', u'at_least_1_teacher_referred_donor!', u'fully_funded!', u'at_least_1_green_donation!', u'great_chat!', u'three_or_more_non_teacher_referred_donors~', u'one_non_teacher_referred_donor_giving_100_plus~', u'donation_from_thoughtful_donor~', u'great_messages_proportion', u'teacher_referred_count', u'non_teacher_referred_count'], dtype='object')
    labels = [u'is_exciting', u'at_least_1_teacher_referred_donor', u'fully_funded', u'at_least_1_green_donation', u'great_chat', u'three_or_more_non_teacher_referred_donors', u'one_non_teacher_referred_donor_giving_100_plus', u'donation_from_thoughtful_donor']
    do.factorInd(labels)
    

    
    # All the features train+test
    dr = dp.DataReader()
    dr.readData("C:\Work\Workspace\KDD2014Project\data\projects.csv")
    removed_list = list(dr.rawdata[dr.rawdata.date_posted<='2010-01-01'].projectid)
    dr.rawdata = dr.rawdata[dr.rawdata.date_posted>'2010-01-01']
    
    train_index = list(set(train_index)-set(removed_list))
    #drop the date column
    #dr.rawdata.drop(['date_posted'],inplace=True,axis=1)
    
    
    #select the labels and process the targets
    do.rawdata = do.rawdata.ix[train_index]
    do.getLabels() 
    do.fillMissngMean(trainOnly=True) #because of factorization, the -1 means a missing vaue. Have to replace with the value at the first row.
    do.fillFactorErrors()
    
    #u'projectid', u'teacher_acctid', u'schoolid', u'school_ncesid', u'school_latitude', u'school_longitude', u'school_city', u'school_state', 
    #u'school_zip', u'school_metro', u'school_district', u'school_county', u'school_charter', u'school_magnet', u'school_year_round', u'school_nlns', 
    #u'school_kipp', u'school_charter_ready_promise', u'teacher_prefix', u'teacher_teach_for_america', u'teacher_ny_teaching_fellow', #u'primary_focus_subject', u'primary_focus_area',
    # u'secondary_focus_subject', u'secondary_focus_area', u'resource_type', u'poverty_level',
    # u'grade_level', u'fulfillment_labor_materials', u'total_price_excluding_optional_support', u'total_price_including_optional_support', u'students_reached', 
    #u'eligible_double_your_impact_match', u'eligible_almost_home_match', u'date_posted']
    
    # change the encoding of some features in the whole data set    
    labels = [u'teacher_acctid', u'schoolid', u'school_ncesid',u'school_city', u'school_state', u'school_zip', u'school_metro', u'school_district',\
    u'school_county', u'school_charter', u'school_magnet', u'school_year_round', u'school_nlns', u'school_kipp',\
    u'school_charter_ready_promise', u'teacher_prefix', u'teacher_teach_for_america', u'teacher_ny_teaching_fellow',\
    u'primary_focus_subject', u'primary_focus_area', u'secondary_focus_subject', u'secondary_focus_area', u'resource_type',\
    u'poverty_level', u'grade_level', u'eligible_double_your_impact_match', u'eligible_almost_home_match']
    

    
    dr.imputeMissing()
    #include month as a feature
    #dr.getMonth()
    dr.factorInd(labels)
    
    #divide into training and testing data
    dr.rawdata = dr.rawdata.set_index('projectid') #set the index to be the project id
    all_index = dr.rawdata.index.values.tolist() #index of training data  
    
    #create new features with teacher information
    #dr.rawdata = findTeacherHistory(dr.rawdata, do.rawdata)
    
    #extract the relevant columns from the essay file
    if PROCESSESSAY==1:
        de.rawdata = de.rawdata.set_index('projectid')
        de.rawdata = de.rawdata.ix[all_index] #make sure the items are in the same order in the two files
        dr.rawdata['elength'] = de.rawdata['elength']

    # drop the columns and add the one hot encoded versions 
    if ONEHOTSELECTED==1:
        labels_categorical = [u'school_metro', u'primary_focus_subject', u'primary_focus_area', u'resource_type',\
        u'poverty_level', u'grade_level']
        dr.oneHotSelected(labels_categorical)
        
    

    #Index([u'teacher_acctid', u'schoolid', u'school_ncesid', u'school_latitude', u'school_longitude', u'school_city', u'school_state', u'school_zip', u'school_metro', u'school_district', u'school_county', u'school_charter', u'school_magnet', u'school_year_round', u'school_nlns', u'school_kipp', u'school_charter_ready_promise', u'teacher_prefix', u'teacher_teach_for_america', u'teacher_ny_teaching_fellow', u'primary_focus_subject', u'primary_focus_area', u'secondary_focus_subject', u'secondary_focus_area', u'resource_type', u'poverty_level', u'grade_level', u'fulfillment_labor_materials', u'total_price_excluding_optional_support', u'total_price_including_optional_support', u'students_reached', u'eligible_double_your_impact_match', u'eligible_almost_home_match'], dtype='object')
    #             0                1             2                     3                4                      5             6            7                  8                       9              10               11                12                13                 14              15                     16                            17                 18                               19                            20                    21                        22                  23                         24                   25                    26            27                                 28                                      29                                       30                         31                                     32

    deactivate_labels = ['teacher_acctid', 'schoolid' ,'school_ncesid', 'school_latitude','school_longitude','secondary_focus_subject', 'secondary_focus_area','school_state']
    dr.remFeatures(deactivate_labels)
    
    
    #divide the training and testing sets
    test_index = list(set(all_index) - set(train_index))   
    dr.trdata = dr.rawdata.ix[train_index]
    dr.testdata = dr.rawdata.ix[test_index]
    
    #date processing
    datefac = dr.calcDateAdj()
    #drop the dates
    dr.trdata.drop(['date_posted'],inplace=True,axis=1)
    dr.testdata.drop(['date_posted'],inplace=True,axis=1)
    
    #etract essay data
    if PROCESSESSAY==1:
        de.processEssay(test_index,train_index)
        esfeats_train = sparse.csr_matrix(de.exdata_train)
        esfeats_test = sparse.csr_matrix(de.exdata_test)
 
    ncols = dr.trdata.columns.size 
    active_features = np.ones([ncols,], bool)
    dr.selFeatures(active_features)

      
    #dr.fillMissngMean()
    
    data_train = dr.exdata_train #extracted data which has been treated for missing values
    data_test = dr.exdata_test 
    targets = do.exdata_train
    id_test = dr.testdata.index.values.tolist()
    
    #===========================================================================
    # #combine essay data
    # if PROCESSESSAY==1:
    #     data_train = sc.sparse.hstack((data_train,esfeats_train)).todense()
    #     data_test = sc.sparse.hstack((data_test,esfeats_test)).todense()
    #     data_train = np.squeeze(np.asarray(data_train))
    #     data_test = np.squeeze(np.asarray(data_test))
    # 
    #===========================================================================
    
    #save data into a binary file
    f = file("C:\Work\Workspace\KDD2014Project\data\data.bin","wb")
    np.save(f, data_train)
    np.save(f, data_test)
    np.save(f, targets)
    np.save(f, id_test)
    np.save(f, datefac)
    f.close()
    
    #save the sparse data sep
    if PROCESSESSAY==1:
        with open('C:\\Work\\Workspace\\KDD2014Project\\data\\data_ess.dat', 'wb') as outfile:
            pickle.dump(esfeats_train, outfile, pickle.HIGHEST_PROTOCOL)
            pickle.dump(esfeats_test, outfile, pickle.HIGHEST_PROTOCOL)
    
    print "Data obtained... !" 

def saveResults(dataMat):
    ds = dp.DataWriter()
    filename = "C:\\Work\\Workspace\\KDD2014Project\\data\\test_res.csv"
    colnames = ['projectid','is_exciting']
    ds.saveData(filename, dataMat, colnames)
       
if __name__ == '__main__':
    
    #readProcessData() 
    f = file("C:\Work\Workspace\KDD2014Project\data\data.bin","rb")
    data_train = np.load(f)
    data_test = np.load(f)
    targets = np.load(f)
    id_test = np.load(f)
    datefac = np.load(f)
    f.close()
    
    with open('C:\\Work\\Workspace\\KDD2014Project\\data\\data_ess.dat', 'rb') as infile:
        esfeats_train = pickle.load(infile)
        esfeats_test = pickle.load(infile)
    
    print("Data loaded")
    
    lmodel = lm.MLModel()
    lmodel.data_test = data_test
    lmodel.data_test_ess = esfeats_test.todense()
    #lmodel.data_train = data_train[0:1000,:]
    #lmodel.labels_train = targets[0:1000,:]
    lmodel.data_train = data_train
    lmodel.data_train_ess = esfeats_train.todense()
    lmodel.labels_train = targets
    
    
    
    lmodel.preprocessTr(method=0)
    
    validations = 1
    trainNtest = 0
    trainSingle = 0
    trainSingleEss = 0
    trainSingleSep = 0
    trainSingleComb = 0
    trainCombNaive = 0
    trainSingleComb2 = 0
    trainSingleComb3 = 0
    trainSingleComb4 = 0
     
    if validations == 1:
        val_results = lmodel.doXValidation(3, 2)
        print(val_results)
        
    if trainNtest==1:
        lmodel.trainModel(0)
        print("Training complete...")
        lmodel.predict()       
        saveResults([id_test,lmodel.est_labels_test])
        print np.sum(lmodel.est_labels_test)
        print('results Saved!')
        
    if trainSingle == 1:
        lmodel.trainModelSingle()
        print("Training complete...")
        lmodel.predictSingle()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')  
        
    if trainSingleEss == 1:
        lmodel.trainModelEss()
        print("Training complete...")
        lmodel.predictSingleEss()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')  
        
    if trainSingleSep == 1:
        lmodel.trainModelSingle()
        print("Training complete...")
        lmodel.predictSingle()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')       
           
    if trainCombNaive == 1:
        lmodel.trainModelCombNaive()
        print("Training complete...")
        lmodel.predictCombNaive()
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')  
                           
    if trainSingleComb == 1:
        lmodel.trainModelComb()
        print("Training complete...")
        lmodel.predictComb()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')   
          
    if trainSingleComb2 == 1:
        lmodel.trainModelComb2()
        print("Training complete...")
        lmodel.predictComb2()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')   
        
    if trainSingleComb3 == 1:
        lmodel.trainModelComb3()
        print("Training complete...")
        lmodel.predictComb3()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')      
    
    if trainSingleComb4 == 1:
        lmodel.trainModelComb4()
        print("Training complete...")
        lmodel.predictComb4()    
        saveResults([id_test,lmodel.est_labels_test])
        print('results Saved!')               