import pandas as pd
import numpy as np
import sys
from RemoveNans import *
from RemoveDuplicates import *
from NormalizeColumns import *
from CategoricalColumns import *
#from LogisticRegression import *
from StemmerCleaning import *
from GrammerCorrection import *
from SQLCols import *
from Impute import *
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from MaskPII import *
from maskpii2 import *

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.top = None
        self.down = None
        self.data = None

def treer(tree_object, dict1, arrayop, currentnode):
    tree_object.data = dict1[currentnode]
    tree_object.left = 'Null'
    tree_object.right = 'Null'
    tree_object.top = 'Null'
    tree_object.down = 'Null'
    
    nodes = np.where(arrayop[:,0] == currentnode)
    if len(nodes[0]) == 0:
        return tree_object
    for i in range(0,len(nodes[0])):
        if i == 0:
            tree_object.left = Tree()
            treer(tree_object.left, dict1, arrayop, int(arrayop[nodes[0][i],1]))
        if i == 1:
            tree_object.right = Tree()
            treer(tree_object.right, dict1, arrayop, int(arrayop[nodes[0][i],1]))
        if i == 2:
            tree_object.top = Tree()
            treer(tree_object.top, dict1, arrayop, int(arrayop[nodes[0][i],1]))
        if i == 3:
            tree_object.down = Tree()
            treer(tree_object.down, dict1, arrayop, int(arrayop[nodes[0][i],1]))
    return tree_object

def treeOperate(tree_object, taskList, masterlist):
    taskList.append(tree_object.data)
    if tree_object.left == 'Null' and tree_object.right == 'Null' and tree_object.top == 'Null' and tree_object.down == 'Null':
        masterlist.append(taskList.copy())
    if tree_object.left != 'Null':
        treeOperate(tree_object.left, taskList, masterlist)
    if tree_object.right != 'Null':
        treeOperate(tree_object.right, taskList, masterlist) 
    if tree_object.top != 'Null':
        treeOperate(tree_object.top, taskList, masterlist)
    if tree_object.down != 'Null':
        treeOperate(tree_object.down, taskList, masterlist)
    taskList.pop()
    return masterlist

def preProcessJson():
    f = open("data.json", "r")
    str = f.read()
    f.close()

    liststr = (str[1:-1]).split(',')
    lenliststr = len(liststr)
    dict1 = [0 for i in range(1000)]
    
    start = '@'
    end = '='

    start_s = '+'
    end_s = '>'

    start_e = '>'
    end_e = '"'

    arrayop = np.zeros((1000, 2))
    start_node = 0
    count = 0
    for i in range(0,lenliststr):
        if (liststr[i][1]) == '@':
            dict1[int((liststr[i].split(start))[1].split(end)[0])] = liststr[i][liststr[i].find('=')+1:-1]
            if liststr[i][4:-1].find('Start') != -1:
                strfind = liststr[i][4:-1]
        if (liststr[i][1]) == '+':
            arrayop[count, 0] = (int((liststr[i].split(start_s))[1].split(end_s)[0]))
            arrayop[count, 1] = (int((liststr[i].split(start_e))[1].split(end_e)[0]))
            count  = count + 1

    start_node = dict1.index(strfind)

    tree_object = Tree()
    tree_object = treer(tree_object, dict1, arrayop, start_node)
    taskList = []
    masterlist = []
    masterlist = treeOperate(tree_object, taskList, masterlist)
    return masterlist


def main():
    bestmodel = []
    task_list = preProcessJson()
    
    with open('parameters.json') as json_file:
        parameters_default = json.load(json_file)
    data = []
 
    for j in range(0, len(task_list)):
        for i in task_list[j]:
            if i.find('Select_Data') != -1:
                print(i)
                if os.path.exists(i +'.json'):
                    print(i)
                    with open(i +'.json') as json_file:
                        parameters = json.load(json_file)
                    print(i)
                    filenamedata = 'dataset/' + parameters['filename']
                    print(i)
                    print(filenamedata)
                else:
                    filenamedata = 'dataset/' + parameters_default['filename']

                if filenamedata[-3:] == 'csv':
                    data = pd.read_csv(filenamedata)
                if filenamedata[-3:] == 'txt':
                    f= open(filenamedata,"r+")
                    datatext = f.read()
                    f.close()
                    datadf = {'text':[datatext]}
                    data = pd.DataFrame(datadf)

            if i.find('Remove_Duplications') != -1:
                RemoveDuplicates.duplicateAll(data)
                
            if i.find('Remove_Missing_Data') != -1:
                RemoveNans.removeAll(data)
            
            if i.find('Normalization') != -1:
                if os.path.exists(i +'.json'):
                    with open(i +'.json') as json_file:
                        parameters = json.load(json_file)
                    data_name = parameters['normalization']
                else:
                    data_name = parameters_default['normalization']
                if data_name == 'minmax':
                    NormalizeColumns.normalizeAllMinMax(data)
                else:
                    NormalizeColumns.normalizeAllStdScl(data)
                    
            if i.find('Categorical_to_Numeric') != -1:
                CategoricalColumns.convertAll(data)
                
            if i.find('SQL') != -1:
                data.to_csv('datadummy.csv')
                if os.path.exists(i +'.json'):
                    with open(i +'.json') as json_file:
                        parameters = json.load(json_file)
                    query = parameters['query'];
                else:
                    query = parameters_default['query'];
                data = SQLCols.executeSQL('datadummy.csv', query)
            
            if i.find('Data_Imputation') != -1:
                Impute.imputeMean(data, 'mean')
                
            if i.find('Test_Model') != -1:
                f = open("testfile.txt", "r")
                data = pd.read_csv(f.read())
                data = data[parameters['features']]
                data['results'] = bestmodel.predict(data);
                f.close();
                
            if i.find('Save') != -1:
                print('rf')
            
            if i.find('Logistic_Regression') != -1:
                if os.path.exists(i +'.json'):
                    with open(i +'.json') as json_file:
                        parameters = json.load(json_file)
                else:
                    parameters = parameters_default;
                
                val1 = float(parameters['logisticregression_split'])/100			
                Y = data[parameters['target']]
                Cs=float(parameters['logisticregression_c'])
                X_train, X_test, y_train, y_test = train_test_split(data[parameters['features']], Y, shuffle=True ,test_size=val1, stratify=Y, random_state=0)
                clf_lr = Pipeline([('lr', LogisticRegression(penalty='l1', C=Cs, random_state=0, solver='liblinear'))]).fit(X_train, y_train.values.ravel())
                X_test['Result'] = clf_lr.predict(X_test)
                X_test['Ground truth'] = y_test
                data = X_test
                
            if i.find('Grammer_Correction') != -1:
                GrammerCorrection.checkGrammer(data)
                
            if i.find('Stemming_Cleaning') != -1:
                StemmerCleaning(data, 'text')
                
            if i.find('MaskPII') != -1:
                if os.path.exists(i +'.json'):
                    with open(i +'.json') as json_file:
                        parameters = json.load(json_file)
                        print('numbers')
                else:
                    parameters = parameters_default

                text = data['text'][0]
                #print(text)
                #if parameters['serialnumbers'] == 'true':
                #    text = mobilevalidator(text)
                    #text = MaskPII.checkserialNumber(text)
                #if parameters['numbers'] == 'true':
                    #text = mobilevalidator(text)
                    #print(text)
                    #text = MaskPII.numbervalidation(text)
                    #print(text)
                if parameters['emailids'] == 'true':
                    text = mobilevalidator(text)
                    text = emailvalidator(text)
                    #text = MaskPII.emailvalidator(text)
                if parameters['urls'] == 'true':
                    text = URLvalidator(text)
                    #text = MaskPII.URLvalidator(text)
                
                if parameters['fulldates'] == 'true':
                    text = datevalidation(text)
                    #print(text)
                    #text = MaskPII.datevalidator(text)
                    #print(text)
                if parameters['fullnames'] == 'true':
                    text = NER(text)
                    #text = MaskPII.NER(text)
                #if parameters['organizations'] == 'true':
                    
                    #text = MaskPII.NER(text)
                datadf = {'text':[text]}
                data = pd.DataFrame(datadf)
            
            #if i.find('Deploy') != -1:
            #    Deploy.deploy(task_list[j])

        data.to_csv('output/results_' + str(j) + '.csv', index=False)

if __name__ == "__main__":
    main()