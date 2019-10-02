import pandas
import re
import sys
#from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn2pmml.decoration import ContinuousDomain, CategoricalDomain
#from sklearn2pmml.preprocessing import PMMLLabelBinarizer
#from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
#from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper, CategoricalImputer
#from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer 
from lxml import etree
from sklearn2pmml import sklearn2pmml
from sklearn import datasets, linear_model
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from sklearn.pipeline import Pipeline
#from imblearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import spatial
import matplotlib.transforms as mtransforms
import pickle
import os
import zipfile
import warnings
warnings.filterwarnings("ignore")

def normalizeData(d):
    if (d < 5):
        d = 5
    if (d > 95):
        d = 95
    return d

def plotImage(y, name):
    x = []
    x.append(-1.5)
    x.append(-0.5)
    x.append(+0.5)
    x.append(+1.5)
    x = np.asarray(x)
    
    y[0] = normalizeData(y[0])
    y[1] = normalizeData(y[1])
    y[2] = normalizeData(y[2])
    y[3] = normalizeData(y[3])
    
    fig, ax = plt.subplots()
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    plt.ylim(0,100) 
    plt.yticks(np.arange(0, 100+1, 25))
    plt.xticks(np.arange(-2, 2))
    plt.xlim(-2,2) 
    
    plt.plot(x, y, color='blue', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12)
    
    ax.fill_between([0,2], 0.75, 1.00, None,
            facecolor='red', alpha=0.75, transform=trans)
    ax.fill_between([1,2], 0.5, 0.75, None,
            facecolor='red', alpha=0.75, transform=trans)
    ax.fill_between([0,1], 0.5, 0.75, None,
            facecolor='orange', alpha=0.75, transform=trans)
    ax.fill_between([-1,0], 0.75, 1.00, None,
            facecolor='orange', alpha=0.75, transform=trans)
    ax.fill_between([1,2], 0.25, 0.50, None,
            facecolor='orange', alpha=0.75, transform=trans)
    ax.fill_between([-1,0], 0.25, 0.75, None,
            facecolor='yellow', alpha=0.75, transform=trans)
    ax.fill_between([0,1], 0.25, 0.5, None,
            facecolor='yellow', alpha=0.75, transform=trans)
    ax.fill_between([-2,-1], 0.0, 1.0, None,
            facecolor='green', alpha=0.75, transform=trans)
    ax.fill_between([-1,2], 0.0, 0.25, None,
            facecolor='green', alpha=0.75, transform=trans)
    plt.grid(color='k', linestyle='-', linewidth=2)
    ax.set_xticklabels(['No_Incident'.rjust(35),'Severity 1'.rjust(35),'Severity 2'.rjust(35),'Severity 3'.rjust(35)])
    fig.savefig('results_' + name + '.png')

def pre_process(data, features):
    categorical_columns = []
    binary_columns = []
    # reading the data as pandas frame
    #data = pandas.read_csv(filename, encoding='ISO-8859-1')
        
    data = data.dropna()
    nltk.download('stopwords')
    stemmer = PorterStemmer()
    words = stopwords.words("english")
    data['cleaned'] = data[features].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("<[^<]+?>|[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    return data

def print_top_features(vectorize, data, test_data, n):
    tfs = vectorize.fit_transform(data['cleaned'])
    response = vectorize.transform(test_data['cleaned'])
    feature_names = vectorize.get_feature_names()
    feature_array = np.array(vectorize.get_feature_names())
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n]
    #print(top_n)
    #print(np.sort(response.toarray()).flatten()[::-1])
    return (top_n[0] + ' // ' + top_n[1] + ' // ' + top_n[2])
    
def similar_data(vectorize, tfs, test_data, data):
    #tfs = vectorize.fit_transform(data['cleaned']) 
    response = vectorize.transform(test_data['cleaned'])
    col1, row1 = tfs.shape
    sortVector = np.zeros((1,col1))
    for i in range(0, col1):
        sortVector[0,i] = 1 - spatial.distance.cosine(tfs[i,:].toarray(), response[0,:].toarray())
    sv1 = np.argsort(sortVector).flatten()[::-1]
    sv2 = np.sort(sortVector).flatten()[::-1]
    data_df = pandas.DataFrame(data)[sv1[0]:sv1[0]+1]
    #print(data_df['Incident Description'])
    #print(sv2[0])
    data_df['similar incident-1'] = data_df['Incident Description']
    data_df['incident-1 confidence'] = sv2[0]
    data_df['incident-1 Level'] = data_df['Record Classification']
    
  #  data_df1 = pandas.DataFrame(data)[sv1[1]:sv1[1]+1]
  #  #print(data_df1['Incident Description'])
  #  #print(sv2[1])
  #  data_df.loc[data_df.index[0], 'similar incident-2'] = data_df1.loc[data_df1.index[0],'Incident Description']
  #  data_df.loc[data_df.index[0], 'incident-2 Level'] = data_df1.loc[data_df1.index[0],'Record Classification']
  #  data_df['incident-2 confidence'] = sv2[1]
  #  #print(data_df['similar incident-2'])
    
  #  data_df1 = pandas.DataFrame(data)[sv1[2]:sv1[2]+1]
  #  #print(data_df1['Incident Description'])
  #  #print(sv2[2])
  #  data_df.loc[data_df.index[0], 'similar incident-3'] = data_df1.loc[data_df1.index[0],'Incident Description']
  #  data_df.loc[data_df.index[0], 'incident-3 Level'] = data_df1.loc[data_df1.index[0],'Record Classification']
  #  data_df['incident-3 confidence'] = sv2[2]
    
  #  data_df1 = pandas.DataFrame(data)[sv1[3]:sv1[3]+1]
  #  #print(data_df1['Incident Description'])
  #  #print(sv2[3])
  #  data_df.loc[data_df.index[0], 'similar incident-4'] = data_df1.loc[data_df1.index[0],'Incident Description']
  #  data_df.loc[data_df.index[0], 'incident-4 Level'] = data_df1.loc[data_df1.index[0],'Record Classification']
  #  data_df['incident-4 confidence'] = sv2[3]
    
  #  data_df1 = pandas.DataFrame(data)[sv1[4]:sv1[4]+1]
  #  #print(data_df1['Incident Description'])
  #  #print(sv2[4])
  #  data_df.loc[data_df.index[0], 'similar incident-5'] = data_df1.loc[data_df1.index[0],'Incident Description']
  #  data_df.loc[data_df.index[0], 'incident-5 Level'] = data_df1.loc[data_df1.index[0],'Record Classification']
  #  data_df['incident-5 confidence'] = sv2[4]

    return data_df


def processData(filename, fileNameAll):
    features = 'Incident Description'
    target = 'Record Classification'
    vectorize = TfidfVectorizer(stop_words="english", sublinear_tf=True, norm='l2', analyzer='word', ngram_range=(1, 2), encoding='ISO-8859-1')
    test_data = pandas.read_csv(filename, encoding='ISO-8859-1')
    test_data = pandas.DataFrame(test_data, columns = [features])
    test_data = pre_process(test_data, features)
	
    text1 = test_data['Incident Description'][0]
	
    model = pickle.load(open('prediction_model.sav', 'rb'))
    #print(test_data)
    pred = model.predict_proba(test_data['cleaned'])
    tfs = vectorize.fit_transform(test_data['cleaned'])
 #   test_data['probability_class-1'] = pred[:,0]*100
 #   test_data['probability_class0'] = pred[:,1]*100
 #   test_data['probability_class1'] = pred[:,2]*100
 #   test_data['probability_class2+'] = pred[:,3]*100
    
    test_data['classification (0:No_incident,1:severity 1 , 2:severity 2, 3:severity 3'] = model.predict(test_data['cleaned'])
    
    text2 = test_data['classification (0:No_incident,1:severity 1 , 2:severity 2, 3:severity 3'][0]
	
    data = pandas.read_csv(fileNameAll, encoding='ISO-8859-1')
    data = pandas.DataFrame(data, columns = [features, target])
    data = pre_process(data, features)
    tfs = vectorize.fit_transform(data['cleaned'])
    rt, ct = test_data.shape
    #print(rt)
    for i in range(0, rt):
        #print(i)
        y = []
        y.append(pred[i,0]*100)
        y.append(pred[i,1]*100)
        y.append(pred[i,2]*100)
        y.append(pred[i,3]*100)
        y = np.asarray(y)
        plotImage(y, str(i))
        
        test_data1 = pandas.DataFrame(test_data)[i:i+1].copy(deep=True)
        data_df = similar_data(vectorize, tfs, test_data1, data)
    
        test_data.loc[test_data.index[i], 'similar incident'] = data_df.loc[data_df.index[0],'similar incident-1']
        #test_data.loc[test_data.index[i], 'match confidence'] = data_df.loc[data_df.index[0],'incident-1 confidence']
        test_data.loc[test_data.index[i], 'incident severity'] = int(int(data_df.loc[data_df.index[0],'incident-1 Level'][-1]) + 1)
    
   #     test_data.loc[test_data.index[i], 'similar incident-2'] = data_df.loc[data_df.index[0],'similar incident-2']
   #     test_data.loc[test_data.index[i], 'incident-2 confidence'] = data_df.loc[data_df.index[0],'incident-2 confidence']
   #     test_data.loc[test_data.index[i], 'incident-2 Level'] = data_df.loc[data_df.index[0],'incident-2 Level']
    
   #     test_data.loc[test_data.index[i], 'similar incident-3'] = data_df.loc[data_df.index[0],'similar incident-3']
   #     test_data.loc[test_data.index[i], 'incident-3 confidence'] = data_df.loc[data_df.index[0],'incident-3 confidence']
   #     test_data.loc[test_data.index[i], 'incident-3 Level'] = data_df.loc[data_df.index[0],'incident-3 Level']
    
   #     test_data.loc[test_data.index[i], 'similar incident-4'] = data_df.loc[data_df.index[0],'similar incident-4']
   #     test_data.loc[test_data.index[i], 'incident-4 confidence'] = data_df.loc[data_df.index[0],'incident-4 confidence']
   #     test_data.loc[test_data.index[i], 'incident-4 Level'] = data_df.loc[data_df.index[0],'incident-4 Level']
    
   #     test_data.loc[test_data.index[i], 'similar incident-5'] = data_df.loc[data_df.index[0],'similar incident-5']
   #     test_data.loc[test_data.index[i], 'incident-5 confidence'] = data_df.loc[data_df.index[0],'incident-5 confidence']    
   #     test_data.loc[test_data.index[i], 'incident-5 Level'] = data_df.loc[data_df.index[0],'incident-5 Level']    
    
   #     test_data.loc[test_data.index[i], 'top three features'] = print_top_features(vectorize, data, test_data1, 3)
    
    test_data_1 = pandas.DataFrame(test_data, columns = ['Incident Description', 'classification (0:No_incident,1:severity 1 , 2:severity 2, 3:severity 3', 'similar incident', 'incident severity'])
	
    test_data_1.to_csv('results.csv')
    
    text3 = test_data['similar incident'][0]
    text4 = test_data['incident severity'][0]
	
    if rt > 1:
        zf = zipfile.ZipFile("results.zip", "w")
        for i in range(0, rt):
            filename = 'results_' + str(i) + '.png'
            zf.write(filename)
        zf.close()
    return rt, text1, text2, text3, text4

def main(argv):
    filename            = argv[1]
    readfromfilevar     = argv[2]
    features            = ['Incident Description']
    
    filename = 'dataset/' + filename
    
    if readfromfilevar == '1':
        file1 = open(r"features.txt","r")
        features1  = file1.read()
        file1.close()
        df = pandas.DataFrame([features1], columns=features)
        df.to_csv('filename_delta.csv')
        filename = 'filename_delta.csv'
    
    rt, text1, text2, text3, text4 = processData(filename, 'compare_data.csv')
    f = open(r"results.txt","w+")
	
    f.write(text1)
    f.write('\n')
    f.write('severity ' + str(text2))
    f.write('\n')
    f.write(text3)
    f.write('\n')
    f.write('severity ' + str(text4))
    f.write('\n')
    f.write("results.csv")
    f.write('\n')
    f.write("results_0.png")
    f.close()
    
if __name__ == "__main__":
    main(sys.argv)
    