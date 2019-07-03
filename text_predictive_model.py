# imports pandas, nltk and sklearn
import pandas
import re
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
from sklearn2pmml.preprocessing import PMMLLabelBinarizer
#from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, recall_score, precision_score
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer 
from lxml import etree
from sklearn2pmml import sklearn2pmml
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")

def text_model(data, features, target):
    categorical_columns = []
    binary_columns = []
    # reading the data as pandas frame
    #data = pandas.read_csv(filename, encoding='ISO-8859-1')
    
    data = data[features + target]
    
    data = data.dropna()
    #data = data[data[target[0]].isin(['9729.0', '2911.0', '9705.0', '9676.0', '9724.0', '9721.0', '9703.0', '9706.0'])]
    nltk.download('stopwords')
    stemmer = PorterStemmer()
    words = stopwords.words("english")
    data['cleaned'] = data[features[0]].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("<[^<]+?>|[^a-zA-Z]", " ", x).split() if i not in words]).lower())

    vectorizer = featureSelection(data, target) 
    
    X = data['cleaned']
    # labels
    Y = data[target[0]]
    # splitting train and test, 90:10
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, stratify=Y, random_state=0)
    classifier_best = build_clf_text(X_train, y_train, X_test, y_test, vectorizer, '')
    return(classifier_best)
    
def featureSelectionClassify(vectorizer, data, target):
    # features
    X = data['cleaned']
    # labels
    Y = data[target[0]]
    # splitting train and test, 90:10
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)
    
    pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k='all')),
                     ('clf', LogisticRegression(penalty='l2', random_state=2018, solver='lbfgs', max_iter=500))])
    model = pipeline.fit(X_train, y_train)
    ytest = np.array(y_test)
    ytest_predicted = model.predict(X_test)
    return(sum(ytest==ytest_predicted)/np.alen(ytest)*100.0)
    
    
def featureSelection(data, target):
    # tFidFeatures with 1% and 5%
    acc = []
 #   vectorizer0 = TfidfVectorizer(stop_words="english", sublinear_tf=True, norm='l2', analyzer='word', ngram_range=(1, 1), encoding='ISO-8859-1')
 #   acc.append(featureSelectionClassify(vectorizer0, data))
    
 #   vectorizer1 = CountVectorizer(stop_words="english", ngram_range=(1, 1), encoding='ISO-8859-1', analyzer='word')
 #   acc.append(featureSelectionClassify(vectorizer1, data))
    
    vectorizer0 = TfidfVectorizer(stop_words="english", sublinear_tf=True, norm='l2', analyzer='word', ngram_range=(1, 2), encoding='ISO-8859-1')
    #acc.append(featureSelectionClassify(vectorizer0, data, target))
    
    #vectorizer1 = TfidfVectorizer(stop_words="english", sublinear_tf=True, norm='l2', analyzer='word', ngram_range=(1, 1), encoding='ISO-8859-1')
    #acc.append(featureSelectionClassify(vectorizer1, data, target))
    
  #  vectorizer3 = CountVectorizer(stop_words="english", ngram_range=(1, 2), encoding='ISO-8859-1', analyzer='word')
  #  acc.append(featureSelectionClassify(vectorizer3, data))
    
    print(acc)
    maxacc = 0
    #maxacc = np.argmax(acc)
    
    if maxacc == 0:
        return(vectorizer0)
    if maxacc == 1:
        return(vectorizer1)
    if maxacc == 2:
        return(vectorizer2)
    if maxacc == 3:
        return(vectorizer3)
    
    return(vectorizer2) 
    
    
def build_clf_text(X_train, y_train, X_test, y_test, vectorizer, labels=None):
    
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, random_state=0)

    # ===============================================
    # XGBClassifier
    # ===============================================

#    estimators = np.arange(10, 30, 20)
#    auc_scores = []
#    mlb = MultiLabelBinarizer()
#    for n_estimators in estimators:
#        clf = Pipeline([
#            ('vect', vectorizer),
#            ('chi',  SelectKBest(chi2, k='all')),
#            ('xgb', OneVsRestClassifier(XGBClassifier(n_estimators=n_estimators, random_state=2018)))
#        ])
#        # clf = RandomForestClassifier(max_depth=max_depth)
#        model = clf.fit(X_train2, y_train2.values.ravel())
#        pred = model.predict_proba(X_val)
        
        
#        auc_scores.append(f1_score(y_val, mlb.fit_transform(model.predict(X_val)), average='macro'))

#    best_n_estimators_xgb = estimators[np.argmax(auc_scores)]

#    best_clf_xgb = Pipeline([
#        ('vect', vectorizer),
#        ('chi',  SelectKBest(chi2, k='all')),
#        ('xgb', OneVsRestClassifier(XGBClassifier(n_estimators=best_n_estimators_xgb, random_state=2018)))
#    ]).fit(X_train, y_train.values.ravel())

#    xgb_auc = f1_score(y_test, mlb.fit_transform(best_clf_xgb.predict(X_test)), average='macro')
    
    
    # ===============================================
    # Logistic Regression with L1 norm
    # ===============================================
    
    Cs = 10.0**np.arange(-5, 6)
    
    auc_scores = []

 #   for C in Cs:
 #       clf = Pipeline([
 #           ('vect', vectorizer),
 #           ('chi',  SelectKBest(chi2, k='all')),
 #           ('lr', LogisticRegression(penalty='l1', C=C, random_state=2018, solver='liblinear', max_iter=500, multi_class='auto'))
 #       ])
 #       model = clf.fit(X_train2, y_train2)
 #       #pred = model.predict(X_val)
 #       auc_scores.append(f1_score(y_val, model.predict(X_val), average='macro'))

 #    best_C_l1 = Cs[np.argmax(auc_scores)]

 #    best_clf_l1 = Pipeline([
 #       ('vect', vectorizer),
 #       ('chi',  SelectKBest(chi2, k='all')),
 #       ('lr', LogisticRegression(penalty='l1', C=best_C_l1, random_state=2018, solver='liblinear', max_iter=500, multi_class='auto'))
 #   ]).fit(X_train, y_train)

  #  l1_auc = f1_score(y_test, best_clf_l1.predict(X_test), average='macro')
    
  #  print(l1_auc)
    l1_auc = 0.0
    #l1_auc = roc_auc_score(y_test.values.ravel(), best_clf_l1.predict_proba(X_test)[:, 1])

    # ===============================================
    # Logistic Regression with L2 norm
    # ===============================================

  #  auc_scores = []

    for C in Cs:
        clf = Pipeline([
            ('vect', vectorizer),
            ('chi',  SelectKBest(chi2, k='all')),
            ('lr', LogisticRegression(penalty='l2', C=C, random_state=2018, solver='lbfgs', multi_class='auto'))
        ])
        model = clf.fit(X_train2, y_train2.values.ravel())
        #pred = model.predict(X_val)
        auc_scores.append(f1_score(y_val, model.predict(X_val), average='macro'))

    best_C_l2 = Cs[np.argmax(auc_scores)]

    best_clf_l2 = clf = Pipeline([
        ('vect', vectorizer),
        ('chi',  SelectKBest(chi2, k='all')),
        ('lr', LogisticRegression(penalty='l2', C=best_C_l2, random_state=2018, solver='lbfgs', multi_class='auto'))
    ]).fit(X_train, y_train)
    
    l2_auc = f1_score(y_test, best_clf_l2.predict(X_test), average='macro')
  #  l2_auc = 0.0
    print(l2_auc)
    

    # ===============================================
    # Neural Network
    # ===============================================
    n2_auc = 0.0
   # def baseline_model(x):
   #     def bm():
   #         # create model
   #         model = Sequential()
   #         model.add(Dense(12, input_dim=x, kernel_initializer='normal', activation='relu'))
   #         model.add(Dense(6, kernel_initializer='normal', activation='relu'))
   #         model.add(Dense(3, kernel_initializer='normal', activation='relu'))
   #         model.add(Dense(1, kernel_initializer='normal', activation='softmax'))
   #         # Compile model
   #         model.compile(loss='mean_absolute_error', optimizer='adam')
   #         return model
   #     return bm
   #                               
   # best_clf_n2 = Pipeline([
   #         ('vect', vectorizer),
   #         ('chi',  SelectKBest(chi2, k='all')),
   #         ('mlp', KerasRegressor(build_fn=baseline_model(X_train.shape[1]), epochs=100, batch_size=5, verbose=0))])
   # best_clf_n2.fit(X_train, y_train)
   # n2_auc = f1_score(y_test, best_clf_n2.predict(X_test), average='macro')
    

    
    # ===============================================
    # Select Best classifier and show it's report
    # ===============================================
    
    #best_clf = best_clf_l1
    clf_name = 'Logistic_Regression_with_L1_norm'
    best_auc = l1_auc
    #feaure_importances = best_clf_l1.named_steps['lr'].coef_.ravel()

    if l2_auc > best_auc:
        best_clf = best_clf_l2
        clf_name = 'Logistic_Regression_with_L2_norm'
        best_auc = l2_auc
     #   feaure_importances = best_clf_l2.named_steps['lr'].coef_.ravel()
    #if n2_auc > best_auc:
    #    best_clf = best_clf_n2
    #    clf_name = 'Deep Neural Network Classifier'
    #    best_auc = n2_auc
      #  feaure_importances = best_clf_xgb.named_steps['xgb'].feature_importances_

    print('Best Classifier: ', clf_name)
    print(best_clf)
    
    print(classification_report(y_test, best_clf.predict(X_test)))
    print(confusion_matrix(y_test, best_clf.predict(X_test)))
    
    return best_clf
    
def preProcessDataText(data, features, target):
    categorical_columns = []
    binary_columns = []
    # reading the data as pandas frame
    #data = pandas.read_csv(filename, encoding='ISO-8859-1')
	
    data = data[features + target]
    
    data = data.dropna()
    #data = data[data[target[0]].isin(['9729.0', '2911.0', '9705.0', '9676.0', '9724.0', '9721.0', '9703.0', '9706.0'])]
    nltk.download('stopwords')
    stemmer = PorterStemmer()
    words = stopwords.words("english")
    data['cleaned'] = data[features[0]].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("<[^<]+?>|[^a-zA-Z]", " ", x).split() if i not in words]).lower())
    
    X = data['cleaned']
    # labels
    Y = data[target[0]]
    return X,Y 