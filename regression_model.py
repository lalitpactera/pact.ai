import numpy as np
import pandas
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

def predictive_model_regression(filename, features, target):
    categorical_columns = []
    binary_columns = []
    # reading the data as pandas frame
    data = pandas.read_csv(filename)
    
    # X is dataframe for features
    X = data[features]
    
    # Y is dataframe for target
    Y = data[target]
    
    # X_final and Y_final are dataframes after preprocessing of data. Here we have initialized them
    X_final = pandas.DataFrame(columns=features)
    Y_final = pandas.DataFrame(columns=target)
    
    # preprocessing step continues here
    for col in features:
        if is_numeric_dtype(X[col]):
            X_final[col] = X[col]
        else:
            categorical_columns.append(col)
            X_final[col] = pandas.to_numeric(getCategoryLabel(X[col])) 
    
    # Converting target to numeric in case it is string
    for col in target:
        if is_numeric_dtype(Y[col]):
            Y_final[col] = Y[col]
        else:
            Y_final[col] = pandas.to_numeric(getCategoryLabel(Y[col]))
            
    X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, shuffle=True ,test_size=0.2, random_state=0)
    classifier_best = build_clf(X_train, y_train, X_test, y_test, '')
    return classifier_best
    
def getCategoryLabel(Xcol):
    Xcolunique = Xcol.unique()
    i1 = 0
    for num in Xcolunique:
        Xcol = Xcol.replace(num, str(i1))
        i1 = i1 + 1
    return(Xcol)
    
def build_clf(X_train, y_train, X_test, y_test, labels=None):
    
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, random_state=0)

    
    # ===============================================
    # Linear Regression
    # ===============================================
    
    mse_scores = []


    best_clf_l1 = Pipeline([
            ('std', StandardScaler()),
            ('lr', linear_model.LinearRegression())])
    best_clf_l1.fit(X_train, y_train)
    pred = best_clf_l1.predict(X_test)
    l1_mse = mean_squared_error(y_test, pred)

    #print('Linear Regression')
    #print(l1_mse)
    #print(pred)
    #print(y_test)
    # ===============================================
    # Support Vector Regression
    # ===============================================
    
    mse_scores = []

    svrs = ['rbf', 'linear', 'poly']
    
    for svr in svrs:
        #print(svr)
        cs = 1.0
        clf = Pipeline([
                ('std', StandardScaler()),
                ('sv', SVR(kernel=svr, C=cs, gamma='auto', degree=2, epsilon=.1,
               coef0=1))])
        model = clf.fit(X_train2, y_train2)
        pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, pred))
        
    best_k_l2 = svrs[np.argmin(mse_scores)]
    
    #print(best_k_l2)
    
    cs = np.arange(1, 5.5, 0.5)
    #print(cs)
    mse_scores = []
    
    for Cs in cs:
        
        clf = Pipeline([
                ('std', StandardScaler()),
                ('sv', SVR(kernel=best_k_l2, C=Cs, gamma='auto', degree=3, epsilon=.1,
               coef0=1))])
        model = clf.fit(X_train2, y_train2)
        pred = model.predict(X_val)
        mse_scores.append(mean_squared_error(y_val, pred))
        

    best_C_l2 = cs[np.argmin(mse_scores)]
                              
    best_clf_l2 = Pipeline([
            ('std', StandardScaler()),
            ('lr', SVR(kernel=best_k_l2, C=best_C_l2, gamma='auto', degree=3, epsilon=.1,
               coef0=1))])
    best_clf_l2.fit(X_train, y_train)
    pred = best_clf_l2.predict(X_test)
    l2_mse = mean_squared_error(y_test, pred)
                              
    #print('Support Vector Regression')
    #print(l2_mse)
    #print(pred)
    #print(y_test)
    
    # ===============================================
    # Neural Network Regression
    # ===============================================

 #   def baseline_model(x):
 #       def bm():
 #           # create model
 #           model = Sequential()
 #           model.add(Dense(12, input_dim=x, kernel_initializer='normal', activation='relu'))
 #           model.add(Dense(6, kernel_initializer='normal', activation='relu'))
 #           model.add(Dense(3, kernel_initializer='normal', activation='relu'))
 #           model.add(Dense(1, kernel_initializer='normal'))
 #           # Compile model
 #           model.compile(loss='categorical_crossentropy', optimizer='adam')
 #           return model
 #       return bm
                                  
 #   best_clf_n2 = Pipeline([
 #           ('std', StandardScaler()),
 #           ('mlp', KerasRegressor(build_fn=baseline_model(X_train.shape[1]), epochs=100, batch_size=5, verbose=0))])
 #   best_clf_n2.fit(X_train, y_train)
 #   pred = best_clf_n2.predict(X_test)
 #   n2_mse = mean_squared_error(y_test, pred)
                              
#    print('Neural Network Regression')
#    print(n2_mse)
#    print(pred)
#    print(y_test)
    # ===============================================
    # Select Best classifier and show it's report
    # ===============================================
    
    best_clf = best_clf_l1
    clf_name = 'LinearRegression'
    best_mse = l1_mse

    if l2_mse < best_mse:
        best_clf = best_clf_l2
        clf_name = 'Support Vector Regression'
        best_mse = l2_mse
    
   # if n2_mse < best_mse:
   #     best_clf = best_clf_n2
   #     clf_name = 'Deep Neural Network Regression'
   #     best_mse = n2_mse

    print('Best Classifier: ', clf_name)
    print(best_clf)
    print('MSE', best_mse)
    return best_clf

def preProcessDataRegression(filename, features, target):
    categorical_columns = []
    binary_columns = []
    # reading the data as pandas frame
    data = pandas.read_csv(filename)
    
    # X is dataframe for features
    X = data[features]
    
    # Y is dataframe for target
    Y = data[target]
    
    # X_final and Y_final are dataframes after preprocessing of data. Here we have initialized them
    X_final = pandas.DataFrame(columns=features)
    Y_final = pandas.DataFrame(columns=target)
    
    # preprocessing step continues here
    for col in features:
        if is_numeric_dtype(X[col]):
            X_final[col] = X[col]
        else:
            categorical_columns.append(col)
            X_final[col] = pandas.to_numeric(getCategoryLabel(X[col])) 
    
    # Converting target to numeric in case it is string
    for col in target:
        if is_numeric_dtype(Y[col]):
            Y_final[col] = Y[col]
        else:
            Y_final[col] = pandas.to_numeric(getCategoryLabel(Y[col]))
    return X_final, Y_final