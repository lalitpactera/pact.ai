import pandas
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.wrappers.scikit_learn import KerasRegressor
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer 
from lxml import etree
from sklearn2pmml import sklearn2pmml

# preprocessing step to convert categorical data as strings are converted to numeric. Examples strings london, newyork and delhi are assigned as 0,1,2
def getCategoryLabel(Xcol):
    Xcolunique = Xcol.unique()
    i1 = 0
    for num in Xcolunique:
        Xcol = Xcol.replace(num, str(i1))
        i1 = i1 + 1
    return(Xcol)
	
# This is the main method responsible for model selection, hyperparameter selection and classification
def build_clf(X_train, y_train, X_test, y_test, column_maps, pmml_filename, folder_name='NewFolder', labels=None):
    
    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, random_state=0)

    # ===============================================
    # Neural Network Regression
    # ===============================================
 #   def baseline_model(x):
 #       def bm():
 #           # create model
 #           model = Sequential()
 #           model.add(Dense(20, input_dim=x, kernel_initializer='normal', activation='relu'))
 #           model.add(Dense(6, kernel_initializer='normal', activation='relu'))
 #           model.add(Dense(3, kernel_initializer='normal', activation='relu'))
 #           model.add(Dense(1, kernel_initializer='normal', activation='softmax'))
 #           # Compile model
 #           model.compile(loss='mean_absolute_error', optimizer='adam')
 #           return model
 #       return bm
 #                               
 #   best_clf_n2 = Pipeline([
 #           ('std', StandardScaler()),
 #           ('mlp', KerasRegressor(build_fn=baseline_model(X_train.shape[1]), epochs=100, batch_size=5, verbose=0))])
 #   best_clf_n2.fit(X_train, y_train)
 #   pred = best_clf_n2.predict(X_test)
 #   n2_auc = roc_auc_score(y_test, pred)
    
    
    # ===============================================
    # Logistic Regression with L1 norm
    # ===============================================
    Cs = 10.0**np.arange(-5, 6)
    
    auc_scores = []

    for C in Cs:
        clf = Pipeline([
            #('mapper', DataFrameMapper(column_maps)),
            ('std', StandardScaler()),
            ('lr', LogisticRegression(penalty='l1', C=C, random_state=2018, solver='liblinear'))
        ])
        clf.fit(X_train2, y_train2.values.ravel())
        pred = clf.predict_proba(X_val)
        auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

    best_C_l1 = Cs[np.argmax(auc_scores)]

    best_clf_l1 = Pipeline([
        #('mapper', DataFrameMapper(column_maps)),
        ('std', StandardScaler()),
        ('lr', LogisticRegression(penalty='l1', C=best_C_l1, random_state=2018, solver='liblinear'))
    ]).fit(X_train, y_train.values.ravel())

    l1_auc = roc_auc_score(y_test.values.ravel(), best_clf_l1.predict_proba(X_test)[:, 1])

    # ===============================================
    # Logistic Regression with L2 norm
    # ===============================================

    auc_scores = []

    for C in Cs:
        clf = Pipeline([
            #('mapper', DataFrameMapper(column_maps)),
            ('std', StandardScaler()),
            ('lr', LogisticRegression(penalty='l2', C=C, random_state=2018, solver='lbfgs'))
        ])
        clf.fit(X_train2, y_train2.values.ravel())
        pred = clf.predict_proba(X_val)
        auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

    best_C_l2 = Cs[np.argmax(auc_scores)]

    best_clf_l2 = clf = Pipeline([
        #('mapper', DataFrameMapper(column_maps)),
        ('std', StandardScaler()),
        ('lr', LogisticRegression(penalty='l2', C=best_C_l1, random_state=2018, solver='lbfgs'))
    ]).fit(X_train, y_train.values.ravel())

    l2_auc = roc_auc_score(y_test.values.ravel(), best_clf_l2.predict_proba(X_test)[:, 1])
    
    
    # ===============================================
    # Decision Tree
    # ===============================================

    max_depths = np.arange(3, 30)
    auc_scores = []

    for max_depth in max_depths:
        clf = Pipeline([
            #('mapper', DataFrameMapper(column_maps)),
            ('dt', DecisionTreeClassifier(max_depth=max_depth))
        ])
        clf.fit(X_train2, y_train2.values.ravel())
        pred = clf.predict_proba(X_val)
        auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

    best_max_depth_dt = max_depths[np.argmax(auc_scores)]

    best_clf_dt = Pipeline([
        #('mapper', DataFrameMapper(column_maps)),
        ('dt', DecisionTreeClassifier(max_depth=best_max_depth_dt))
    ]).fit(X_train, y_train.values.ravel())

    dt_auc = roc_auc_score(y_test.values.ravel(), best_clf_dt.predict_proba(X_test)[:, 1])
    
    # ===============================================
    # Random Forest
    # ===============================================

    max_depths = np.arange(3, 10)
    auc_scores = []

    for max_depth in max_depths:
        clf = Pipeline([
            #('mapper', DataFrameMapper(column_maps)),
            ('rf', RandomForestClassifier(max_depth=max_depth, random_state=2018, n_estimators=100))
        ])
        clf.fit(X_train2, y_train2.values.ravel())
        pred = clf.predict_proba(X_val)
        auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

    best_max_depth_rf = max_depths[np.argmax(auc_scores)]

    best_clf_rf = Pipeline([
        #('mapper', DataFrameMapper(column_maps)),
        ('rf', RandomForestClassifier(max_depth=best_max_depth_rf, random_state=2018, n_estimators=100))
    ]).fit(X_train, y_train.values.ravel())
    rf_auc = roc_auc_score(y_test.values.ravel(), best_clf_rf.predict_proba(X_test)[:, 1])
    
    
    # ===============================================
    # XGBClassifier
    # ===============================================

    estimators = np.arange(10, 100, 50)
    auc_scores = []

    for n_estimators in estimators:
        clf = Pipeline([
            #('mapper', DataFrameMapper(column_maps)),
            ('xgb', XGBClassifier(n_estimators=n_estimators, random_state=2018))
        ])
        # clf = RandomForestClassifier(max_depth=max_depth)
        clf.fit(X_train2, y_train2.values.ravel())
        pred = clf.predict_proba(X_val)
        auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

    best_n_estimators_xgb = estimators[np.argmax(auc_scores)]

    best_clf_xgb = Pipeline([
        #('mapper', DataFrameMapper(column_maps)),
        ('xgb', XGBClassifier(n_estimators=best_n_estimators_xgb, random_state=2018))
    ]).fit(X_train, y_train.values.ravel())

    xgb_auc = roc_auc_score(y_test.values.ravel(), best_clf_xgb.predict_proba(X_test)[:, 1])

    
    # ===============================================
    # Select Best classifier and show it's report
    # ===============================================
    
    best_clf = best_clf_l1
    clf_name = 'Logistic_Regression_with_L1_norm'
    best_auc = l1_auc
    feaure_importances = best_clf_l1.named_steps['lr'].coef_.ravel()

    if l2_auc > best_auc:
        best_clf = best_clf_l2
        clf_name = 'Logistic_Regression_with_L2_norm'
        best_auc = l2_auc
        feaure_importances = best_clf_l2.named_steps['lr'].coef_.ravel()
    if dt_auc > best_auc:
        best_clf = best_clf_dt
        clf_name = 'DecisionTree'
        best_auc = dt_auc
        feaure_importances = best_clf_dt.named_steps['dt'].feature_importances_
    if rf_auc > best_auc:
        best_clf = best_clf_rf
        clf_name = 'RandomForest'
        best_auc = rf_auc
        feaure_importances = best_clf_rf.named_steps['rf'].feature_importances_
    if xgb_auc > best_auc:
        best_clf = best_clf_xgb
        clf_name = 'XGBClassifier'
        best_auc = xgb_auc
        feaure_importances = best_clf_xgb.named_steps['xgb'].feature_importances_
  #  if n2_auc > best_auc:
  #      best_clf = best_clf_n2
  #      clf_name = 'Deep Neural Network Classifier'
  #      best_auc = n2_auc
        
    print('Best Classifier: ', clf_name)
    print(best_clf)
    
    report_roc(best_clf, X_train, y_train.values.ravel(), X_test, y_test.values.ravel())
    #plot_weights(feaure_importances, best_clf.named_steps['mapper'].transformed_names_, top_n=30)
    #plt.show()

    plot_thresholds(best_clf,X_test, y_test)
    plt.show()

    print('\nConfusion Matrix on different probability thresholds')

    probability_thresholds = [0.5, 0.6]

    for threshold in probability_thresholds:
        plot_confusion_matrix_(y_test, predict(best_clf, X_test, threshold=threshold), labels=labels)
        plt.title('Threshold = ' + str(threshold))
        plt.show()
    
    #pmml_filename = pmml_filename + '.pmml'
    #print('\nPmml File:', pmml_filename)
    #save_pmml(best_clf, pmml_filename)
    
    return best_clf
	
# supporting method for predicting the probabilities
def predict(clf, X, threshold=0.5):
    prob = clf.predict_proba(X)
    return pandas.Series(prob[:, 1]).apply(lambda p: 1 if p>=threshold else 0).values
	
# method to save model as pmml
def save_pmml(pmml_clf, filename):
    sklearn2pmml(pmml_clf, filename)
	
# method to plot confusion matrix
def plot_confusion_matrix_(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pandas.DataFrame(cm, index = labels,
                  columns = labels)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap="YlGnBu");
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
	
# supporting method to show results
def plot_thresholds(clf, X_test, y_test, n_thresholds=100):
    prob = clf.predict_proba(X_test)
    thresholds = np.linspace(0, max(prob[:, 1]), n_thresholds)
    recall_scores = []
    precision_scores = []

    for threshold in thresholds:
        y_pred = pandas.Series(prob[:, 1]).apply(lambda p: 1 if p>=threshold else 0)
        recall_scores.append(recall_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
    
    plt.plot(thresholds, recall_scores, label='recall scores')
    plt.plot(thresholds, precision_scores, label='precision scores')
    plt.xlabel('Thresholds')
    plt.ylabel('Score')
    plt.title('Precision and recall on different thresholds')
    plt.legend()
    plt.show()
	
# supporting method to show results
def report_roc(clf, X_train, y_train, X_test, y_test):
    train_rank  = clf.predict_proba(X_train)[:,1]
    test_rank = clf.predict_proba(X_test)[:,1]
    
    train_fpr, train_tpr, train_threshold = roc_curve(y_train, train_rank)
    test_fpr, test_tpr, test_threshold = roc_curve(y_test, test_rank)
    
    train_auc = auc(train_fpr, train_tpr)
    test_auc = auc(test_fpr, test_tpr)
    print('Train AUC =', train_auc)
    print('Test AUC =', test_auc)
    plt.title('Receiver Operating Characteristic')
    plt.plot(train_fpr, train_tpr, 'b',label='Train AUC = %0.2f'% train_auc)
    plt.plot(test_fpr, test_tpr, 'r',label='Test AUC  = %0.2f'% test_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'k--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
	
# supporting method to show results
def plot_weights(weights, column_names, top_n=30):
    feature_weights = list(zip(column_names, weights))
    feature_weights = pandas.DataFrame.from_records(feature_weights, columns=['feature', 'weights'])
    feature_weights['abs_weights'] = feature_weights['weights'].apply(abs)
    feature_weights.sort_values('abs_weights', ascending=False, inplace=True)

    fig, ax =plt.subplots(figsize=(12,20))

    pallete = feature_weights['weights'].map(lambda w: '#0074D9' if w > 0 else '#FF4136')
    sns.barplot(x='weights', y='feature', data=feature_weights.head(top_n), palette=pallete, ax=ax);
	
# preprocess and classify
def predictive_model_classification(filename, features, target):
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
    
    # Checking for binary columns, we need to know them a priori for data imputation
    for col in features:
        if np.alen(X[col].unique()) == 2:
            binary_columns.append(col)
            
    # preparing data for data imputation in case of null values
    column_maps = []
    for col in X.columns:
        if (col in binary_columns) or (col in categorical_columns):
            column_maps.append((col, [CategoricalDomain(), PMMLLabelBinarizer()]))
        else:
            column_maps.append(([col], [ContinuousDomain(missing_value_treatment='as_median'), SimpleImputer(strategy='median')]))
            
    X_train, X_test, y_train, y_test = train_test_split(X_final, Y_final, shuffle=True ,test_size=0.3, stratify=Y, random_state=0)
    
    classifier_best = build_clf(X_train, y_train, X_test, y_test, column_maps, '')
    return classifier_best
	
def preProcessDataClassification(filename, features, target):
    categorical_columns = []
    binary_columns = []
    # reading the data as pandas frame
    data = pandas.read_csv(filename)
    
    # X is dataframe for features
    X = data[features]
    
    # X_final and Y_final are dataframes after preprocessing of data. Here we have initialized them
    X_final = pandas.DataFrame(columns=features)
            
    # preprocessing step continues here
    for col in features:
        if is_numeric_dtype(X[col]):
            X_final[col] = X[col]
        else:
            categorical_columns.append(col)
            X_final[col] = pandas.to_numeric(getCategoryLabel(X[col])) 
    
    # Checking for binary columns, we need to know them a priori for data imputation
    for col in features:
        if np.alen(X[col].unique()) == 2:
            binary_columns.append(col)
    
    if target != '':
        # Y is dataframe for target
        Y = data[target]
        Y_final = pandas.DataFrame(columns=target)
        # Converting target to numeric in case it is string
        for col in target:
            if is_numeric_dtype(Y[col]):
                Y_final[col] = Y[col]
            else:
                Y_final[col] = pandas.to_numeric(getCategoryLabel(Y[col]))
    else:
        Y_final = ''
        
    return X_final, Y_final
