import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class LogisticRegression:
    def classify(X_train, y_train, automl = False, Cs=0.5, Norm = 'l2'):
        
		#print(X_train)
		#print(y_train)
		
        
        if automl == False:
            clf_lr = Pipeline([('std', StandardScaler()), ('lr', LogisticRegression(penalty=Norm, C=Cs, random_state=0, solver='liblinear'))])
            model = clf_lr.fit(X_train, y_train.values.ravel())
            return model
			
        X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, random_state=0)
        # ===============================================
        # Logistic Regression with L1 norm
        # ===============================================
        Cs = 10.0**np.arange(-5, 6)
    
        auc_scores = []

        for C1 in Cs:
            clf = Pipeline([
                ('std', StandardScaler()),
                ('lr', LogisticRegression(penalty='l1', C=C1, random_state=2018, solver='liblinear'))
            ])
            clf.fit(X_train2, y_train2.values.ravel())
            pred = clf.predict_proba(X_val)
            auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

        best_C_l1 = Cs[np.argmax(auc_scores)]

        best_clf_l1 = Pipeline([
            ('std', StandardScaler()),
            ('lr', LogisticRegression(penalty='l1', C=best_C_l1, random_state=2018, solver='liblinear'))
        ]).fit(X_train, y_train.values.ravel())

        l1_auc = roc_auc_score(y_test.values.ravel(), best_clf_l1.predict_proba(X_test)[:, 1])

        # ===============================================
        # Logistic Regression with L2 norm
        # ===============================================

        auc_scores = []

        for C1 in Cs:
            clf = Pipeline([
                ('std', StandardScaler()),
                ('lr', LogisticRegression(penalty='l2', C=C1, random_state=2018, solver='lbfgs'))
            ])
            clf.fit(X_train2, y_train2.values.ravel())
            pred = clf.predict_proba(X_val)
            auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

        best_C_l2 = Cs[np.argmax(auc_scores)]

        best_clf_l2 = clf = Pipeline([
            ('std', StandardScaler()),
            ('lr', LogisticRegression(penalty='l2', C=best_C_l1, random_state=2018, solver='lbfgs'))
        ]).fit(X_train, y_train.values.ravel())

        l2_auc = roc_auc_score(y_test.values.ravel(), best_clf_l2.predict_proba(X_test)[:, 1])
    
    
        best_clf = best_clf_l1
        clf_name = 'Logistic_Regression_with_L1_norm'
        best_auc = l1_auc
        #feaure_importances = best_clf_l1.named_steps['lr'].coef_.ravel()

        if l2_auc > best_auc:
            best_clf = best_clf_l2
            clf_name = 'Logistic_Regression_with_L2_norm'
            best_auc = l2_auc
            #feaure_importances = best_clf_l2.named_steps['lr'].coef_.ravel()
        
        print('Best Classifier: ', clf_name)
        print(best_clf)
     
        return best_clf
        