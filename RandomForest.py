import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, recall_score, precision_score

class RandomForest:
    def classify(X_train, y_train, automl = True, max_depths=5):
        X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, random_state=0)
        
        if automl = False:
			clf_rf = Pipeline([
				('rf', RandomForestClassifier(max_depth=max_depths, random_state=2018, n_estimators=100))
				]).fit(X_train, y_train.values.ravel())
			return clf_lr
   
        max_depths = np.arange(3, 10)
		auc_scores = []

		for max_depth in max_depths:
			clf = Pipeline([('rf', RandomForestClassifier(max_depth=max_depth, random_state=2018, n_estimators=100))])
			clf.fit(X_train2, y_train2.values.ravel())
			pred = clf.predict_proba(X_val)
			auc_scores.append(roc_auc_score(y_val.values.ravel(), pred[:, 1]))

		best_max_depth_rf = max_depths[np.argmax(auc_scores)]

		best_clf_rf = Pipeline([
        	('rf', RandomForestClassifier(max_depth=best_max_depth_rf, random_state=2018, n_estimators=100))
			]).fit(X_train, y_train.values.ravel())
		
        print('Best Classifier: ', clf_name)
        print(best_clf)
     
        return best_clf
        