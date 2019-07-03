import pickle
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix, recall_score, precision_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import sys
import pickle
import pandas
from image_classification_predict import *
from text_predictive_model import preProcessDataText, text_model
from regression_model import preProcessDataRegression, predictive_model_regression
from predictive_model import preProcessDataClassification, predictive_model_classification
#import cv2

def main(argv):
    task                = argv[1]
    filename            = argv[2]
    readfromfilevar     = argv[3]
    target              = argv[5].split(',')
	
    if task == '1':
        target              = argv[5]
        if target == '-':
            target = "Not Known"        		
        result = image_model_predict(filename)
        f = open("results.txt", "w")
        f.write('Predicted Classification')
        f.write('\n')
        f.write(result)
        f.write('\n')
        f.write('True Classification')
        f.write('\n')
        f.write(target)
        f.write('\n')
        f.write(filename)
        f.write('\n')
        f.write('x_ray_model.png')
        f.close()
        #Im_read = cv2.imread(filename)
        #cv2.imwrite('display.png', Im_read)	
		
    if task == '2':       
        features = ['History']
        target = ['Parent']	
        modelfile = 'text_classification.sav'
        imagefilename = 'text_model.png'		

    if task == '3':
        target = ['Price']
        features = ['Year','Month','Units', 'Product_Level1']
        modelfile = 'data_regression.sav'
        imagefilename = 'regression_model.png'
		
    if task == '4':
        features     = ['Supplies Subgroup', 'Supplies Group', 'Region',
          'Route To Market', 'Elapsed Days In Sales Stage',
          'Sales Stage Change Count', 'Total Days Identified Through Closing',
          'Total Days Identified Through Qualified', 'Opportunity Amount USD',
          'Client Size By Revenue', 'Client Size By Employee Count',
          'Revenue From Client Past Two Years', 'Competitor Type',
          'Ratio Days Identified To Total Days',
          'Ratio Days Validated To Total Days',
          'Ratio Days Qualified To Total Days', 'Deal Size Category']
        target = ['Opportunity Result']
        modelfile = 'data_classification.sav'
        imagefilename = 'predictive_model.png'

    if task=='6':
        file1 = open(r"MyFile2.txt","r")
        features1  = ((file1.readline()).rstrip('\n')).split(',') 
        target = ((file1.readline()).rstrip('\n')).split(',') 
        imagefilename = 'results.jpeg'
        task = file1.readline() 
        file1.close()
        features     = [x.strip() for x in features1]
        modelfile = 'model.sav'

    if task == '2':
        if readfromfilevar == '1':
            file1 = open(r"features.txt","r")
            features1  = file1.read()
            file1.close()			
            #print(features1)			
            df = pandas.DataFrame([features1], columns=features)
            df[target[0]] = 1.0
            df.to_csv('filename_delta.csv')
            filename = 'filename_delta.csv'
			
        data = pandas.read_csv(filename, encoding='ISO-8859-1')
        X_test, Y_test = preProcessDataText(data, features, target)
        loaded_model = pickle.load(open(modelfile, 'rb'))
        result = loaded_model.predict(X_test)
        X_test['Results Clasification'] = result
        X_test['True Classification'] = Y_test
        X_test.to_csv('results.csv')
        f = open("results.txt", "w")
        if readfromfilevar == '1':
            f.write('Predicted Classification')
            f.write('\n')
            f.write(str(result[0]))
            f.write('\n')
            f.write('True Classification')
            f.write('\n')
            f.write('Not Known')
        else:
            rec = recall_score(Y_test, result, average='macro')
            pre = precision_score(Y_test, result, average='macro')
            f.write('Recall')
            f.write('\n')
            f.write(str(rec))
            f.write('\n')
            f.write('Precision')
            f.write('\n')
            f.write(str(pre))
			
        f.write('\n')
        f.write('results.csv')
        f.write('\n')
        f.write(imagefilename)
        f.close()
        #Im_read = cv2.imread('results.jpeg')
        #cv2.imwrite('display.png', Im_read)
    
    if task == '3':
        if readfromfilevar == '1':
            file1 = open(r"features.txt","r")
            features1  = ((file1.readline()).rstrip('\n')).split(',') 
            file1.close()			
            #print(features1)			
            df = pandas.DataFrame([features1], columns=features)
            df[target[0]] = 1.0
            df.to_csv('filename_delta.csv')
            filename = 'filename_delta.csv'
        X_test, Y_test = preProcessDataRegression(filename, features, target)
        loaded_model = pickle.load(open(modelfile, 'rb'))
        result = loaded_model.predict(X_test)
        X_test['Results Clasification'] = result
        X_test['True Classification'] = Y_test
        X_test.to_csv('results.csv')
        f = open("results.txt", "w")
        if readfromfilevar == '1':
            f.write('Predicted Classification')
            f.write('\n')
            f.write(str(result[0]))
            f.write('\n')
            f.write('True Classification')
            f.write('\n')
            f.write('Not Known')
        else:
            mse_res = mean_squared_error(Y_test, result)
            f.write('MSE')
            f.write('\n')
            f.write(str(mse_res))
            f.write('\n')
            f.write('--')
            f.write('\n')
            f.write('--')
        f.write('\n')
        f.write('results.csv')
        f.write('\n')
        f.write(imagefilename)
        f.close()
        #Im_read = cv2.imread('results.jpeg')
        #cv2.imwrite('display.png', Im_read)

    if task == '4':
        if readfromfilevar == '1':
            file1 = open(r"features.txt","r")
            features1  = ((file1.readline()).rstrip('\n')).split(',') 
            file1.close()			
            #print(features1)			
            df = pandas.DataFrame([features1], columns=features)
            df[target[0]] = 1.0
            df.to_csv('filename_delta.csv')
            filename = 'filename_delta.csv'
        X_test, Y_test = preProcessDataClassification(filename, features, target)
        loaded_model = pickle.load(open(modelfile, 'rb'))
        result = 1-loaded_model.predict(X_test)
        X_test['Results Clasification'] = result
        X_test['True Classification'] = Y_test
        X_test.to_csv('results.csv')
        f = open("results.txt", "w")
        if readfromfilevar == '1':
            f.write('Predicted Classification')
            f.write('\n')
            f.write(str(result[0]))
            f.write('\n')
            f.write('True Classification')
            f.write('\n')
            f.write('Not Known')
        else:
            rec = recall_score(Y_test, result, average='macro')
            pre = precision_score(Y_test, result, average='macro')
            f.write('Recall')
            f.write('\n')
            f.write(str(rec))
            f.write('\n')
            f.write('Precision')
            f.write('\n')
            f.write(str(pre))
        f.write('\n')
        f.write('results.csv')
        f.write('\n')
        f.write(imagefilename)
        f.close()
        #Im_read = cv2.imread('results.jpeg')
        #cv2.imwrite('display.png', Im_read)
		
    if task =='5':
        features1    = argv[4].split(',')
        features     = [x.strip() for x in features1]
        file1 = open(r"MyFile2.txt","w+")
        file1.write(argv[4])
        file1.write("\n")	
        file1.write(argv[5])
        file1.write("\n")	
        file1.write(argv[6])	
        file1.close()
        if argv[6] == "2":
            data = pandas.read_csv(filename, encoding='ISO-8859-1')
            classifier_best = text_model(data, features, target)
        if argv[6] == "3":
           classifier_best = predictive_model_regression(filename, features, target)
        if argv[6] == "4":
           classifier_best = predictive_model_classification(filename, features, target)
        filename = 'model.sav'
        pickle.dump(classifier_best, open(filename, 'wb'))
	

           
	
if __name__ == "__main__":
    main(sys.argv)