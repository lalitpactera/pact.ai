from shutil import copyfile
import os
class Deploy:
    def deploy(tasklist):
        foldername = '1'
        copyfile('parameters.json', foldername + 'parameters.json')
        for i in tasklist:
            if i.find('Remove_Duplications') != -1:
                copyfile('RemoveDuplicates.py', foldername + 'RemoveDuplicates.py')
                
            if i.find('Remove_Missing_Data') != -1:
                copyfile('Remove_Missing_Data.py', foldername + 'Remove_Missing_Data.py')
            
            if i.find('Normalization') != -1:
                if os.path.exists(i +'.json'):
                    copyfile(i +'.json', foldername + i +'.json')
                copyfile('Normalization.json', foldername + 'Normalization.json')
                    
            if i.find('Categorical_to_Numeric') != -1:
                 copyfile('Categorical_to_Numeric.py', foldername + 'Categorical_to_Numeric.py')
                
            if i.find('SQL') != -1:
                if os.path.exists(i +'.json'):
                    copyfile(i +'.json', foldername + i +'.json')
                copyfile('SQLCols.py', foldername + 'SQLCols.py')
            
            if i.find('Data_Imputation') != -1:
                copyfile('DataImputation.py', foldername + 'DataImputation.py')
                
            if i.find('Logistic_Regression') != -1:
                copyfile('model.xyz', foldername + 'model.xyz')
                
            if i.find('Grammer_Correction') != -1:
                copyfile('GrammerCorrection.py', foldername + 'GrammerCorrection.py')
                
            if i.find('Stemming_Cleaning') != -1:
                copyfile('StemmerCleaning.py', foldername + 'StemmerCleaning.py')
                
            if i.find('MaskPII') != -1:
                if os.path.exists(i +'.json'):
                    copyfile(i +'.json', foldername + i +'.json')
                copyfile('MaskPII.py', foldername + 'MaskPII.py')
                