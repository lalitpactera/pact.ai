import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
class StemmerCleaning:
    def __init__(self):
        nltk.download('stopwords')
        stemmer = PorterStemmer()
        words = stopwords.words("english")
    
    def cleanAll(self, data, features):
        data[features[0]] = data[features[0]].apply(lambda x: " ".join([self.stemmer.stem(i) for i in re.sub("<[^<]+?>|[^a-zA-Z]", " ", x).split() if i not in self.words]).lower())