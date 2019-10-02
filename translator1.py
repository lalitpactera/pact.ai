import speech_recognition as sr     
from gingerit.gingerit import GingerIt
import csv
import pandas as pd
import spacy
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
import speech_recognition as sr
from os import path
import sys

l=["PERSON","PROPN","NORP","ORG","GPE","LOC","COUNTRY","NUM","EMAIL"]
spacy_nlp = spacy.load('en')

def pii(name_file):
	df_1=pd.read_csv(name_file, encoding='ISO-8859-1')
	#column=['','','','text']
	#print(df_1)
	df = pd.DataFrame(df_1, columns = ['text'])
	#print(df)
	# df.columns=column
	text3 = '--'
	text4 = '--'
	
	at=[]
	ct=[]
	for row in df['text']:
		print(row)
		k=output(row)
		ct.append(k)
	df['corrected_text']=ct
	for row in df['corrected_text']:
	    s=""
	    document = spacy_nlp(row)
	    for element in document:
	    	if element.pos_ in l:
	    		element="**PHI**"
	    		s=s+str(element)+" "
	    	else:
	    		s=s+str(element)+" "
	    at.append(s)
	df['annotated_text']=at
	df.to_csv(r'results.csv')
	text1 = at[0]
	text2 = df['text'][0]
	if len(at) > 1:
		text3 = at[1]
		text4 = df['text'][1]
	return text1, text2, text3, text4

def correction(text):
	parser = GingerIt()
	print(text)
	pred=parser.parse(text)
	return pred['result']

def output(text):
	return correction(text)

def speechpii():
	r = sr.Recognizer()                 
	with sr.Microphone() as source:     
	    print("Speak Anything :")
	    audio = r.listen(source)        
	    try:
	        text = r.recognize_google(audio)    
	        with open("speech.csv",'w') as f:
	        	f.write(format(text))
	        pii("speech.csv")
	    except:
	        print("Sorry could not recognize your voice")


def audiofile(AUDIO_FILE):                                      
	r = sr.Recognizer()
	df_aud = pd.DataFrame(columns=['text'])
	x = []
	with sr.AudioFile(AUDIO_FILE) as source:
		audio = r.record(source)
		# try:
		text = r.recognize_google(audio)
		x.append(format(text))
	df_aud['text'] = x
		#with open("speech.csv",'w') as f:
		#	f.write(format(text))
	df_aud.to_csv('speech.csv')
	text1, text2, text3, text4 = pii("speech.csv")
		# except:
			# print("Sorry could not recognize your voice")
	return text1, text2, text3, text4

def main(argv):
	filename            = argv[1]
	readfromfilevar     = argv[2]
	features            = ['text']
	
	if readfromfilevar == '1':
		file1 = open(r"features.txt","r")
		features1  = file1.read()
		file1.close()
		df = pd.DataFrame([features1], columns=features)
		df.to_csv('filename_delta.csv')
		filename = 'filename_delta.csv'
		
	#print(filename)
	if filename[-3:] == 'csv':
		text1, text2, text3, text4 = pii(filename)
	if filename[-3:] == 'wav':
		text1, text2, text3, text4 = audiofile(filename)
	f = open(r"results.txt","w+")
	f.write(text1)
	f.write('\n')
	f.write(text2)
	f.write('\n')
	f.write(text3)
	f.write('\n')
	f.write(text4)
	f.write('\n')
	f.write("results.csv")
	f.write('\n')
	f.write("results.jpeg")
	f.close()
	
if __name__ == "__main__":
	main(sys.argv)
	