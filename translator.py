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

l=["PERSON","PROPN","NORP","ORG","GPE","LOC","COUNTRY","NUM","EMAIL"]
spacy_nlp = spacy.load('en')

def pii(name_file):
	df=pd.read_csv(name_file,header=None)
	column=['text']
	df.columns=column
	at=[]
	for row in df['text']:
	    s=""
	    #with open(name_file,'r') as f:
	    #   for line in f:
	    document = spacy_nlp(row)
	    for element in document:
	    	if element.pos_ in l:
	    		element="**PHI**"
	  			s=s+str(element)+" "
	   		else:
				s=s+str(element)+" "
		at.append(s)
	df['annotated_text']=at
	df.to_csv(r'result.csv')

def correction(text):
	parser = GingerIt()
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
	        #with open("speech.csv",'r') as f:
	        #	for line in f:
	        #		output(line)
			#		extract_email_addresses(line)
	        pii()
	    except:
	        print("Sorry could not recognize your voice")


def audiofile(AUDIO_FILE):                                      
	r = sr.Recognizer()
	with sr.AudioFile(AUDIO_FILE) as source:
		audio = r.record(source)
		try:
			text = r.recognize_google(audio)
			with open("speech.csv",'w') as f:
				f.write(format(text))
			#with open("speech.csv",'r') as f:
			#	for line in f:
			#		output(line)
			pii("speech.csv")
		except:
			print("Sorry could not recognize your voice")

def main(argv):
	filename            = argv[1]
    #readfromfilevar     = argv[3]
    #target              = argv[5].split(',')
	if filename[-3:] == '.csv':
		pii(name_file)
	if filename[-3:] == '.wav':
		audiofile(name_file)
	
	f = open(r"results.txt","w+")
	f.write(text1)
	f.write('\n')
	f.write(str(text2))
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
	