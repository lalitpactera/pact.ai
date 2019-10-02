from dateutil.parser import parse
import pandas as pd
import re
import os
from urlextract import URLExtract
from flair.data import Sentence
from flair.models import SequenceTagger



class MaskPII:
    def checkserialNumber(text):
        replstr=''
        strnumber=text.split(" ")
        for i in strnumber:
            if(bool(re.match('^(?=.*[0-9])(?=.*[a-zA-Z.-])',i))==True and len(i)>=5 and i[-1]!='.'):
                text=text.replace(i,"**PHI**")
        return text

    def numbervalidation(text):
        print(text)
        number=re.compile('(([0-9]*(\\.[0-9]*))(\\.[0-9]*))')
        numlist=number.findall(text)
        for i in range(len(numlist)):
            if (numlist[i][0][0] in text):
                print(numlist[i][0])
                xt=text.replace(numlist[i][0],"**PHI**")
            text=xt
        return text
 
    def datevalidator(text):
        shortmonths={"1":"jan","2":"feb","3":"mar","4":"apr","5":"may","6":"jun","7":"jul","8":"aug","9":"sept","10":"oct","11":"nov","12":"dec"}
        fullmonths={"1":"january","2":"february","3":"march","4":"april","5":"may","6":"june","7":"july","8":"august","9":"september","10":"october","11":"november","12":"december"}
        text=text.lower()
        print("date validator",text)
        try:
            dat=parse(text, fuzzy_with_tokens=True)[0]
            print(dat)
            year=(str(dat.year))
            month=(str(dat.month))
            day=(str(dat.day))
            dtext=''
            print(year,month,day)
            dmonth=''
            dday=''

            if (year in text):
                if(int(month)<10):
                    dmonth='0'+month
                if(int(day)<10):
                    dday='0'+day
                if(month in text):
                    print("month in int")
                    if(day in text):
                        if(day in text or dday in text):
                            xt=text.replace(day,"**PHI**")
                            text=xt
                        if(month in text or dmonth in text):
                            xt=text.replace(month,"**PHI**")
                            text=xt
                        if(year in text):
                            xt=text.replace(year,"**PHI**")
                            text=xt
                else:
#                 if(shortmonths[month] in text) or (fullmonths[month] in text ):
                    print("month is ordinal")
                    if(day in text):
                        if(day in text ):
                            xt=text.replace(day,"**PHI**")
                            text=xt
                            print(text)                
                        if(fullmonths[month] in text):
                           	xt=text.replace(fullmonths[month],"**PHI**")
                           	text=xt
                           	print(text)
                        if(shortmonths[month] in text):
                           	xt=text.replace(shortmonths[month],"**PHI**")
                           	text=xt                            
                        if(year in text):
                           	xt=text.replace(year,"**PHI**")
                           	text=xt
            text=text.replace("**PHI**rd","**PHI**")
            text=text.replace("**PHI**nd","**PHI**")
            text=text.replace("**PHI**th","**PHI**")
            text=text.replace("**PHI**st","**PHI**")
            return text
        except:
            return text
           
	
    def URLvalidator(inputtext):
        extractor = URLExtract()
        urls = extractor.find_urls(inputtext)
        for i in urls:
            if i in inputtext:
                inputtext=inputtext.replace(i,"**PHI**")
        return inputtext

    def emailvalidator(my_str):
        emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", my_str)
        for mail in emails:
            if mail in my_str:
                my_str=my_str.replace(mail,"**PHI**")
        return my_str
    
    def NER(text):
        xt=''
        dic1={}
        k=[]
        sentence = Sentence(text)
        tagger = SequenceTagger.load('ner')
        tagger.predict(sentence)
        for entity in sentence.get_spans('ner'):
            en=str(entity)
            lis=en.split("]: \"")
            l1=(lis[0].split("["))
            l2=lis[1].split("\"")
            k=l2[0]
            v=l1[0]
            dic1[k]=v
		
        for i in dic1.keys():
            if dic1[i]=="PER-span " and "'" not in dic1[i]:
                if i in text:
                    if " " in i:
                        xt=text.replace(i,"**PHI**")
                        text=xt
            if(dic1[i]=="ORG-span "):
                if i in text:
                    xt=text.replace(i,"**PHI**")
                text=xt    
        return text