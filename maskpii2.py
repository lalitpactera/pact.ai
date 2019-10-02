import datefinder
import re
import pandas as pd
from urlextract import URLExtract
from flair.data import Sentence
from flair.models import SequenceTagger
import os


def datevalidation(string_with_dates):
    shortmonths={"01":"jan","02":"feb","03":"mar","04":"apr","05":"may","06":"jun","07":"jul","08":"aug","09":"sept","10":"oct","11":"nov","12":"dec"}
    fullmonths={"01":"january","02":"february","03":"march","04":"april","05":"may","06":"june","07":"july","08":"august","09":"september","10":"october","11":"november","12":"december"}
    capsmonths={"01":"January","02":"February","03":"March","04":"April","05":"May","06":"June","07":"July","08":"August","09":"September","10":"October","11":"November","12":"December"}
    xt=''
    xt=string_with_dates
    matches = datefinder.find_dates(xt)
    try:
#         print(string_with_dates)
        for match in matches:
            l1=(str(match).split(" "))
#             print(l1[0][0:4])
            year=l1[0][0:4]
#             print(l1[0][5:7])
            month=l1[0][5:7]
#             print(l1[0][8::])
            day=l1[0][8::]    
            if(year in string_with_dates):
                if(month in xt):                    
                    if(day+"rd" in xt):
                        xt=xt.replace(day+"rd","**PHI**")
                        xt=xt.replace(month,"**PHI**")
                        xt=xt.replace(year,"**PHI**")
                    if(day+"st" in xt):
                        xt=xt.replace(day+"st","**PHI**")
                        xt=xt.replace(month,"**PHI**")
                        xt=xt.replace(year,"**PHI**")
                    if(day+"th" in xt):
                        xt=xt.replace(day+"th","**PHI**")
                        xt=xt.replace(month,"**PHI**")
                        xt=xt.replace(year,"**PHI**")
                    if(day+"nd" in xt):
                        xt=xt.replace(day+"nd","**PHI**")
                        xt=xt.replace(month,"**PHI**")
                        xt=xt.replace(year,"**PHI**")
                    if(day in xt):
                        xt=xt.replace(day,"**PHI**")
                        xt=xt.replace(month,"**PHI**")
                        xt=xt.replace(year,"**PHI**")
                else:                               
                    if(fullmonths[month] in xt):
                        if(day+"rd" in xt):
                            xt=xt.replace(day+"rd","**PHI**")
                            xt=xt.replace(fullmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"st" in xt):
                            xt=xt.replace(day+"st","**PHI**")
                            xt=xt.replace(fullmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"th" in xt):
                            xt=xt.replace(day+"th","**PHI**")
                            xt=xt.replace(fullmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"nd" in xt):
                            xt=xt.replace(day+"nd","**PHI**")
                            xt=xt.replace(fullmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day in xt):
                            xt=xt.replace(day,"**PHI**")
                            xt=xt.replace(fullmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                            
                    if(shortmonths[month] in xt):
                        if(day+"rd" in xt):
                            xt=xt.replace(day+"rd","**PHI**") 
                            xt=xt.replace(shortmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"st" in xt):
                            xt=xt.replace(day+"st","**PHI**")
                            xt=xt.replace(shortmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"th" in xt):
                            xt=xt.replace(day+"th","**PHI**")
                            xt=xt.replace(shortmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"nd" in xt):
                            xt=xt.replace(day+"nd","**PHI**")
                            xt=xt.replace(shortmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day in xt):
                            xt=xt.replace(day,"**PHI**")
                            xt=xt.replace(shortmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                    if(capsmonths[month] in xt):
                        if(day+"rd" in xt):
                            xt=xt.replace(day+"rd","**PHI**")
                            xt=xt.replace(capsmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"st" in xt):
                            xt=xt.replace(day+"st","**PHI**")
                            xt=xt.replace(capsmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"th" in xt):
                            xt=xt.replace(day+"th","**PHI**")
                            xt=xt.replace(capsmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day+"nd" in xt):
                            xt=xt.replace(day+"nd","**PHI**")
                            xt=xt.replace(capsmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
                        if(day in xt):
                            xt=xt.replace(day,"**PHI**")
                            xt=xt.replace(capsmonths[month],"**PHI**")
                            xt=xt.replace(year,"**PHI**")
        return xt 
    except:
        print("Error")

def emailvalidator(my_str):
    emails = re.findall("([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", my_str)
    for mail in emails:
        if mail in my_str:
            my_str=my_str.replace(mail,"**PHI**")
    return my_str


def URLvalidator(inputtext):
    extractor = URLExtract()
    urls = extractor.find_urls(inputtext)
    for i in urls:
        if i in inputtext:
            inputtext=inputtext.replace(i,"**PHI**")
    return inputtext

def mobilevalidator(inputtext):
    z=re.findall("(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})", inputtext)
    for i in z:
        if i in inputtext:
            inputtext=inputtext.replace(i,"**PHI**")
    return inputtext

def NER(inputtext):
    xt=''
    text=inputtext
    #text=mobilevalidator(text)
    #text=emailvalidator(text)
    #text=URLvalidator(text)
    #text=datevalidation(text)
    dic1={}
    k=[]
    v=[]
    sentence = Sentence(text)
    tagger = SequenceTagger.load('ner')
    tagger.predict(sentence)
    for entity in sentence.get_spans('ner'):
        en=str(entity)
        lis=en.split("]: \"")
#         print(en)
        l1=(lis[0].split("["))
        l2=lis[1].split("\"")
        k=l2[0]
        v=l1[0]
        dic1[k]=v
#     print("d1:",dic1)
    for i in dic1.keys():
        if dic1[i]=="PER-span ":
            if i in text:
                if " " in i:
                    if "I'm" in i:
                        if i.count(' ')>1:
                            xt=text.replace(i,"**PHI**")
                            text=xt
                    else:
                        xt=text.replace(i,"**PHI**")
                        text=xt                
        if(dic1[i]=="ORG-span "):
            if i in text:
                xt=text.replace(i,"**PHI**")
            text=xt
        if(dic1[i]=="LOC-span "):
            if i in text:
                xt=text.replace(i,"**PHI**")
            text=xt
    return text

#inputsentence=''' Elon Reeve Musk born June 28, 1971 is a technology entrepreneur, investor, and engineer. He holds South African, Canadian, and U.S. citizenship and is the founder, CEO, and lead designer of SpaceX; co-founder, CEO, and product architect of Tesla, Inc. co-founder of Neuralink; founder of The Boring Company; co-founder and co-chairman of OpenAI; and co-founder of PayPal. In December 2016, he was ranked 21st on the Forbes list of The World's Most Powerful People. He has a net worth of $19.4 billion and is listed by Forbes as the 40th-richest person in the world.Born and raised in Pretoria, South Africa, Musk moved to Canada when he was 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received an economics degree from the Wharton School and a degree in physics from the College of Arts and Sciences. He began a Ph.D. in applied physics and material sciences at Stanford University in 1995 but dropped out after two days to pursue an entrepreneurial career. He subsequently co-founded Zip2, a web software company, which was acquired by Compaq for $340 million in 1999. Musk then founded X.com, an online bank. It merged with Confinity in 2000 and later that year became PayPal, which was bought by eBay for $1.5 billion in October 2002. For contact website is https://www.tesla.com/elon-musk, and you can contact at sample USA numbers (555) 555-1234 or 555-555-1234 or 5555551234 '''
#print("Please input the text for PII masking:: ")
#inputsentence=str(input())
#text = mobilevalidator(inputsentence)
#print(datevalidation(text))
