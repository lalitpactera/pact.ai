	def namesandorganizations(inputtext):
    		xt=''
    		text=inputtext
    		text=convert(text)
    		if "dollar" or "dollars" or "invoice" in text:
        		pass
    		text=text.replace(". ",".")
    		text=text.replace("-",".")
    		text=numbervalidation(text)
    		text=checkserialNumber(text)
    		dic1={}
    		k=[]
    		v=[]

    		sentence = Sentence(text)
    		tagger = SequenceTagger.load('ner')
    		tagger.predict(sentence)
    		for entity in sentence.get_spans('ner'):
        		en=str(entity)
        		lis=en.split("]: \"")
        		print(en)
        		l1=(lis[0].split("["))
        		l2=lis[1].split("\"")
        		k=l2[0]
        		v=l1[0]
        		dic1[k]=v
    		print("d1:",dic1)

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
