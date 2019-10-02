#!python

#import requests

#url = 'http://localhost/new1/'
#data = {'model': 'This is California.'}
#r = requests.post(url, data)
#print(r)


import cgi
form = cgi.FieldStorage()
searchterm =  form.getvalue('model')
print(searchterm)