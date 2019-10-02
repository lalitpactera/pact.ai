import pandas as pd
import numpy as np
import sys
from RemoveNans import *
from RemoveDuplicates import *
from NormalizeColumns import *
from CategoricalColumns import *
from LogisticRegression import *
from StemmerCleaning import *

class Tree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.top = None
        self.down = None
        self.data = None

def treer(tree_object, dict1, arrayop, currentnode):
    
    tree_object.data = dict1[currentnode]
    tree_object.left = 'Null'
    tree_object.right = 'Null'
    tree_object.top = 'Null'
    tree_object.down = 'Null'
    
    nodes = np.where(arrayop[:,0] == currentnode)
    print(np.alen(nodes[0]))
    if len(nodes[0]) == 0:
        return tree_object
    for i in range(0,len(nodes[0])):
        if i == 0:
            tree_object.left = Tree()
            treer(tree_object.left, dict1, arrayop, int(arrayop[nodes[0][i],1]))
        if i == 1:
            tree_object.right = Tree()
            treer(tree_object.right, dict1, arrayop, int(arrayop[nodes[0][i],1]))
        if i == 2:
            tree_object.top = Tree()
            treer(tree_object.top, dict1, arrayop, int(arrayop[nodes[0][i],1]))
        if i == 3:
            tree_object.down = Tree()
            treer(tree_object.down, dict1, arrayop, int(arrayop[nodes[0][i],1]))
    return tree_object

def treeOperate(tree_object, taskList, masterlist):
    taskList.append(tree_object.data)
    if tree_object.left == 'Null' and tree_object.right == 'Null' and tree_object.top == 'Null' and tree_object.down == 'Null':
        masterlist.append(taskList.copy())
    if tree_object.left != 'Null':
        treeOperate(tree_object.left, taskList, masterlist)
    if tree_object.right != 'Null':
        treeOperate(tree_object.right, taskList, masterlist) 
    if tree_object.top != 'Null':
        treeOperate(tree_object.top, taskList, masterlist)
    if tree_object.down != 'Null':
        treeOperate(tree_object.down, taskList, masterlist)
    taskList.pop()
    return masterlist

def preProcessJson():
    f = open("data.json", "r")
    str = f.read()
    f.close()

    liststr = (str[1:-1]).split(',')
    lenliststr = len(liststr)
    dict1 = [0 for i in range(1000)]
    
    start = '@'
    end = '='

    start_s = '+'
    end_s = '>'

    start_e = '>'
    end_e = '"'

    arrayop = np.zeros((1000, 2))

    count = 0
    for i in range(0,lenliststr):
        if (liststr[i][1]) == '@':
            dict1[int((liststr[i].split(start))[1].split(end)[0])] = liststr[i][4:-1]
        if liststr[i][4:] == "Start":
            start_node = (liststr[i].split(start))[1].split(end)[0]
        if (liststr[i][1]) == '+':
            arrayop[count, 0] = (int((liststr[i].split(start_s))[1].split(end_s)[0]))
            arrayop[count, 1] = (int((liststr[i].split(start_e))[1].split(end_e)[0]))
            count  = count + 1
    
    tree_object = Tree()
    tree_object = treer(tree_object, dict1, arrayop, start_node)








#print(tree_object.data)
#print((tree_object.left).data)
#print(((tree_object.left).left).data)
#print(((tree_object.left).right).data)
#print((((tree_object.left).left).left).data)


taskList = []
masterlist = []
masterlist = treeOperate(tree_object, taskList, masterlist)
print(masterlist[0][1])