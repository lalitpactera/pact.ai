import pandas as pd
import numpy as np
import sys
from RemoveNans import *
from RemoveDuplicates import *
from NormalizeColumns import *
from CategoricalColumns import *

def main(argv):
	filename        = argv[1]
	task            = argv[2]
	
	print(filename)
	
	#data = pd.read_csv(filename)
	task_list = list(task)
	
	print(task_list)
	
	count = 0
	
	for i in range(0,np.alen(task_list)):
		
		if task_list[i] == 'A':
			input_size = (256,256,1)
			inputs = Input(input_size)
		if task_list[i] == 'B':
			kernel = 3
			if count == 0 
				conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
			else
				conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
			count = count + 1
		if task_list[i] == 'C':
			conv_1 = BatchNormalization()(conv_1)
		if task_list[i] == 'E':
			conv_1 = Activation("relu")(conv_1)
		if task_list[i] == 'F':
			conv_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)
		if task_list[i] == 'G':
			conv_1 = Dropout(pool_size=(2, 2))(conv_1)
		if task_list[i] == 'H':
			conv_1 = Concatenate(pool_size=(2, 2))(conv_1)
		if task_list[i] == 'I':
			conv_1 = Reshape(pool_size=(2, 2))(conv_1)
		if task_list[i] == 'J':
			conv_1 = Reshape(pool_size=(2, 2))(conv_1)
		 
		
	data.to_csv('results.csv')

if __name__ == "__main__":
    main(sys.argv)