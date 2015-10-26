#To download the dataset: wget http://www-bcf.usc.edu/~gareth/ISL/Credit.csv
#In this case, there is a credit card data named Credit.csv, which contain the Income, Limit, Rating, and so on information of users, assuame that a new user has: Imcome: 70.00, Limit:5600, Rating: 420, Cards: 3, Age, 50, Education: 14, Gender: Female, Student: No, Married: Yes, Ethnicity: Caucasian. Make a prediction of her Balance, using SVM
import numpy as np
import cv2
import csv

#define a function to replace value
def replace(l, c, X, Y):
	for i,v in enumerate(l[:, c]):
		if v == X:
			l[i, c] = Y

#read the data to an array
reader = csv.reader(file('/home/Credit.csv', 'rb'), delimiter=',')
list = list(reader)
oldArray = np.array(list)

#replace'Yes' to '1', 'No' to '0', 'Female' to '0', ' Male' to '1', 'African American' to '0', 'Asian' to '1', 'Caucasian' to '2'
replace(oldArray, 7, 'Female', 0)
replace(oldArray, 7, ' Male', 1)
replace(oldArray, 8, 'No', 0)
replace(oldArray, 8, 'Yes', 1)
replace(oldArray, 9, 'No', 0)
replace(oldArray, 9, 'Yes', 1)
replace(oldArray, 10, 'African American', 0)
replace(oldArray, 10, 'Asian', 1)
replace(oldArray, 10, 'Caucasian', 2)

#reshape the array
oldArray = oldArray[1: oldArray.shape[0], 1: oldArray.shape[1]]

#set training data
trainData = oldArray[:, 0: oldArray.shape[1] - 1].astype(np.float32)

#set response data
response = oldArray[:, oldArray.shape[1] - 1].astype(np.float32)
response = np.array([response]).T

#set svm parameters
svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 3, gamma = 6 )
svm = cv2.SVM()

#svm training
svm.train(trainData,response, params=svm_params)

#set newcomer
nwcmr = np.array([70, 5600, 420, 3, 50, 14, 0, 0, 1, 2]).astype(np.float32)

#make prediction
result = svm.predict(nwcmr)
print result
