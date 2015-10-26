1,
#To get the dataset ticdata2000.txt: wget http://kdd.ics.uci.edu/databases/tic/ticdata2000.txt
#To get the dataset ticeval2000.txt: wget http://kdd.ics.uci.edu/databases/tic/ticeval2000.txt
#The Insurance Company Benchmark wants to know who will be interested in buying a caravan insurance policy, their client data(ticdata2000.txt) has 86 colums, at which last colum use 0 or 1 to present that the client is interested in buying caravan insurance or not, now use these data as training data, to predict the caravan status of new comers with their data containing 85 colums, use KNN; there is a new comer with his prameters: 22 1 2 3 5 0 7 0 2 6 0 3 3 2 4 2 3 4 0 2 0 3 4 0 3 2 0 5 0 0 9 7 2 0 7 2 4 2 4 0 0 3 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
#use SVM to predict if he wants to buy caraven


import numpy as np
import cv2
import csv

#to read the ticdata2000.txt and store it in an array
with open('YourPath/ticdata2000.txt','r') as f:
	data = [x.strip().split('\t') for x in f]
data = np.array(data)

#set up training data, which contains 0 to 84 column
trainData = data[:, 0:85].astype(np.float32)

#set up testing data, which contains the 85 column
targetData = data[:, 85].astype(np.float32)
target = np.array([target]).T

#use knn to train the model
knn = cv2.KNearest()
knn.train(trainData, targetData)

#read the ticeval2000.txt as array
with open('YouPath/ticeval2000.txt','r') as f:
	testData = [x.strip().split('\t') for x in f]
testData = np.array(testData).astype(np.float32)

#use knn to predict the 85 column of the ticeval2000.txt, with k = 1
ret, results, neighbours ,dist = knn.find_nearest(testData, 1)
print results

#set up new comer
newcomer = np.array([
22, 1, 2, 3, 5, 0, 7, 0, 2, 6, 0, 3, 3, 2, 4, 2, 3, 4, 0, 2, 0, 3, 4, 0, 3, 2, 0, 5, 0, 0, 9, 7, 2, 0, 7, 2, 4, 2, 4, 0, 0, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).astype(np.float32)

#set svm parameters
svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )


#use svm to train the data of ticdata2000.txt
svm = cv2.SVM()
svm.train(trainData,targetData, params=svm_params)

#make prediction of new comer
result = svm.predict(newcomer)#svm doesn't support predict_all on my board
print result
