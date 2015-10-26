#in this code, we have a data of automobiles containing their mpg, cylinderss, displacement, horsepower, weight, acceleration, year, origin, and name, we want to use kmeans classification to make 5 classes of similar cars, and output their names. To get Auto.csv, use wget http://www-bcf.usc.edu/~gareth/ISL/Auto.csv

import numpy as np
import cv2
import csv


#read the file Auto.csv
reader = csv.reader(file('/YourPath/Auto.csv', 'rb'), delimiter=',')
list = list(reader)
#out put csv to an array
oldArray = np.array(list)
newArray = []
#there are some invalid data '?' in Auto data, skip it
for line in range(1, oldArray.shape[0]):
	if '?' not in oldArray[line, :]:
		newArray.append(oldArray[line, :])

newArray = np.array(newArray)

#use the auto parameters except their name as input
input = newArray[:, 0:7].astype(np.float32)

#make a criteria, which is the accuracy of the kmeans, then do kmeans with k = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center=cv2.kmeans(input,5,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

#get the results, which are 5 cluster of cars
result = []
for tmp in range(0, 5):
	result.append(newArray[label.ravel() == tmp, 8])
result = np.array(result)
for tmp in range(0, 5):
	print result[tmp]
	print


