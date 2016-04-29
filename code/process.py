import sys
import pandas as p
from sklearn import linear_model
import numpy as np
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import re
import threading
import csv
trainDataPath = 'newdata.csv'
testDataPath = 'newquiz.csv'
# numericAttriList = ['2', '11', '27','28','29','30','31','32','33','34','35','36','37',\
# 	'38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55',\
# 	'59', '60', '62', '63', '64']


categoricalFeatureDictList = []
singleValueIndexList = [17, 19, 20, 23]
categoricalAttriIndexList = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 44, 45, 46]	
categoricalFeatureValueNumList = [13, 112, 2, 13, 13, 112, 2, 13, 145, 4, 3031, 4, 138, 102, 102, 2090]
cateNumericIndexList = [1, 6, 15, 16, 18,21,22,24,25,26,27,28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,49,50,51]
numericAttriIndexList = [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51]

# Analyse the feature type

#currentIndex = 0
# for line in open('allfeatures.txt'):
# 	if not 'numeric' in line:
# 		#categoricalAttriIndexList.append(currentIndex)
# 		#categoricalFeatureValueNumList.append(len(line.split()) - 1)
# 		features = line.split()[1:]
# 		tmpDict = {}
# 		for i in range(len(features)):
# 			tmpDict[features[i]] = i
# 		categoricalFeatureDictList.append(tmpDict)


	#currentIndex += 1
# print len(categoricalFeatureDictList)
# print len(categoricalAttriIndexList)
# print categoricalFeatureDictList[0]
# exit()


# t = p.read_csv(testDataPath)

# mm = t.as_matrix()
# for i in range(len(mm)):
# 	for j in range(len(categoricalAttriIndexList)):

# 		mm[i][categoricalAttriIndexList[j]] = categoricalFeatureDictList[j]['v' + mm[i][categoricalAttriIndexList[j]]]

# filehandler = open('newquiz.csv', 'w')
# filehandler.write('0,2,5,7,8,9,11,14,16,17,18,20,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,62,63,64,label\n')
# colNum = len(mm[0])
# for i in range(len(mm)):
# 	for j in range(colNum):
# 		filehandler.write(str(mm[i][j]))
# 		if j == colNum - 1:
# 			filehandler.write('\n')
# 		else:
# 			filehandler.write(',')
# filehandler.close()
# exit()
#print len(mm)
#print len(mm[0])
#exit()
def preProcessData(trainFeatureMatrix, testFeatureMatrix):
	totalFeatureNum = 52
	singleValueIndexList = [17, 19, 20, 23]
	categoricalAttriIndexList = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 44, 45, 46]	
	categoricalFeatureValueNumList = [13, 112, 2, 13, 13, 112, 2, 13, 145, 4, 3031, 4, 138, 102, 102, 2090]
	cateNumericIndexList = [1, 6, 15, 16, 18,21,22,24,25,26,27,28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,49,50,51]

	numericAttriIndexList = [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51]
	
	# for i in range(len(trainFeatureSpace[0])):
	# 	if not i in categoricalAttriIndexList:
	# 		#print 'numerical', i, len(list(set(trainFeatureSpace[:,i])))
	# 		print '%s, numerical, train: %s, test:%s' % (i, len(list(set(trainFeatureMatrix[:,i]))), len(list(set(testFeatureMatrix[:,i]))))
	# 	else:
	# 		print '%s, categorical, train: %s, test:%s' % (i, len(list(set(trainFeatureMatrix[:,i]))), len(list(set(testFeatureMatrix[:,i]))))
	


	tempResultMatrix =  np.concatenate((trainFeatureMatrix, testFeatureMatrix), axis=0)

	# print len(trainFeatureMatrix), len(trainFeatureMatrix[0])
	# print len(testFeatureMatrix), len(testFeatureMatrix[0])
	# print len(tempResultMatrix), len(tempResultMatrix[0])

	# exit()

	# for i in range(len(trainFeatureMatrix)):
	# 	for j in range(len(trainFeatureMatrix[0])):
	# 		if j in cateNumericIndexList:
	# 			trainFeatureMatrix[i][j] = int(trainFeatureMatrix[i][j])

	# for i in range(len(testFeatureMatrix)):
	# 	for j in range(len(testFeatureMatrix[0])):
	# 		if j in cateNumericIndexList:
	# 			testFeatureMatrix[i][j] = int(testFeatureMatrix[i][j])

	#selectedFeatureList = []
	# for i in range(53):
	# 	if not i in singleValueIndexList:
	# 		selectedFeatureList.append(i)

	# trainFeatureMatrix = trainFeatureMatrix[ : , selectedFeatureList]
	# testFeatureMatrix = testFeatureMatrix[ : , selectedFeatureList]
	from sklearn.preprocessing import OneHotEncoder
	enc = OneHotEncoder()
	enc.__init__(categorical_features = categoricalAttriIndexList + cateNumericIndexList)
	enc.fit(tempResultMatrix)
	trainFeatureMatrix = enc.transform(trainFeatureMatrix).toarray()
	testFeatureMatrix = enc.transform(testFeatureMatrix).toarray()

	print 'old feature num is ', len(trainFeatureMatrix[0]), len(testFeatureMatrix[0])


	#tempResultMatrix =  np.concatenate((trainFeatureMatrix, testFeatureMatrix), axis=0)


	sel = VarianceThreshold()
	sel.fit(trainFeatureMatrix)
	trainFeatureMatrix = sel.transform(trainFeatureMatrix)
	testFeatureMatrix = sel.transform(testFeatureMatrix)
	print 'new feature num is ', len(trainFeatureMatrix[0]), len(testFeatureMatrix[0])
	#exit()
	return trainFeatureMatrix, testFeatureMatrix







t = p.read_csv(trainDataPath)

mm = t.as_matrix()
#print mm[1:3,]


t2 = p.read_csv(testDataPath)


trainLabel = mm[:, -1]
trainFeatureSpace = mm[:, :-1]
# for i in range(len(trainFeatureSpace[0])):
# 	if not i in categoricalAttriIndexList:
# 		print 'numerical', i, len(list(set(trainFeatureSpace[:,i])))
# 	else:
# 		print 'categorical', i, len(list(set(trainFeatureSpace[:,i])))

# exit()



testFeatureSpace = t2.as_matrix()
newTrainFeatureSpace, newTestFeatureSpace =  preProcessData(trainFeatureSpace, testFeatureSpace)


# pca = PCA()
# pca.fit(newTrainFeatureSpace)
# newTrainFeatureSpace = pca.transform(newTrainFeatureSpace)

# newTestFeatureSpace = pca.transform(newTestFeatureSpace)

#print 'after pca, the feature num is ' , len(newTrainFeatureSpace[0]), len(newTestFeatureSpace[0])


#print len(trainFeatureSpace)
#print len(trainFeatureSpace[0])

# from sklearn.preprocessing import OneHotEncoder
# enc = OneHotEncoder()

# enc.__init__(n_values = categoricalFeatureValueNumList, categorical_features = categoricalAttriIndexList)
# newTrainFeatureSpace = enc.fit_transform(trainFeatureSpace).toarray()
# newTestFeatureSpace = enc.fit_transform(testFeatureSpace).toarray()
# print 'old feature num is ', len(newTrainFeatureSpace[0]), len(newTestFeatureSpace[0])
	

# sel = VarianceThreshold()
# sel.fit(newTrainFeatureSpace)
# newTrainFeatureSpace = sel.transform(newTrainFeatureSpace)
# newTestFeatureSpace = sel.transform(newTestFeatureSpace)
# print 'new feature num is ', len(newTrainFeatureSpace[0]), len(newTestFeatureSpace[0])



# pca = PCA(n_components=500)
# pca.fit(newTrainFeatureSpace)
# newTrainFeatureSpace = pca.transform(newTrainFeatureSpace)

# newTestFeatureSpace = pca.transform(newTestFeatureSpace)
#print len(newTrainFeatureSpace)
# print len(newTrainFeatureSpace[0])
# print len(newTestFeatureSpace)
# print len(newTestFeatureSpace[0])
# exit()



#clf = linear_model.RidgeCV(alphas=(0.1, 0.55, 1, 10), normalize=True, cv=10)
#clf = linear_model.Ridge ()
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                        algorithm="SAMME",
                        n_estimators=60)
#clf = tree.DecisionTreeClassifier(max_depth=250)

print 'start to fit'
#clf.fit(newTrainFeatureSpace, trainLabel)
bdt.fit(newTrainFeatureSpace, trainLabel)

print 'start to predict'
#test_prediction = clf.predict(newTestFeatureSpace)
test_prediction = bdt.predict(newTestFeatureSpace)
#print len(test_prediction)

print 'start to write result'
filehandler = open('40adaresult.csv', 'w') 
filehandler.write('Id,Prediction\n')
for i in range(len(test_prediction)):
	#filehandler.write('%s,%s\n' % (i+1, test_prediction[i]))
	#filehandler.write('%s,%s\n' % (i+1, test_prediction[i]))

	if test_prediction[i] < 0:
		filehandler.write('%s,%s\n' % (i+1, -1))
	else:
		filehandler.write('%s,%s\n' % (i+1, 1))
filehandler.close()
exit()


# reader = None
# resultDictList = []
# with open(testDataPath) as csvfile:
# 	reader = csv.DictReader(csvfile)
# 	for row in reader:
# 		tmpDict = row
# 		for numericAttri in numericAttriList:
# 			tmpDict[numericAttri] = float(tmpDict[numericAttri])
# 		resultDictList.append(tmpDict)

# 		#print(row['0'], row['label'])
# 	#print len(reader)
# #print reader[0]

# resultMatrix = vec.fit_transform(resultDictList).toarray()
# print len(resultMatrix) , len(resultMatrix[0])
# print resultMatrix[0]



t = p.read_csv(trainDataPath)
mm = t.as_matrix()
print len(mm[0])
exit()
