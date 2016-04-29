import sys
#import pandas as p
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import svm
import re
import threading
import csv
trainDataPath = 'newdata.csv'
testDataPath = 'newquiz.csv'
# numericAttriList = ['2', '11', '27','28','29','30','31','32','33','34','35','36','37',\
# 	'38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55',\
# 	'59', '60', '62', '63', '64']


categoricalFeatureDictList = []

#categoricalAttriIndexList = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 44, 45, 46]
#feature 26 to 39 contains {0,1,2}
categoricalAttriIndexList = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14]+range(26,40)+[ 44, 45, 46]
#categoricalFeatureValueNumList = [13, 112, 2, 13, 13, 112, 2, 13, 145, 4, 3031, 4, 138, 102, 102, 2090]

#numericAttriIndexList = [1, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 50, 51]

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

mm = np.loadtxt(open(trainDataPath,"rb"),delimiter=",",skiprows=1)
mm = mm.astype('int')

#t = p.read_csv(trainDataPath)
#mm = t.as_matrix()
trainLabel = mm[:, -1]
trainFeatureSpace = mm[:, :-1]


t2 = np.loadtxt(open(testDataPath,"rb"),delimiter=",",skiprows=1)
testFeatureSpace = t2.astype('int')
#t2 = p.read_csv(testDataPath)
#testFeatureSpace = t2.as_matrix()

print len(trainFeatureSpace) #126837
print len(trainFeatureSpace[0]) #52
#print trainFeatureSpace[0]
#exit()


enc = OneHotEncoder()
print 'OneHotEncoder'
#enc.__init__(n_values = categoricalFeatureValueNumList, categorical_features = categoricalAttriIndexList)
enc.__init__(categorical_features = categoricalAttriIndexList)
newTrainFeatureSpace = enc.fit_transform(trainFeatureSpace).toarray()
newTestFeatureSpace = enc.transform(testFeatureSpace).toarray()
#print len(newTrainFeatureSpace[0])
print len(newTestFeatureSpace[0])

#print 'PCA'
#pca = PCA(n_components=500)
#pca.fit(newTrainFeatureSpace)
#newTrainFeatureSpace = pca.transform(newTrainFeatureSpace)
#newTestFeatureSpace = pca.transform(newTestFeatureSpace)
#print len(newTrainFeatureSpace)
#print len(newTrainFeatureSpace[0])
# print len(newTestFeatureSpace)
# print len(newTestFeatureSpace[0])
# exit()

#print 'scaling'
#scaler = StandardScaler().fit(newTrainFeatureSpace)
#newTrainFeatureSpace = scaler.transform(newTrainFeatureSpace)
#newTestFeatureSpace = scaler.transform(newTestFeatureSpace)


# display the relative importance of each attribute
#model = ExtraTreesClassifier()
#model.fit(newTrainFeatureSpace, trainLabel)
#print(model.feature_importances_)


#clf = linear_model.LogisticRegression() #0.89179 #after scaling 0.88878
#clf = naive_bayes.GaussianNB() #0.55772
#clf = neighbors.KNeighborsClassifier() #0.88129
#clf = svm.SVC() #too slow
#clf = svm.LinearSVC() #0.82141 #after scaling 0.86674


#clf = linear_model.Ridge() #0.88582 #alpha =0.57 slightly better than 0.55,but worse than no paramter
#Ridge no PCA 0.88582
#500 0.88325 not sure...
#1000 0.88605
#1500 0.88653
#2000 0.88643
#3000 0.88594

#clf = tree.DecisionTreeClassifier()
#gini d10 0.86985 d15 0.90082 d20 0.91368 d25 0.91868 d30 0.91984 d35 0.91931 d40 0.92061 d45 0.92019 d50 0.92214 d55 0.92043
#gini d200 0.92155 d300 0.92189 d400 0.92175
#entropy d80 0.92368 d100 0.92317 d110 0.92307 d130 0.92295 d150 0.92250 d170 0.92408 d180 0.92506 d190 0.92262 d200 0.92372 d250 0.92402 d300 0.92427
#PCA1000 entropy0.89548 gini0.89552
#PCA1500 gini 0.89469
#PCA3000 gini 0.89429
#PCA4000 entropy 0.89041
#PCA5000 entropy 0.89023
#entropy 180 min_samples_split=5 0.92147
#min_samples_split=4 entropy 0.92305 gini 0.92140
#criterion='entropy' 0.92392  /'gini' 0.92073
#all features 0.92387

clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy'),algorithm="SAMME",n_estimators=40)
#0.92940
#entropy gini
#SAMME SAMME.R
#n_estimators




print 'start to fit'
clf.fit(newTrainFeatureSpace, trainLabel)

print 'start to predict'
test_prediction = clf.predict(newTestFeatureSpace)
#print len(test_prediction)

print 'start to write result'
filehandler = open('newresult.csv', 'w')
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

#cross validation
print 'cross validation score:'
X_train, X_test, y_train, y_test = cross_validation.train_test_split(newTrainFeatureSpace, trainLabel, test_size=0.4,random_state=0)
clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)
count = 0
for i in range(len(y_prediction)):
    if y_prediction[i] * y_test[i] > 0:
        count += 1
print count
score = float(count)/float(len(y_prediction))
#score = cross_validation.cross_val_score(clf, newTrainFeatureSpace, trainLabel, cv=3)
print score
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
