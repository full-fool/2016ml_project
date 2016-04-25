import sys
import pandas as p
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

import re
import threading
import csv

extraction_method=0


paths = ['../data/data.csv', '../data/quiz.csv']	
#paths = ['../data/mytrainStem.csv', '../data/mytestStem.csv']	
print 'opening data'
# t = p.read_csv(paths[0])
# mm = t.as_matrix()

# print len(mm)
# print len(mm[0])

with open(paths[0]) as csvfile:
	reader = csv.DictReader(csvfile)

	vec = DictVectorizer()

	mm = vec.fit_transform(reader).toarray()
	print len(mm)
	print len(mm[0])
#exit()




# featureNum = len(mm[0])
# onePart = mm[mm[:,featureNum-1] == 1,:]
# otherPart = mm[mm[:,featureNum-1] == -1, :]
# print 'start nine times'
# newOnePart = np.repeat(onePart, 10, axis = 0)
# print 'start concatenate'
# #print len(newOnePart)
# newMM = np.concatenate((newOnePart, otherPart), axis=0)
# print len(newMM)



#print len(mm), len(mm[0])
#exit()
print 'processing data'
X = mm[:,:-1]
y = mm[:,-1]

# print 'transform non value'
# for i in range(len(X)):
# 	for j in range(4):
# 		if X[i][j] == 0:
# 			X[i][j] = -2

# imp = Imputer(missing_values=-2, strategy='mean', axis=0)
# oldX = imp.fit_transform(X)


# indices of categorical features: 0(0), 2(5), 3(7), 4(8), 5(9), 7(14), 8(16), 9(17), 10(18), 11(20), 
# 12(23), 13(25), 14(26), 44(56), 45(57), 46(58),  

# enc = preprocessing.OneHotEncoder()
# enc.__init__(categorical_features=[0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 44, 45, 46])
# newX = enc.fit_transform(X).toarray()


# t2 = p.read_csv(paths[1])
# mm2 = t2.as_matrix()
# test = mm2

with open(paths[1]) as csvfile:
	reader = csv.DictReader(csvfile)

	vec = DictVectorizer()

	test = vec.fit_transform(reader).toarray()
	print len(test)
	print len(test[0])
exit()
# for i in range(len(test)):
# 	for j in range(4):
# 		if test[i][j] == 0:
# 			test[i][j] = -2

#oldTest = imp.fit_transform(test)
#newTest = enc.fit_transform(test).toarray()


clf = linear_model.Ridge (alpha = 0.55)
#clf = linear_model.BayesianRidge()
print 'start to fit'
clf.fit(X,y)

print 'start to predict'
test_prediction = clf.predict(test)
#print len(test_prediction)
filehandler = open('newresult.csv', 'w')
filehandler.write('Id,Prediction\n')
for i in range(len(test_prediction)):
	#filehandler.write('%s,%s\n' % (i+1, test_prediction[i]))

	if test_prediction[i] < 0:
		filehandler.write('%s,%s\n' % (i+1, -1))
	else:
		filehandler.write('%s,%s\n' % (i+1, 1))
filehandler.close()
exit()







print "start extraction"
# delete the max_feature, score is higher
if extraction_method==0:
	print "Tfidf......"
	

	tfidf = TfidfVectorizer(strip_accents='unicode', analyzer='word',  ngram_range=(1,3))


	tfidf.fit(t['tweet'])
	X = tfidf.transform(t['tweet'])
	test = tfidf.transform(t2['tweet'])
	y = np.array(t.ix[:,4:])
else:
	print "Count......"
	countmethod = CountVectorizer(strip_accents='unicode', analyzer='word', lowercase=True)
	countmethod.fit(t['tweet'])
	X = countmethod.transform(t['tweet'])
	test = countmethod.transform(t2['tweet'])
	y = np.array(t.ix[:,4:])


print "extraction done"

print "start fit"
#0.6=0.15134 0.8=0.15148 0.4=0.15140 best 0.65=0.15136 0.55=0.15133
clf = linear_model.Ridge (alpha = 0.55)

clf.fit(X,y)
print "fit done"
print "start prediction"
test_prediction = clf.predict(test)



for i in xrange(len(test_prediction)):
	for j in xrange(len(test_prediction[i])):  #0.015,0.99 best
		if test_prediction[i][j] <= 0.01:
			test_prediction[i][j] = 0
		elif test_prediction[i][j] >= 0.99:
			test_prediction[i][j] = 1

	# normalize attitude
	summary = 0
	for j in xrange(0, 5):
		summary += test_prediction[i][j]
	if (summary != 0):
		for j in xrange(0, 5):
			test_prediction[i][j] /= summary
	# normalize time
	summary = 0
	for j in xrange(5, 9):
		summary += test_prediction[i][j]
	if (summary != 0):
		for j in xrange(5, 9):
			test_prediction[i][j] /= summary
print "prediction done"


coname = ['id','s1','s2','s3','s4','s5','w1','w2','w3','w4','k1','k2','k3','k4','k5','k6','k7','k8','k9','k10',
	'k11','k12','k13','k14','k15']

first = np.matrix(coname)
print "start writing"
prediction = np.array(np.hstack([np.matrix(t2['id']).T, test_prediction])) 
col = '%i,' + '%f,'*23 + '%f'

np.savetxt('../data/myresult0.55-0.009.csv', prediction ,col, delimiter=',')
print "writing done"