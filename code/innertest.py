import sys
import pandas as p
from sklearn import linear_model
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


from sklearn import cross_validation
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import threading

extraction_method=0


paths = ['../data/newdata1.csv', '../data/newdata2.csv']	
#paths = ['../data/mytrainStem.csv', '../data/mytestStem.csv']	

t = p.read_csv(paths[0])
mm = t.as_matrix()
#print len(mm), len(mm[0])
#exit()



# Partition the trainning set
# firstPart = len(mm) * 4 / 5
# col = 'sex,city,build,race,pct,pos_knife,pos_handgun,pos_rifle,pos_assault,pos_machgun,pos_otherweap,pos_illegal,cs_susp_obj,rf_unseasonal_attire,cs_crime_attire,cs_susp_bulge,cs_match_desc,cs_recon,cs_lookout,cs_drug_trade,cs_covert,rf_violent,cs_violent,ac_crime_area,ac_crime_time,ac_crime_assoc,ac_avoid_cops, label\n'
# filehandler = open('newdata1.csv', 'w')
# filehandler.write(col)
# for i in range(firstPart):
# 	for j in range(len(mm[0])):
# 		if j == len(mm[0]) - 1:
# 			filehandler.write(str(mm[i][j]) + '\n')
# 		else:
# 			filehandler.write(str(mm[i][j]) + ',')
# filehandler.close()
# filehandler = open('newdata2.csv', 'w')
# filehandler.write(col)
# for i in range(firstPart, len(mm)):
# 	for j in range(len(mm[0])):
# 		if j == len(mm[0]) - 1:
# 			filehandler.write(str(mm[i][j]) + '\n')
# 		else:
# 			filehandler.write(str(mm[i][j]) + ',')
# filehandler.close()	

# exit()

threshold = -0.7981


featureNum = len(mm[0])
onePart = mm[mm[:,featureNum-1] == 1,:]
otherPart = mm[mm[:,featureNum-1] == -1, :]
print 'start nine times'
newOnePart = np.repeat(otherPart, 10, axis = 0)
print 'start concatenate'
#print len(newOnePart)
newMM = np.concatenate((onePart, newOnePart), axis=0)
print len(newMM)
#exit()
X = newMM[:,:-1]
y = newMM[:,-1]

#X = mm[:,:-1]
#y = mm[:,-1]
# print 'transform non value'
# for i in range(len(X)):
# 	for j in range(4):
# 		if X[i][j] == 0:
# 			X[i][j] = -2


#imp = Imputer(missing_values=-2, strategy='mean', axis=0)
#oldX = imp.fit_transform(X)


enc = preprocessing.OneHotEncoder()
enc.__init__(categorical_features=[0,1,2,3,4])
trainX = enc.fit_transform(X).toarray()

#pca = PCA(n_components=120)
#X_rmodel = pca.fit(newX)

#trainX = X_rmodel.transform(newX)


t2 = p.read_csv(paths[1])
mm2 = t2.as_matrix()
test = mm2[:,:-1]
# for i in range(len(test)):
# 	for j in range(4):
# 		if test[i][j] == 0:
# 			test[i][j] = -2

# oldTest = imp.fit_transform(test)
predictTest = enc.fit_transform(test).toarray()
#predictTest = X_rmodel.transform(newTest)

#exit()
#pca = PCA(n_components=2)
#X_r = pca.fit(X).transform(X)


testResut = mm2[:,-1]

#clf = svm.LinearSVC()

clf = linear_model.Ridge(alpha = 0.55)
#clf = linear_model.BayesianRidge()
#
#clf = QuadraticDiscriminantAnalysis()
#clf = NearestCentroid()
#clf = GaussianNB()
#clf = tree.DecisionTreeRegressor()



print 'start fit'
clf.fit(trainX, y)

print 'start predict'
test_prediction = clf.predict(predictTest)
print 'finish predicting'
print test_prediction[:10]
errNum = 0
for j in range(len(test_prediction)):
	if test_prediction[j] < threshold and testResut[j] == 1:
		errNum += 10
	elif test_prediction[j] >= threshold and testResut[j] == -1:
		errNum += 1
print '%s\n' % (float(errNum) )

exit()




filehandler = open('newresult.csv', 'w')
filehandler.write('Id,Prediction\n')
for i in range(len(test_prediction)):
	#filehandler.write('%s,%s\n' % (i+1, test_prediction[i]))

	if test_prediction[i] < -0.8:
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