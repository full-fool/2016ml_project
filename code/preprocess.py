import sys
import pandas as p
from sklearn import linear_model
import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
import threading

extraction_method=0


paths = ['../data/quiz.csv', '../data/quiz.csv']	
#paths = ['../data/mytrainStem.csv', '../data/mytestStem.csv']	

t = p.read_csv(paths[0])
#print len(t['sex'])
mm = t.as_matrix()


for i in range(len(mm)):
	if mm[i][0] == 'M':
		mm[i][0] = 1
	elif mm[i][0] == 'F':
		mm[i][0] = 2
	else:
		mm[i][0] = 0

	if mm[i][1] == 'Z':
		mm[i][1] = 0
	else:
		mm[i][1] = int(mm[i][1].split('_')[1])


	if mm[i][2] == 'Z':
		mm[i][2] = 0
	else:
		mm[i][2] = int(mm[i][2].split('_')[1])

	if mm[i][3] == 'Z':
		mm[i][3] = 0
	else:
		mm[i][3] = int(mm[i][3].split('_')[1])

	mm[i][4] = int(mm[i][4].split('_')[1])



col = 'sex,city,build,race,pct,pos_knife,pos_handgun,pos_rifle,pos_assault,pos_machgun,pos_otherweap,pos_illegal,cs_susp_obj,rf_unseasonal_attire,cs_crime_attire,cs_susp_bulge,cs_match_desc,cs_recon,cs_lookout,cs_drug_trade,cs_covert,rf_violent,cs_violent,ac_crime_area,ac_crime_time,ac_crime_assoc,ac_avoid_cops\n'
filehadler = open('../data/newquiz.csv', 'w')
filehadler.write(col)
for i in range(len(mm)):
	for j in range(len(mm[i])):
		if j == len(mm[0]) - 1:
			filehadler.write(str(mm[i][j]) + '\n')
		else:
			filehadler.write(str(mm[i][j]) + ',')
filehadler.close()




#np.savetxt('../data/result.csv', t ,col, delimiter=',')
exit()