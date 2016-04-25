filehandler = open('newresult.csv').read().split('\n')
resultFile = open('nnresult.csv', 'w')
for i in range(1, len(filehandler)):
	resultId, score = filehandler[i].split(',')[0], filehandler[i].split(',')[1]
	if float(score) < 0:
		resultFile.write('%s,%s\n' % (resultId, -1))
	else:
		resultFile.write('%s,%s\n' % (resultId, 1))
resultFile.close()
