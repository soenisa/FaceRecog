import os
import shutil
from pca_example import pca_main
from scipy.misc import imread
import numpy as np

def SSD(v1,v2):
    '''Sum of Squared Differences function'''
    return np.sum(np.linalg.norm(v1 - v2))

def strip_name(name):
	'''Strips an image of its extension and removes numbers to identify
	the actor pictured in the image'''
	name = ''.join([i for i in name if not i.isdigit()]).lower()
	name = name.replace(".jpg", "")
	name = name.replace(".jpeg", "")
	name = name.replace(".png", "")
	return name

act = ['eckhart',  'sandler',   'brody',  'anders',    'benson',    'applegate',    'agron',  'anderson']
masc = ['eckhart', 'sandler', 'brody'] # Masculine featured actors


trainSet = {}
validSet = {}
testSet = {}

if __name__=='__main__':
	# Take cropped images and sort them into train, valid and test
	# directories
	for f in os.listdir("cropped"):
		for a in act:
			fr = strip_name(f)
			if a == fr:
				if a not in trainSet:
					trainSet[a] = []
					testSet[a] = []
					validSet[a] = []

				if len(trainSet[a]) < 100:
					trainSet[a].append(f)
					if not os.path.exists("train"):
						os.makedirs("train")
					shutil.copy("cropped/"+f, "train")
				elif len(validSet[a]) < 10:
					validSet[a].append(f)
					if not os.path.exists("valid"):
						os.makedirs("valid")
					shutil.copy("cropped/"+f, "valid")
				elif len(testSet[a]) < 10:
					testSet[a].append(f)
					if not os.path.exists("test"):
						os.makedirs("test")
					shutil.copy("cropped/"+f, "test")
				else:
					break

def actor_recog():
	'''Loops through k values to find best number of eigenfaces
	for best correctness using "valid" set. Then uses that to
	identify actors in the"test" directory. Returns Correctness
	of k values, all failed	test set matches and correctness of
	test set matches.'''
	correct = {}

	# Go through each k value for the valid set and collect their correctness
	k = [2, 5, 10, 20, 50, 80, 100, 150, 200]
	for kval in k:
		ccount = 0
		for valid in os.listdir("valid"):
			if valid[-3:] != ".db":
				validIMG = imread("valid/"+valid)

				V = pca_main("train/")
				Xpca = V[:kval,:] * validIMG.flatten()

				trainDir = os.listdir("train")
				Ypca = V[:kval,:] * imread("train/"+trainDir[0]).flatten()
				minSSD = SSD(Xpca, Ypca)
				bestMatch = ""
				for train in trainDir:
					if train[-3:] != ".db":
						trainIMG = imread("train/"+train)
						Ypca = V[:kval,:] * trainIMG.flatten()
						ssd = SSD(Xpca, Ypca)
						if ssd < minSSD:
							minSSD = ssd
							bestMatch = train	
				if strip_name(bestMatch) == strip_name(valid):
					ccount+=1
		correct[str(kval)] = ccount


	# Find the k value with the best match stats
	bestK = 0
	bestC = 0
	for kval in correct:
		if correct[str(kval)] > bestC:
			bestK = kval
			bestC = correct[str(kval)]

	# Use best k value to recognize test set
	ccount = 0
	fail = []
	for test in os.listdir("test"):
			if test[-3:] != ".db":
				testIMG = imread("test/"+test)

				V = pca_main("train/")
				Xpca = V[:bestK,:] * testIMG.flatten()

				trainDir = os.listdir("train")
				Ypca = V[:bestK,:] * imread("train/"+trainDir[0]).flatten()
				minSSD = SSD(Xpca, Ypca)
				bestMatch = ""
				for train in trainDir:
					if train[-3:] != ".db":
						trainIMG = imread("train/"+train)
						Ypca = V[:bestK,:] * trainIMG.flatten()
						ssd = SSD(Xpca, Ypca)
						if ssd < minSSD:
							minSSD = ssd
							bestMatch = train	
				if strip_name(bestMatch) == strip_name(test):
					ccount+=1
				else:
					fail.append([bestMatch, test])
	return correct, fail, ccount

def masc_recog():
	'''Loops through k values to find best number of eigenfaces
	for best correctness using "valid" set. Then uses that to
	determine whether actor in "test" directory are masculine or
	feminine featured. Returns Correctness of k values, all failed
	test set matches and correctness of test set matches.'''
	correct = {}

	# Go through each k value for the valid set and collect their correctness
	k = [2, 5, 10, 20, 50, 80, 100, 150, 200]
	for kval in k:
		ccount = 0
		c=0
		for valid in os.listdir("valid"):
			if valid[-3:] != ".db":
				validIMG = imread("valid/"+valid)

				V = pca_main("train/")
				Xpca = V[:kval,:] * validIMG.flatten()

				trainDir = os.listdir("train")
				Ypca = V[:kval,:] * imread("train/"+trainDir[0]).flatten()
				minSSD = SSD(Xpca, Ypca)
				bestMatch = ""
				for train in trainDir:
					if train[-3:] != ".db":
						trainIMG = imread("train/"+train)
						Ypca = V[:kval,:] * trainIMG.flatten()
						ssd = SSD(Xpca, Ypca)
						if ssd < minSSD:
							minSSD = ssd
							bestMatch = train	
				if (strip_name(bestMatch) in masc and strip_name(valid) in masc) or (strip_name(bestMatch) not in masc and strip_name(valid) not in masc):
					ccount+=1
		correct[str(kval)] = ccount

	# return correct


	# Find the k value with the best match stats
	bestK = 0
	bestC = 0
	for kval in correct:
		if correct[str(kval)] > bestC:
			bestK = int(kval)
			bestC = correct[str(kval)]

	# Use best k value to recognize test set
	ccount = 0
	fail = []
	for test in os.listdir("test"):
			if test[-3:] != ".db":
				testIMG = imread("test/"+test)

				V = pca_main("train/")
				Xpca = V[:bestK,:] * testIMG.flatten()

				trainDir = os.listdir("train")
				Ypca = V[:bestK,:] * imread("train/"+trainDir[0]).flatten()
				minSSD = SSD(Xpca, Ypca)
				bestMatch = ""
				for train in trainDir:
					if train[-3:] != ".db":
						trainIMG = imread("train/"+train)
						Ypca = V[:bestK,:] * trainIMG.flatten()
						ssd = SSD(Xpca, Ypca)
						if ssd < minSSD:
							minSSD = ssd
							bestMatch = train	
				if (strip_name(bestMatch) in masc and strip_name(test) in masc) or (strip_name(bestMatch) not in masc and strip_name(test) not in masc):
					ccount+=1
				else:
					fail.append([bestMatch, test])
	return fail, ccount