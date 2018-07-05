#############################################################################
# Capstone Project: Filipino, Thai, and Indonesian Ethnicity Classification #
#############################################################################

import cv2
import sys
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#####################################################################
# Gets the mean of all the pixels then subtract it to all the pixel #
# newMean - output folder											#
# newGray - input folder (location of the grayscaled images) 		#
#####################################################################
def meansub():
	# creation of the folder newMean if it is not yet existing
	if not "newMean" in os.listdir("."):
		os.mkdir("newMean")

	imageList  = []
	imageDir = "newGray/"
	# directory of the grayscaled images 

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)

		# acquiring the mean value
		mean = np.mean(img)

		# subtracting the mean to the pixel values of the image
		newIMG = img - mean
		strIMG = imagePath.split("/")
		outname = "newMean/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)
	
	print("DONE")

##########################################################
# Converts the image into grayscale 					 #
# newGray - output folder								 #
# newPre - input folder (location of the resized images) #
##########################################################
def grayscale():
	# creation of the folder if it is not yet existing
	if not "newGray" in os.listdir("."):
		os.mkdir("newGray")

	imageList  = []
	imageDir = "newPre/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)

		# converting the images from RGB to GRAY		
		newIMG = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		strIMG = imagePath.split("/")
		outname = "newGray/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)
	
	print("DONE")

#################################################
# Flips the images to produce additional images #
# newFlip - output folder 						#
# newPre - input folder (resized images) 		#
#################################################
def augment():
	# creation of the folder if it is not yet existing
	if not "newFlip" in os.listdir("."):
		os.mkdir("newFlip")

	imageList  = []
	imageDir = "newPre/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)

		# flip the images horizontally
		newIMG = np.fliplr(img)

		strIMG = imagePath.split("/")
		outname = "newFlip/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)

	print("DONE")

#####################################
# Resizes the images into 128 x 128	#
# newPre - output folder			#
# Extracted - input folder (crop)	#
#####################################
def resize():
	# creation of the folder if it is not yet existing
	if not "newPre" in os.listdir("."):
		os.mkdir("newPre")

	imageList  = []
	imageDir = "Extracted/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)

		# resizing the images to 128x128
		newIMG = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)

		strIMG = imagePath.split("/")
		outname = "newPre/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)

	print("DONE")

###############################################################
# Detects the faces in the images and gets that specific area #
# Extracted - output folder									  #
# raw - input folder (location of the raw images) 			  #
###############################################################
def crop():
	# creation of the folder if it is not yet existing
	if not "Extracted" in os.listdir("."):
		os.mkdir("Extracted")

	imageList = []
	imageDir = "raw/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for image_path in imageList:

		# reading the face cascale xml file
		cascade="Face_cascade.xml"
		face_cascade=cv2.CascadeClassifier(cascade)

		image=cv2.imread(image_path)

		# for better detection, we must convert the image from RGB to GRAY
		image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(image_grey, scaleFactor=1.16, minNeighbors=5, minSize=(25,25), flags=0)

		for x,y,w,h in faces:
		    sub_img=image[y-10:y+h+10,x-10:x+w+10]
		    os.chdir("Extracted")
		    cv2.imwrite(str(y)+".jpg",sub_img)
		    os.chdir("../")
		    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	#cv2.imshow("Faces Found",image)
	#cv2.waitKey(0)
	print("DONE")

########################
# Preprocess functions #
########################
def preprocess():
	while(1):
		print("Choose")
		print("(1) Crop Training Images")
		print("(2) Resize")
		print("(3) Augment Images")
		print("(4) Convert to Grayscale")
		print("(5) Apply Mean Subtraction")
		
		choice = int(input())

		if choice == 1:
			print("\nCropping...")
			crop()
		elif choice == 2:
			print("\nResizing...")
			resize()
		elif choice == 3:
			print("\nAugmenting...")
			augment()
		elif choice == 4:
			print("\nConverting...")
			grayscale()
		elif choice == 5:
			print("\nApplying...")
			meansub()
		else:
			break

############################
# Feature vector functions #
############################
def featureVector():
	print("Choose")
	print("(1) Train Set")
	print("(2) Test Set")

	choice = int(input())

	if choice == 1:
		print("\nCreating train feature vector...")
		trainFeature()
	elif choice == 2:
		print("\nCreating test feature vector...")
		testFeature()
	else:
		print("\nERROR")

#############################################################
# Creates the feature vector function for the train dataset #
# newMean - location of the preprocessed images 			#
# datasetTrain.csv - output file (feature vector) 			#
#############################################################
def trainFeature():
	# load the HOG descriptor
	hgd = cv2.HOGDescriptor()

	imageList  = []
	imageDir = "newMean/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	featuresVector = np.zeros((len(imageList), 34020))
	# create the featuresVector

	for imageIndex, imagePath in enumerate(imageList):
		img = cv2.imread(imagePath)

		# compute the HOG values of the image
		val = hgd.compute(img)

		# transfer the values to the featuresVector
		for i in range(34020):
			featuresVector[imageIndex][i] = val[i]

	# create a CSV file to dump the train features vector
	csvFile = open("datasetTrain.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for i in range(len(imageList)):
		writer.writerow(featuresVector[i])

	print("DONE")

############################################################
# Creates the feature vector function for the test dataset #
# newTest - location of the preprocessed images 		   #
# datasetTest.csv - output file (feature vector) 		   #
############################################################
def testFeature():
	# load the HOG descriptor
	hgd = cv2.HOGDescriptor()

	imageList  = []
	imageDir = "newTest/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	featuresVector = np.zeros((len(imageList), 34020))

	for imageIndex, imagePath in enumerate(imageList):
		img = cv2.imread(imagePath)

		# compute the HOG values
		val = hgd.compute(img)

		# transfer the values to the featuresVector
		for i in range(34020):
			featuresVector[imageIndex][i] = val[i]

	# create a CSV file to dump the features vector
	csvFile = open("datasetTest.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for i in range(len(imageList)):
		writer.writerow(featuresVector[i])

	print("DONE")

#############################################################
# Trains the chosen model with the datasetTrain.csv 		#
# trainModelOK.sav - output file 							#
#############################################################				
def train():
	labelsMatrix = np.zeros(852)

	# open the target train labels and read and store
	with open("labels.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrix[index] = int(line[0])
	#print(labelsMatrix.shape)

	# read the features vector of the train dataset from the csv file
	datasetTrain = np.genfromtxt("datasetTrain.csv", delimiter=",")
	#print(datasetTrain.shape)

	# model
	linearSVM = LinearSVC(C=0.01, max_iter=10000, loss='hinge', intercept_scaling=1000)
	#linearSVM = LinearSVC(max_iter=100)

	# fir the training to the model
	linearSVM.fit(datasetTrain, labelsMatrix)

	# save the model to a pickle dump
	filename = "trainModelOK.sav"
	pickle.dump(linearSVM, open(filename, "wb"))

	print("DONE")

#########################################################
# Gets the accuracy score for both train and test sets	#
# labels.txt - target labels of train set				#
# pred.txt - target labels of test set 					#
#########################################################
def test():
	labelsMatrixTrain = np.zeros(852)
	labelsMatrixTest = np.zeros(567)

	# open and load and read the target labels for the train set
	with open("labels.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrixTrain[index] = int(line[0])

	# open and load and read the target labels for the test set
	with open("pred.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrixTest[index] = int(line[0])

	# read and store the features vector of the train set
	datasetTrain = np.genfromtxt("datasetTrain.csv", delimiter=",")

	# read and store the features vetor of the test set
	datasetTest = np.genfromtxt("datasetTest.csv", delimiter=",")

	# load the pretrained model 
	loadModel = pickle.load(open("trainModelOK.sav", "rb"))

	# store the predicted labels for both train and test set
	predTrain = loadModel.predict(datasetTrain)
	predTest = loadModel.predict(datasetTest)

	# print their accuracy_scores 
	print(accuracy_score(labelsMatrixTrain, predTrain))
	print(accuracy_score(labelsMatrixTest, predTest))
	#print(loadModel.score(datasetTrain, labelsMatrixTrain))
	#print(loadModel.score(datasetTest, labelsMatrixTest))
	print()

	print("DONE")

#########################################################
# Predicts a certain image 								#
#########################################################
def predict():
	# load the HOG descriptor
	hgd = cv2.HOGDescriptor()

	# load the pretrained model
	loadModel = pickle.load(open("trainModelOK.sav", "rb"))

	featuresVector = []

	# image to be predicted
	img = cv2.imread("newMean/56.jpg")

	# compute the HOG values
	val = hgd.compute(img)
	featuresVector.append(val)
	featuresVector = np.asarray(featuresVector)
	featuresVector = featuresVector.reshape(len(featuresVector), -1)

	#print(featuresVector.shape)

	# get the model's predicteion
	prediction = loadModel.predict(featuresVector)

	# convert the prediction of the model to string
	# 1: Filipino
	# 2: Thai
	# 3: Indonesian
	#print(int(prediction))
	if int(prediction) == 1:
		print("\nFilipino")
	elif int(prediction) == 2:
		print("\nThai")
	elif int(prediction) == 3:
		print("\nIndonesian")

	#print("DONE")

#################
# main function #
#################
def main():
	print("Choose")
	print("(1) Preprocess")
	print("(2) Feature Vector")
	print("(3) Train Model")
	print("(4) Accuracy Score")
	print("(5) Predict")
	
	choice = int(input())

	if choice == 1:
		print("\nPreprocessing...")
		preprocess()
	elif choice == 2:
		print("\nCreating feature vector...")
		featureVector()
	elif choice == 3:
		print("\nTraining...")
		train()
	elif choice == 4:
		print("\nObtaining accuracy score...")
		test()
	elif choice == 5:
		print("\nPredicting...")
		predict()
	else:
		print("\nERROR")

if __name__ == "__main__":
	main()