import cv2
import sys
import os
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def meansub():
	if not "newMean" in os.listdir("."):
		os.mkdir("newMean")

	imageList  = []
	imageDir = "newGray/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)
		mean = np.mean(img)
		newIMG = img - mean
		strIMG = imagePath.split("/")
		outname = "newMean/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)
	
	print("DONE")

def grayscale():
	if not "newGray" in os.listdir("."):
		os.mkdir("newGray")

	imageList  = []
	imageDir = "newFlip/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)
		newIMG = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		strIMG = imagePath.split("/")
		outname = "newGray/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)
	
	print("DONE")

def augment():
	if not "newFlip" in os.listdir("."):
		os.mkdir("newFlip")

	imageList  = []
	imageDir = "newPre/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)
		newIMG = np.fliplr(img)
		strIMG = imagePath.split("/")
		outname = "newFlip/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)

	print("DONE")

def resize():
	if not "newPre" in os.listdir("."):
		os.mkdir("newPre")

	imageList  = []
	imageDir = "Extracted/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for imagePath in imageList:
		img = cv2.imread(imagePath)
		newIMG = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
		strIMG = imagePath.split("/")
		outname = "newPre/{}".format(strIMG[1])
		cv2.imwrite(outname, newIMG)

	print("DONE")

def crop():
	if not "Extracted" in os.listdir("."):
		os.mkdir("Extracted")

	imageList = []
	imageDir = "raw/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	for image_path in imageList:
		cascade="Face_cascade.xml"
		face_cascade=cv2.CascadeClassifier(cascade)

		image=cv2.imread(image_path)
		image_grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		faces = face_cascade.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(25,25),flags=0)

		for x,y,w,h in faces:
		    sub_img=image[y-10:y+h+10,x-10:x+w+10]
		    os.chdir("Extracted")
		    cv2.imwrite(str(y)+".jpg",sub_img)
		    os.chdir("../")
		    cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)

	#cv2.imshow("Faces Found",image)
	#cv2.waitKey(0)
	print("DONE")

def preprocess():
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
		print("\nERROR")

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

def trainFeature():
	hgd = cv2.HOGDescriptor()

	imageList  = []
	imageDir = "newMean/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	featuresVector = np.zeros((len(imageList), 34020))

	for imageIndex, imagePath in enumerate(imageList):
		img = cv2.imread(imagePath)
		val = hgd.compute(img)
		for i in range(34020):
			featuresVector[imageIndex][i] = val[i]

	csvFile = open("datasetTrain.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for i in range(len(imageList)):
		writer.writerow(featuresVector[i])

	print("DONE")

def testFeature():
	hgd = cv2.HOGDescriptor()

	imageList  = []
	imageDir = "newTest/"

	for filename in os.listdir(imageDir):
		imageList.append(os.path.join(imageDir, filename))
	featuresVector = np.zeros((len(imageList), 34020))

	for imageIndex, imagePath in enumerate(imageList):
		img = cv2.imread(imagePath)
		val = hgd.compute(img)
		for i in range(34020):
			featuresVector[imageIndex][i] = val[i]

	csvFile = open("datasetTest.csv", "w", newline='')
	writer = csv.writer(csvFile)
	for i in range(len(imageList)):
		writer.writerow(featuresVector[i])

	print("DONE")

def train():
	labelsMatrix = np.zeros(852)
	with open("labels.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrix[index] = int(line[0])
	print(labelsMatrix.shape)
	datasetTrain = np.genfromtxt("datasetTrain.csv", delimiter=",")
	print(datasetTrain.shape)

	linearSVM = LinearSVC(C=0.01, max_iter=10000, loss='hinge', tol=0.00001, intercept_scaling=100)
	linearSVM.fit(datasetTrain, labelsMatrix)
	filename = "trainModelOK.sav"
	pickle.dump(linearSVM, open(filename, "wb"))

	print("DONE")

def test():
	labelsMatrixTrain = np.zeros(852)
	labelsMatrixTest = np.zeros(567)
	with open("labels.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrixTrain[index] = int(line[0])
	with open("pred.txt", "r") as labels:
		for index, line in enumerate(labels):
			line = line.split()
			labelsMatrixTest[index] = int(line[0])

	datasetTrain = np.genfromtxt("datasetTrain.csv", delimiter=",")
	datasetTest = np.genfromtxt("datasetTest.csv", delimiter=",")

	loadModel = pickle.load(open("trainModelOK.sav", "rb"))
	print(loadModel.score(datasetTrain, labelsMatrixTrain))
	print(loadModel.score(datasetTest, labelsMatrixTest))

	print("DONE")

def predict():
	print("HELLO")

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
	elif choice == 5:
		print("\nPredicting...")
	else:
		print("\nERROR")

if __name__ == "__main__":
	main()