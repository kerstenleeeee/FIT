import cv2
import sys
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pickle
import csv

#train: 852
#test: 567

labelsMatrix = np.zeros(852)
with open("labels.txt", "r") as labels:
	for index, line in enumerate(labels):
		line = line.split()
		labelsMatrix[index] = int(line[0])
print(labelsMatrix.shape)
#print(labelsMatrix)
datasetTrain = np.genfromtxt("datasetTrain.csv", delimiter=",")
print(datasetTrain.shape)
#kNN = KNeighborsClassifier()
#kNN.fit(datasetTrain, labelsMatrix)
aNN = MLPClassifier(hidden_layer_sizes=(17010,17010))
aNN.fit(datasetTrain, labelsMatrix)
filename = "trainModel.sav"
pickle.dump(aNN, open(filename, "wb"))

print("DONE")

'''loadModel = pickle.load(open("trainModel.sav", "rb"))
labelsMatrix = np.zeros(567)
with open("pred.txt", "r") as labels:
	for index, line in enumerate(labels):
		line = line.split()
		labelsMatrix[index] = int(line[0])
print(labelsMatrix.shape)
datasetTest = np.genfromtxt("datasetTest.csv", delimiter=",")
print(datasetTest.shape)
loadModel.predict(datasetTest)
print(loadModel.predict(datasetTest))
print(loadModel.score(datasetTest, labelsMatrix))
print("DONE")'''