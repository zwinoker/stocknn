# Playing around with pybrain to see if it's what I need for neural networks

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

import json
import os
import time

dataFolderPath = "/Users/zacharywinoker/Desktop/gradSchool/spr2016/classes/parallelProgramming/finalProject/data/training-data"



# List of stock symbols we have data on
symbolsListFile = "/Users/zacharywinoker/Desktop/gradSchool/spr2016/classes/parallelProgramming/finalProject/data/scripts/symbolslist.csv"

# Import symbols list
syms = open(symbolsListFile, 'r')
symlist = syms.read().split(',')

# Add training data from each symbol
for symbol in symlist :
	print symbol
	symbol = "".join([symbol,"_"])
	dataPath = os.path.join(dataFolderPath, symbol)

	dataFile = open(dataPath, 'r')
	data = dataFile.read()

	symboljson = json.loads(data)

	dataSet = SupervisedDataSet(10, 1)

	for datetime in symboljson.keys() :
		inputData = symboljson[datetime]["Input"]
		outputData = symboljson[datetime]["Output"]

		inputVector = []
		for inpVal in inputData.keys() :
			inputVector.append(inputData[inpVal])

		outputVector = outputData["close_i"]

		dataSet.appendLinked(inputVector, outputVector)

print dataSet

net = buildNetwork(10, 10, 1)
trainer = BackpropTrainer(net, dataSet)

startTime = time.time()
error = trainer.trainUntilConvergence(maxEpochs=100)
endTime = time.time()

print "Training time is : ", endTime - startTime , " seconds."

print error


