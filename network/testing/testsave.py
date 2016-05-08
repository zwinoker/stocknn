# Test saving NN to XML and reading NN from XML.

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader

import json
import os
import time

dataFolderPath = "/Users/zacharywinoker/Desktop/gradSchool/spr2016/classes/parallelProgramming/finalProject/data/training-data"



# List of stock symbols we have data on
symbolsListFile = "/Users/zacharywinoker/Desktop/gradSchool/spr2016/classes/parallelProgramming/finalProject/data/scripts/symbolslist.csv"

# Import symbols list
syms = open(symbolsListFile, 'r')
symlist = syms.read().split(',')

symbol = "AAPL_"
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

error = trainer.train()

NetworkWriter.writeToFile(net, "saved-net")

nn = NetworkReader.readFrom("saved-net")


# True if both networks are the same
print nn.activate([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]) == net.activate([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])
















