# Implements the Parallel Pattern Training method described here: https://www.cs.swarthmore.edu/~newhall/papers/pdcn08.pdf

# PPT ALGORITHM
# -1) Full copy of data at each node. 
# 0) Create a network at the master node and broadcast it to all other nodes. 
# 1) Train each network on a random subset of the data (if there are n nodes, then maybe N/n, 
	# where N is the number of training samples?)
# 2) Then write the network to xml.
# 3) Transmit all xml to master node
# 4) Master node combines them somehow into a composite network
# 5) Find error measure for composite network
# 6a) If error below threshold:
# 	then return network.
# 6b) Else:
	# Write composite network to xml
	# Broadcast this xml to all nodes
	# Construct new network at each node
	# Repeat back to 1)


########################################################################################################

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from mpi4py import MPI

from pybrain.tools.xml import NetworkWriter
from pybrain.tools.xml import NetworkReader

import numpy

import json
import os
import time
import random
import math

import xml.etree.ElementTree as ET

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numProc = comm.Get_size()

# A copy of this class will be instantiated on each node.
class PPTInstance() :
	def __init__(self):
		# Parameters for data input. Each node has copy of data and symbolslist.csv
		self.dataFolderPath = "../../data/training-data"
		self.symbolsListFile = "../../data/scripts/symbolslist.csv"

		# Import symbols list
		self.syms = open(self.symbolsListFile, 'r')
		self.symlist = self.syms.read().split(',')
		self.mySyms = []

		# File that network XML is saved to
		self.nnXMLFile = 'saved-network-' + str(rank)

		# Dataset for training
		self.dataSet = SupervisedDataSet(10, 1)
		self.dataSubset = SupervisedDataSet(10, 1)
		self.numTrainingSamples = 0

		# Local copy of network
		self.network = None

		# Local trainer
		self.trainer = None

		# Record training errors here
		self.errorFile = 'training-errors'
		self.totalError = numpy.zeros(1)
		self.thresholdError = 0.0000001
		self.errors = []

		# XML strings of all networks
		self.nnStringArray = []

		# Matrix of params of different NNs
		self.paramMatrix = [[],[],[],[]]
		self.avgParamsMatrix = [[],[],[],[]]

		# Continue training?
		self.keepTraining = True

	# Initialize network
	def initializeNN(self) :
		self.network = buildNetwork(10, 10, 1)

	# Build NN from xml file
	def nnFromXML(self) :
		self.network = None
		self.network = NetworkReader.readFrom(self.nnXMLFile)

	# Generate NN and broacast to other nodes
	def broadcastMasterNN(self) :
		if rank == 0 :
			NetworkWriter.writeToFile(self.network,self.nnXMLFile)
			nnFile = open(self.nnXMLFile, 'r')
			nnString = nnFile.read()
			nnFile.close()

		else :
			nnString = None

		# Collect master network
		nnString = comm.bcast(nnString, root=0)

		# If worker node, write master network and use it to generate a NN
		if rank != 0 :
			nnFile = open(self.nnXMLFile, 'w')
			nnFile.write(nnString)
			nnFile.close()

			self.nnFromXML()


	# Create a pybrain dataset using a randomly subset of the symbols' data.
	def loadData(self):
		# Find subset of symbols that this node is responsible for.
		# Add training data from each symbol
		for symbol in self.symlist :
			symbol = "".join([symbol,"_"])
			dataPath = os.path.join(self.dataFolderPath, symbol)

			dataFile = open(dataPath, 'r')
			data = dataFile.read()

			symboljson = json.loads(data)

			for datetime in symboljson.keys() :
				inputData = symboljson[datetime]["Input"]
				outputData = symboljson[datetime]["Output"]

				inputVector = []
				for inpVal in inputData.keys() :
					inputVector.append(inputData[inpVal])

				outputVector = outputData["return"]

				self.dataSet.appendLinked(inputVector, outputVector)
				self.numTrainingSamples += 1
		# print self.dataSet.randomBatches("Input", 2)

	# Creates data subset for training. Proportation is the fraction of the 
	# 	data that this node is responsible for.
	def subsetData(self, proportion) :
		indicies = numpy.random.permutation(self.numTrainingSamples)
		separator = int(self.numTrainingSamples * proportion)
		myIndicies = indicies[:separator]
		self.dataSubset = SupervisedDataSet(inp=self.dataSet['input'][myIndicies].copy(),
                                   target=self.dataSet['target'][myIndicies].copy())

	# Train network using local data and record error.
	def trainOnLocal(self) :
		# Train
		self.trainer = BackpropTrainer(self.network, self.dataSubset, learningrate=0.1)
		error = numpy.zeros(1)
		error[0] = self.trainer.train()
		totalError = numpy.zeros(1)

		comm.Allreduce(error, totalError, op=MPI.SUM)

		totalError = totalError / float(numProc)
		self.totalError = totalError
		self.errors.insert(0,self.totalError)

		# Record error
		if rank == 0:
			errFile = open(self.errorFile, 'a')
			errFile.write(str(totalError[0]))
			errFile.write('\n')
			errFile.close()

		# Write new NN weights
		NetworkWriter.writeToFile(self.network, self.nnXMLFile)

	# After training for one epoch, worker nodes send data back to master node
	def gatherNN(self) :
		# Get string of NN XML
		nnFile = open(self.nnXMLFile, 'r')
		nnString = nnFile.read()
		nnFile.close()

		# Gather all nnString and store their parameters in self.paramMatrix
		self.nnStringArray = comm.gather(nnString, root=0)
		if rank == 0 :
			for proc in range(0,numProc) :
				root = ET.fromstring(self.nnStringArray[proc])
				for index, conn in enumerate(root.iter('Parameters')) :
					listText = conn.text.strip('[').strip(']').split(',')
					listText = map(lambda x: float(x), listText)
					self.paramMatrix[index].append(listText)

	# Averages the weights of each network
	def findAvgWeights(self) :
		if rank == 0 :
			self.avgParamsMatrix = [[],[],[],[]]
			for row in range(0,len(self.paramMatrix)) :
				avgParams = []
				numParams = len(self.paramMatrix[row][0])
				for i in range(0, numParams) :
					avgVal = 0.0
					for nn in range(0, numProc) :
						avgVal += self.paramMatrix[row][nn][i]
					avgVal = avgVal / float(numProc)
					self.avgParamsMatrix[row].append(avgVal)

	# Write avg params to xml for use in new NN.
	def avgWeightsNN(self) :
		if rank == 0 :
			root = ET.fromstring(open(self.nnXMLFile,'r').read())
			for index, conn in enumerate(root.iter('Parameters')) :
				conn.text = str(self.avgParamsMatrix[index])
			tree = ET.ElementTree(element=root)
			tree.write(self.nnXMLFile, xml_declaration=True)

# Run it
if __name__ == "__main__":
	startTime = time.time()

	ppt = PPTInstance()
	ppt.initializeNN()

	genstart = time.time()
	ppt.loadData()
	proportion = 1.0/float(numProc)
	# proportion = 0.01

	counter = 1

	while ppt.keepTraining :
		if rank == 0 :
			print "Epoch number : ", counter, "\n"
			counter += 1

		ppt.broadcastMasterNN() 

		substart = time.time()
		ppt.subsetData(proportion)
		ppt.trainOnLocal()

		comm.Barrier()

		# If we fail to improve the network by more than the threshold change, then stop training.
		if counter >= 3 and (math.fabs(ppt.errors[1]-ppt.errors[0]) <= ppt.thresholdError):
			if rank == 0 :
				ppt.keepTraining = False

		ppt.keepTraining = comm.bcast(ppt.keepTraining, root=0)

		if not ppt.keepTraining:
			break

		ppt.gatherNN()
		ppt.findAvgWeights()
		ppt.avgWeightsNN()

		if rank == 0 :
			ppt.nnFromXML()

	# comm.MPI_finalize()
	endTime = time.time()
	if rank == 0 :		
		print "\n Neural network training completed in : ", str(endTime - startTime) , " seconds."
	



























