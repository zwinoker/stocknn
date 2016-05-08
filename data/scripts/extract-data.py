# Creates training data from the full data set stored in the MySQL database. 
# 	Data is indexed by date-time and outputed as a JSON file.

import mysql.connector
import json
import collections
import os

###### Parameters #####

# Market close and open times
marketCloseTime = "16:00:00"
marketOpenTime = "09:30:00"

# Start/end dates
startDate = "2016-03-28"
endDate = "2016-04-01"

# For MySQL queries
startDateTime = " ".join([startDate, marketOpenTime])
endDateTime = " ".join([endDate, marketCloseTime])

# Training data JSONS written to this folder
dataFolderPath = "/Users/zacharywinoker/Desktop/gradSchool/spr2016/classes/parallelProgramming/finalProject/data/training-data"

# List of stock symbols we have data on
symbolsListFile = "symbolslist.csv"

##### End Parameters #####

# Import symbols list
syms = open(symbolsListFile, 'r')
symlist = syms.read().split(',')

# Get data from MySQL database
conn = mysql.connector.connect(user='root', password='sqlpass', host='localhost',database='stockmarket')
cursor = conn.cursor()

for symbol in symlist :
	print symbol
	symbol = "".join([symbol, "_"])
	query = "".join(["select * from ", symbol, " where Date <= \'", endDateTime, "\' and Date >= \'", startDateTime,"\'"])
	cursor.execute(query)

	# Convert to JSON. We take advantage of the fact that database rows are sorted by date-time. 
	jsonElt = collections.OrderedDict()
	row_min_1 = []
	row_min_2 = []

	# Process row data.
	for row in cursor:
		# If in first row, just set row_min_1 to current row and move on.
		if not row_min_1 and not row_min_2 :
			row_min_1 = row

		# If in second row:
		elif not row_min_2 :
			row_min_2 = row_min_1
			row_min_1 = row

		# If at market close, clear previous rows and start over. Since network is only for intra-day trading,
		# 	we don't want to train on inter-day data.
		elif (row[0].split(" "))[1] == marketCloseTime :
			row_min_1 = []
			row_min_2 = []

		# Else generate training datum for this date-time
		else :
			trainingData = collections.OrderedDict()
			inputData = collections.OrderedDict()
			outputData = collections.OrderedDict()

			outputData['return'] = (row[1] - row_min_1[1])/row_min_1[1]

			inputData['close_i-1'] = row_min_1[1]
			inputData['open_i-1'] = row_min_1[4]
			inputData['high_i-1'] = row_min_1[2]
			inputData['low_i-1'] = row_min_1[3]
			inputData['volume_i-1'] = row_min_1[5]

			inputData['close_i-2'] = row_min_1[1]
			inputData['open_i-2'] = row_min_1[4]
			inputData['high_i-2'] = row_min_1[2]
			inputData['low_i-2'] = row_min_1[3]
			inputData['volume_i-2'] = row_min_1[5]

			trainingData['Input'] = inputData
			trainingData['Output'] = outputData

			row_min_2 = row_min_1
			row_min_1 = row

			jsonElt[row[0]] = trainingData

	# Write JSON to file
	fulljson = json.dumps(jsonElt)
	writePath = os.path.join(dataFolderPath, symbol)
	writefile = open(writePath, 'w')
	writefile.write(fulljson)
	writefile.close()

cursor.close()
conn.close()





