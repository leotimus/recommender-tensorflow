import pandas as pd
from src.AAUfilename import *
import random as rd
import numpy as np

COLUMNS_NAME = ["DOC_NUMBER_TEXT","CALDAY","MATERIAL","MATERIAL_TEXT","MATL_GROUP","QUANTITY","is_accessory","CUSTOMER_ID"]


def getUniqueMC(filename):
	chunks = pd.read_csv(getAAUfilename(filename), chunksize = 50000)
	res = {}
	customers_id = set()
	materials_id = set()
	i = 0
	for chunck in chunks:
		for l in range(len(chunck["CUSTOMER_ID"])):
			if i != 0:
				lineC = chunck["CUSTOMER_ID"][i]
				"""if type(lineC) != str:
					print("line", i,"not str:", lineC)
					print(chunck["DOC_NUMBER_TEXT"][i],"|",chunck["CALDAY"][i],"|",chunck["MATERIAL"][i],"|",chunck["MATERIAL_TEXT"][i],"|",chunck["MATL_GROUP"][i],"|",chunck["QUANTITY"][i],"|",chunck["is_accessory"][i],"|",chunck["CUSTOMER_ID"][i])
				elif not lineC.isdigit():
					print("line", i,"not digit:", lineC)
					print(chunck["DOC_NUMBER_TEXT"][i],"|",chunck["CALDAY"][i],"|",chunck["MATERIAL"][i],"|",chunck["MATERIAL_TEXT"][i],"|",chunck["MATL_GROUP"][i],"|",chunck["QUANTITY"][i],"|",chunck["is_accessory"][i],"|",chunck["CUSTOMER_ID"][i])"""
				lineM = chunck["MATERIAL"][i]
				lineQ = chunck["QUANTITY"][i]
				customers_id.add(lineC)
				materials_id.add(lineM)
				if (lineC, lineM) in res:
					res[(lineC, lineM)] += lineQ
				else:
					res[(lineC, lineM)] = lineQ
			i += 1
		print("reading until:",i,"done.", end = "\r")
	
	print("")
	return {"res":res, "customers_id":customers_id, "materials_id":materials_id}

def writeToFile(data, customer_id, material_id, filename):
	print("writing data.")
	with open(getAAUfilename(filename),"w") as f:
		#f.write("CUSTOMER_ID,MATERIAL,WAS_BOUGHT\n")
		f.write("CUSTOMER_ID,MATERIAL,QUANTITY\n")
		"""for customer in customer_id:
			for material in material_id:
				wasBought = 0.0
				if (customer, material) in data:
					wasBought = 1.0
				
				f.write(str(customer)+","+str(material)+","+str(wasBought)+"\n")"""
		
		for pair in data:
			f.write(str(pair[0])+","+str(pair[1])+","+str(data[pair])+"\n")


def seeLines(lineNbrStart, lineNbrEnd, filename):
	i = 0
	for line in open(getAAUfilename(filename), "r"):
		i += 1
		if i%10000 == 0:
			print("curent line:", i, end = "\r")
		if i >= lineNbrStart:
			print("line",i,":",line)
			
		if i == lineNbrEnd:
			break

def writeBufIfFull(filename, buf, bufSize):
	if len(buf) >= bufSize:
		with open(filename, "a") as f:
			f.write("".join(buf))
		return True
	return False

def generateValues(materialId, userId):
	np.random.shuffle(materialId)
	res = []
	for i in range(len(userId)):
		res.append((materialId[i], userId[i]))
	return res

def populateDataset(filename, newFilename, addProb, explicitVal, implicitVal, bufSize, it=0):
	filename = getAAUfilename(filename)
	newFilename = getAAUfilename(newFilename)
	
	data = pd.read_csv(filename, dtype = {"MATERIAL":str, "CUSTOMER_ID": str})
	
	materialId = pd.unique(data["MATERIAL"])
	nbrMaterial = len(materialId)
	usersId = pd.unique(data["CUSTOMER_ID"])
	nbrUser = len(usersId)
	
	#add the expected output for the neural network
	#data["is_real"] = [float(explicitVal) for i in range(len(data["MATERIAL"]))]
	
	with open(newFilename, "w") as f:
		f.write("CUSTOMER_ID, MATERIAL, is_real\n")
	
	buf = []
	#create a set containing the existing pairs
	nbrLine = len(data["MATERIAL"])
	explicit = set()
		
	for i in range(nbrLine):
		print("\rexplicit: " + str(i) + "/" + str(nbrLine), end = "")
		lineM = data["MATERIAL"][i]
		lineC = data["CUSTOMER_ID"][i]
		explicit.add((lineM, lineC))
		buf.append(lineC + "," + lineM + "," + str(explicitVal)+ "\n")
		if writeBufIfFull(newFilename, buf, bufSize):
			buf = []
	print("")
	nbrDone=0
	userDone = set()
	userZeros = {}
	for u in usersId:
		userZeros[u] = 0
		
	while nbrDone < nbrUser:
		print("\rusers ready: "+str(nbrDone)+"/"+str(nbrUser), end = "")
		for pair in generateValues(materialId, usersId):
			if userZeros[pair[1]] != it and pair[1] not in userDone:
				if pair not in explicit:
					explicit.add(pair)
					userZeros[pair[1]] += 1
					buf.append(pair[1] + "," + pair[0] + "," + str(implicitVal)+ "\n")
			else:
				nbrDone += 1
				userDone.add(pair[1])
			
				if writeBufIfFull(newFilename, buf, bufSize):
					buf = []
			
	print("")
	writeBufIfFull(newFilename, buf, 1)
	
	#populate with implicit values
	"""i = 0
	bigNumber = 1000000
	for user in usersId:
		
		for material in materialId:
			print("\rimplicit: "+str(i)+"/"+str(nbrMaterial*nbrUser), end = "")
			if ((user, material) not in explicit) and rd.randint(0,bigNumber) <= bigNumber*addProb:
				buf.append(user + "," + material + "," + str(implicitVal)+ "\n")
			
			if writeBufIfFull(newFilename, buf, bufSize):
				buf = []
			i += 1
	
	writeBufIfFull(newFilename, buf, 1)"""
	
	
	
	
				
if __name__ == "__main__":
	#MC = getUniqueMC("CleanDatasets/no0s-unique-noBlanks-noNegatives.csv")
	#writeToFile(MC["res"], MC["customers_id"], MC["materials_id"], "CleanDatasets/MCQ.csv")
	#seeLines(2464070,2464090, "clean_test/data4project_global_cleaned.csv")
	populateDataset("CleanDatasets/no_0s/binary_MC_global_no0s.csv", "CleanDatasets/no_0s/binary_MC_no0s_populated400_exact.csv", 0.001, 1.0, 0.0, 500000,400)
	
	
	














































