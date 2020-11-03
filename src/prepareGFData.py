import pandas as pd
from src.AAUfilename import *

COLUMNS_NAME = ["DOC_NUMBER_TEXT","CALDAY","MATERIAL","MATERIAL_TEXT","MATL_GROUP","QUANTITY","is_accessory","CUSTOMER_ID"]


def getUniqueMC(filename):
	chunks = pd.read_csv(getAAUfilename(filename), chunksize = 50000)
	res = set()
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
				customers_id.add(lineC)
				materials_id.add(lineM)
				res.add((lineC, lineM))
			i += 1
		print("reading until:",i,"done.", end = "\r")
	
	return {"res":res, "customers_id":customers_id, "materials_id":materials_id}

def writeToFile(data, customer_id, material_id, filename):
	print("writing data.")
	with open(getAAUfilename(filename),"w") as f:
		#f.write("CUSTOMER_ID,MATERIAL,WAS_BOUGHT\n")
		f.write("CUSTOMER_ID,MATERIAL\n")
		"""for customer in customer_id:
			for material in material_id:
				wasBought = 0.0
				if (customer, material) in data:
					wasBought = 1.0
				
				f.write(str(customer)+","+str(material)+","+str(wasBought)+"\n")"""
		
		for pair in data:
			f.write(str(pair[0])+","+str(pair[1])+"\n")


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

#MC = getUniqueMC("clean_test/data4project_global_cleaned.csv")
#writeToFile(MC["res"], MC["customers_id"], MC["materials_id"], "clean_test/binary_MC_global_only1.csv")
seeLines(1734550,1734560, "clean_test/data4project_global_cleaned.csv")
	
	
	














































