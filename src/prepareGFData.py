import pandas as pd

COLUMNS_NAME = ["DOC_NUMBER_TEXT","CALDAY","MATERIAL","MATERIAL_TEXT","MATL_GROUP","QUANTITY","is_accessory","CUSTOMER_ID"]
REPO_PATH = "/run/user/1000/gvfs/smb-share:server=cs.aau.dk,share=fileshares/IT703e20"

def getUniqueMC(filename):
	chunks = pd.read_csv(REPO_PATH+"/"+filename, sep=',', names = COLUMNS_NAME, chunksize = 50000)
	res = set()
	customers_id = set()
	materials_id = set()
	i = 0
	for chunck in chunks:
		for l in range(len(chunck["CUSTOMER_ID"])):
			if i != 0:
				lineC = chunck["CUSTOMER_ID"][i]
				lineM = chunck["MATERIAL"][i]
				customers_id.add(lineC)
				materials_id.add(lineM)
				res.add((lineC, lineM))
			i += 1
		print("reading until:",i,"done.")
	
	return {"res":res, "customers_id":customers_id, "materials_id":materials_id}

def writeToFile(data, customer_id, material_id, filename):
	print("writing data.")
	with open(REPO_PATH+"/"+filename,"w") as f:
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


MC = getUniqueMC("data4project_global.csv")
writeToFile(MC["res"], MC["customers_id"], MC["materials_id"], "binary_MC_global_only1.csv")
	
	
	

