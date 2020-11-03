from src.AAUfilename import * 

def joinStr(line):
	last = None
	res = []
	strDelCount = None
	for i in range(len(line)):
		if last != None:
			if line[i] != "" and line[i][-1] == "\"":
				strDelCount += line[i].count("\"")
				if not strDelCount % 2:
					res.append("".join(line[last : i+1]))
					last = None
			else:
				line[i] = line[i]+","
				
		elif line[i] != "" and line[i][0] == "\"" and line[i].count("\"")%2: # bool(0) = False, bool(1) = True
			last = i
			line[i] = line[i]+","
			strDelCount = 1
		
		else:
			res.append(line[i])
	
	if last != None:
		return []
	return res


def nbrCol(line):
	line = line.split(sep = ",")
	line = joinStr(line)
	#print(line)
	#input()
	return len(line)

def hasNbrCol(n, line):
	return n == nbrCol(line)

def cleanData(currentFilename, newFilename, rejectFilename, nbrCol, bufferSize):
	currentFilename = getAAUfilename(currentFilename)
	newFilename = getAAUfilename(newFilename)
	rejectFilename = getAAUfilename(rejectFilename)
	
	buf = []
	i = 0
	rejectF = open(rejectFilename, "w")
	rejectF.close()
	newF = open(newFilename, "w")
	newF.close()
	for line in open(currentFilename, "r"):
		if hasNbrCol(8, line):
			buf.append(line)
			
		else:
			with open(rejectFilename, "a") as rejectF:
				rejectF.write(line + "\n")
		
		if len(buf) >= bufferSize:
			with open(newFilename, "a") as newF:
				newF.write("".join(buf))
			buf = []
		
		i+=1
		print("\rline "+ str(i) + " done", end = "")
	print("")
		
	if len(buf) != 0:
		with open(newFilename, "a") as newF:
			newF.write("".join(buf))

def findDif(bigFilename, smallFilename):
	bigFilename = getAAUfilename(bigFilename)
	smallFilename = getAAUfilename(smallFilename)
	i = 0
	
	with open(bigFilename, "r") as bf:
		for sfLine in open(smallFilename, "r"):
			print("current line: ", i, end="\r")
			bfLine = bf.readline()
			
			if bfLine != sfLine:
				print(bfLine)
			
			while bfLine != sfLine:
				bfLine = bf.readline()
			i+=1
				
	

if __name__ == "__main__":
	cleanData("data4project_global.csv", "clean_test/data4project_global_cleanedV2.csv", "clean_test/data4project_global_rejectV2.csv", 8, 100000)
	#findDif("data4project_global.csv", "clean_test/data4project_global_cleaned.csv")



















































