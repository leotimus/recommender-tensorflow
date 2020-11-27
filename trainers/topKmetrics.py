  
import pandas as pd
import tensorflow as tf
from multiprocessing import Pool

# To do parallelism using multiprocessing, lambda functions aren't allowed
# and I can't find an easy way to accept multiple arguments, so the other 
# two just go in global variables.  ~ asg h
__currentModel = None
__currentUserId = None
def __predictForCurrentUser(i):
	return (__currentModel.predict([pd.array([__currentUserId]), pd.array([i])]), i)

def topKRatings(k, model, usersId, itemsId, mtype=None):
	global __currentModel
	global __currentUserId
	topK = []
	isTwoTower = mtype == "two tower"
	count = 1
	itemslst = [i for i in itemsId]
	__currentModel = model
	for u in usersId:
		print("\rComputing top"+str(k)+": "+str(count)+"/"+str(len(usersId)), end="")
		ratings = []
		if isTwoTower:
			candidate = pd.DataFrame({"CUSTOMER_ID":[u for i in range(len(itemsId))], "MATERIAL":itemslst})
			ratings = list(model.predict(tf.data.Dataset.from_tensor_slices(dict(candidate)).batch(len(itemsId))))
			ratings = [(ratings[i], itemsId[i]) for i in range(len(itemsId))]
		else:
			__currentUserId = u
			with Pool() as p:
				ratings = p.map(__predictForCurrentUser, itemsId)
		ratings.sort(reverse=True, key=(lambda x: x[0]))
		topK.append((u, ratings[:k]))
		count += 1
	print("")
	return topK

def topKMetrics(predictions, positives, usersId, itemsId):
	nbrUser = len(usersId)
	nbrItem = len(itemsId)
	total = nbrUser * nbrItem
	
	real = set(positives)
	
	tp = 0
	fp = 0
	for u, topk in predictions:
		for r, i in topk:
			if (u,i) in real:
				tp += 1
			else:
				fp += 1
	fn = len(real) - tp
	tn = total - tp - fp - fn
	
	return{"tp":tp, "tn":tn, "fp":fp, "fn":fn, "precision":tp/(tp+fp), "recall": tp/(tp+fn)}