import pandas as pd
import tensorflow as tf

def topKRatings(k, model, usersId, itemsId, mtype=None):
	topK = {}
	isTwoTower = mtype == "two tower"
	count = 1
	for u in usersId:
		print("\rComputing top"+str(k)+": "+str(count)+"/"+str(len(usersId)), end="")
		ratings = []
		if isTwoTower:
			candidate = pd.DataFrame({"CUSTOMER_ID":[u for i in range(len(itemsId))], "MATERIAL":[i for i in itemsId]})
			ratings = list(model.predict(tf.data.Dataset.from_tensor_slices(dict(candidate)).batch(len(itemsId))))
			ratings = [(ratings[i], itemsId[i]) for i in range(len(itemsId))]
		else:
			for i in itemsId:
				ratings.append((model.predict([pd.array([u]), pd.array([i])])), i)
		ratings.sort(reverse=True, key = (lambda x: x[0]))
		topK[u] = ratings[:k]
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
	for u in predictions:
		for r, i in predictions[u]:
			if (u,i) in real:
				tp += 1
			else:
				fp += 1
	fn = len(real) - tp
	tn = total - tp - fp - fn
	
	return{"tp":tp, "tn":tn, "fp":fp, "fn":fn, "precision":tp/(tp+fp), "recall": tp/(tp+fn)}
