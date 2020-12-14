import numpy as np
import pandas as pd
import keras
import keras.utils
import tensorflow as tf
import time
#import os
# import neptune
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.benchmarkLogger import benchThread
#from getpass import getpass
import smbclient as smbc
import trainers.topKmetrics as trainerTop
#from trainers.topKmetrics import topKMetrics


start_time = time.time()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print ('Loading dataset..')

smbc.ClientConfig(username='jpolas20@student.aau.dk', password='NMvcbchacsnL2022')

with smbc.open_file((r"\\cs.aau.dk\Fileshares\IT703e20\(NEW)CleanDatasets\NCF\2m(OG)\train.csv"), mode="r") as f:
     train = pd.read_csv(f, header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])

with smbc.open_file((r"\\cs.aau.dk\Fileshares\IT703e20\(NEW)CleanDatasets\NCF\2m(OG)\test.csv"), mode="r") as f:
     test = pd.read_csv(f, header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])

with smbc.open_file((r"\\cs.aau.dk\Fileshares\IT703e20\(NEW)CleanDatasets\NCF\2m(OG)\positives2m.csv"), mode="r") as f:
     allPositives = pd.read_csv(f, header=0, names=['id','customer_id', 'normalized_customer_id', 'material', 'product_id'])

with smbc.open_file((r"\\cs.aau.dk\Fileshares\IT703e20\(NEW)CleanDatasets\NCF\2m(OG)\positives_5th_split.csv"), mode="r") as f:
     positives_5th_split = pd.read_csv(f, header=0, names=['id','customer_id', 'normalized_customer_id', 'material', 'product_id'])


print ('Dataset loaded')
#frames = (train, test)
mergeddata_datasets = train.append(test)

num_customers = len(mergeddata_datasets.normalized_customer_id.unique())
num_materials_unique = len(mergeddata_datasets.product_id.unique())
num_materials = mergeddata_datasets.product_id.max()

#shuffle here
train = shuffle(train)
train.drop(columns=['customer_id','material'])
test.drop(columns=['customer_id','material'])
allPositives.drop(columns=['customer_id','material'])
positives_5th_split.drop(columns=['customer_id','material'])

# neptune.init(
#     api_token="",
#     project_qualified_name=""
#  )

# neptune.set_project('jitka7532/CS-project')
# neptune.create_experiment(name='100k')


# class NeptuneMonitor(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         for metric_name, metric_value in logs.items():
#             neptune.log_metric(metric_name, metric_value)

#comp resource benchmarking
#bmThread = benchThread(1,1,'2m_benchmarks.csv') #create the thread
#bmThread.start()


#Build the model
print ("Building model")
latent_dim = 100

material_input = Input(shape=[1],name='material-input')
customer_input = Input(shape=[1], name='customer-input')

# MLP Embeddings
material_embedding_mlp = Embedding(num_materials + 1, latent_dim, name='material-embedding-mlp')(material_input)
material_vec_mlp = Flatten(name='flatten-material-mlp')(material_embedding_mlp)

customer_embedding_mlp = Embedding(num_customers + 1, latent_dim, name='customer-embedding-mlp')(customer_input)
customer_vec_mlp = Flatten(name='flatten-customer-mlp')(customer_embedding_mlp)

# MF Embeddings
material_embedding_mf = Embedding(num_materials + 1, latent_dim, name='material-embedding-mf')(material_input)
material_vec_mf = Flatten(name='flatten-material-mf')(material_embedding_mf)

customer_embedding_mf = Embedding(num_customers + 1, latent_dim, name='customer-embedding-mf')(customer_input)
customer_vec_mf = Flatten(name='flatten-customer-mf')(customer_embedding_mf)

PARAMS = {'epoch_nr': 20,
          'batch_size': 50000,
          'lr': 0.01,
          'momentum': 0.4,
          'unit_nr1': 100,
          'unit_nr1': 50,
          'dropout': 0.5}

# MLP layers
concat = tf.keras.layers.Concatenate(axis=-1)([material_vec_mlp, customer_vec_mlp])
concat_dropout = Dropout(0.2)(concat)
fc_1 = Dense(100, name='fc-1', activation='sigmoid')(concat_dropout)
fc_1_bn = BatchNormalization(name='batch-norm-1')(fc_1)
fc_1_dropout = Dropout(PARAMS['dropout'])(fc_1_bn)
fc_2 = Dense(50, name='fc-2', activation='sigmoid')(fc_1_dropout)
fc_2_bn = BatchNormalization(name='batch-norm-2')(fc_2)
fc_2_dropout = Dropout(PARAMS['dropout'])(fc_2_bn)

# Prediction from both layers
pred_mlp = Dense(10, name='pred-mlp', activation='sigmoid')(fc_2_dropout)
pred_mf = tf.keras.layers.Dot(axes=1)([material_vec_mf, customer_vec_mf])
combine_mlp_mf = tf.keras.layers.Concatenate(axis=-1)([pred_mf, pred_mlp])

# Final prediction
result = Dense(1, name='result', activation='sigmoid')(combine_mlp_mf)
optimizer = keras.optimizers.Adam(lr=PARAMS['lr'])
model = Model([customer_input, material_input], result)
model.compile(optimizer, loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.BinaryCrossentropy(), 'mse', 'mae', tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.BinaryAccuracy()])

print ('Training model')

#history = model.fit([train.normalized_customer_id, train.product_id], train.rating_type, epochs=PARAMS['epoch_nr'], batch_size=PARAMS['batch_size'], shuffle=True, verbose=2, callbacks=[NeptuneMonitor()])
history = model.fit([train.normalized_customer_id, train.product_id], train.rating_type, epochs=PARAMS['epoch_nr'], batch_size=PARAMS['batch_size'], shuffle=True, verbose=2)
#model.save('D:/ML/GrundfosData/NCF_savedModels/100k/with0s')

#pd.Series(history.history['loss']).plot(logy=True)
#plt.xlabel("Epoch")
#plt.ylabel("Train Error")
print ('Training model...Done')

print ('Predict model')
y_hat = np.round(model.predict([test.normalized_customer_id, test.product_id], verbose=1), decimals=2)
y_true = test.rating_type
mean_absolute_error(y_true, y_hat)
print ('Predict model...Done')

#bmThread.active = 0 #deactivate the thread, will exit on the next while loop cycle
#bmThread.join()  # wait for it to exit on its own, since its daemon as a precaution
#print("Done",flush=True)

print ('Evaluate model')
model.evaluate([test.normalized_customer_id, test.product_id], test.rating_type, verbose=2)
print ('Evaluate model...Done')

#topK
num_customers = len(mergeddata_datasets.normalized_customer_id.unique())
num_materials = mergeddata_datasets.product_id.max()

unique_customers = allPositives.normalized_customer_id.unique()
unique_products = allPositives.product_id.unique()

res= []
res2= []

topk = trainerTop.topKRatings(10, model, unique_customers, unique_products, mtype="NFC")

print ("positives_5th")
res.append(trainerTop.topKMetrics(topk, [(positives_5th_split.normalized_customer_id[i], positives_5th_split.product_id[i]) for i in positives_5th_split.id], unique_customers, unique_products))
print(res[-1], flush=True)

f = open("testsplit_results.txt", "a")
f.write(res[-1])
f.close()

print ("allpositives")
res2.append(trainerTop.topKMetrics(topk, [(allPositives.normalized_customer_id[i], allPositives.product_id[i]) for i in allPositives.id], unique_customers, unique_products))
print(res2[-1], flush=True)

f = open("allPositives_results.txt", "a")
f.write(res2[-1])
f.close()
