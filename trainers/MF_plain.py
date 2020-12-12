import numpy as np
import pandas as pd
import keras
import keras.utils
import tensorflow as tf
import time
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

start_time = time.time()

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print ('Loading dataset...')

#dataset = pd.read_csv('//cs.aau.dk/Fileshares/IT703e20/CleanDatasets/binary_cleaned_incl_customers.csv', header=0, names=['index', 'customer_id', 'material_id', 'is_real'])
dataset = pd.read_csv('D:/ML/dataset/binary_cleaned_incl_customers.csv', header=0, names=['index', 'customer_id', 'material_id', 'is_real'])

print ('Dataset loaded')

num_customers = len(dataset.customer_id.unique())
num_materials = len(dataset.material_id.unique())

#shuffle here
dataset = shuffle(dataset)

dataset.drop('index', axis=1, inplace=True)
train, test = train_test_split(dataset, test_size=0.2)

#Build the model
print ("Building model")
latent_dim = 10

material_input = Input(shape=[1],name='material-input')
material_embedding = Embedding(num_material + 1, latent_dim, name='material-embedding')(material_input)
material_vec = Flatten(name='material-flatten')(material_embedding)

customer_input = Input(shape=[1],name='customer-input')
customer_embedding = Embedding(num_users + 1, latent_dim, name='customer-embedding')(customer_input)
customer_vec = Flatten(name='customer-flatten')(customer_embedding)

prod = tf.keras.layers.Dot(axes=1)([material_vec, customer_vec])

model = Model([user_input, movie_input], prod)
model.compile('adam', 'mean_squared_error', metrics=['mse', 'mae'])

plot_model(model, show_shapes=True)

history = model.fit([train.user_id, train.movie_id], train.rating, epochs=10)
pd.Series(history.history['loss']).plot(logy=True)
plt.xlabel("Epoch")
plt.ylabel("Train Error")
plt.show()

y_hat = np.round(model.predict([test.user_id, test.movie_id]), decimals=2)
y_true = test.rating
mean_absolute_error(y_true, y_hat)

model.evaluate([test.user_id, test.movie_id], test.rating)


