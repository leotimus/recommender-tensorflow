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

#dataset = pd.read_csv('', header=0, names=['index', 'customer_id', 'material_id', 'is_real'])
dataset = pd.read_csv('', header=0, names=['index', 'customer_id', 'material_id', 'is_real'])

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
customer_input = Input(shape=[1], name='customer-input')
