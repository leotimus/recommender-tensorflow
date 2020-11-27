import numpy as np
import pandas as pd
import keras
import keras.utils
import tensorflow as tf
import time
import os
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from src.AAUFile import *
from getpass import getpass
import smbclient as smbc
import trainers.topKmetrics as trainerTop

if __name__ == "__main__":
   # reconstructed_model = keras.models.load_model('D:/ML/dataset/savedModels/mymodel')
    checkpoint_path = 'D:/ML/dataset/savedModels/mymodel'
    reconstructed_model = tf.keras.models.load_model(checkpoint_path)
    dataset = pd.read_csv('D:/ML/LeoRepository/recommender-tensorflow/src/data/mlBinary.csv', header=0, names=['id','movie_id','user_id', 'rating'])
    train, test = train_test_split(dataset, test_size=0.2)
    #train = pd.read_csv('D:/ML/dataset/split100k/train.csv', header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])
    #test = pd.read_csv('D:/ML/dataset/split100k/test.csv', header=0, names=['customer_id', 'normalized_customer_id', 'material', 'product_id', 'rating_type'])

    # train.drop(columns=['customer_id','material'])
    # test.drop(columns=['customer_id','material'])

    mergeddata_datasets = train.append(test)

    # num_customers = len(mergeddata_datasets.normalized_customer_id.unique())
    # num_materials = mergeddata_datasets.product_id.max()

    # unique_customers = mergeddata_datasets.normalized_customer_id.unique()
    # unique_products = mergeddata_datasets.product_id.unique()

    num_customers = len(dataset.user_id.unique())
    num_materials = len(dataset.movie_id.unique())

    unique_customers = dataset.user_id.unique()
    unique_products = dataset.movie_id.unique()

    res= []
    print(reconstructed_model)

    trainerTop.topKRatings(10, reconstructed_model, unique_customers, unique_products)

    res.append(trainerTop.topKMetrics(res[-1], [(i["user_id"], i["movie_id"]) for i in test], unique_customers, unique_products))

   # res.append(trainerTop.topKMetrics(res[-1], [(i["normalized_customer_id"], i["product_id"]) for i in test], unique_customers, unique_products))
    print(res[-1])


