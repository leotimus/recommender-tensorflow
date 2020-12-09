import numpy as np
import pandas as pd
import keras
import keras.utils
import tensorflow as tf
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
    
    reconstructed_model = keras.models.load_model('//cs.aau.dk/Fileshares/IT703e20/NCF_savedModels/2m')

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


        #topK
    unique_customers = allPositives.normalized_customer_id.unique()
    unique_products = allPositives.product_id.unique()

    res= []

    topk = trainerTop.topKRatings(10, reconstructed_model, unique_customers, unique_products, mtype="NFC")

    res.append(trainerTop.topKMetrics(topk, [(positives_5th_split.normalized_customer_id[i], positives_5th_split.product_id[i]) for i in positives_5th_split.id], unique_customers, unique_products))
    print(res[-1], flush=True)


