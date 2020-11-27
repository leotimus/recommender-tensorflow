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

reconstructed_model = keras.models.load_model('')


