import pickle
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from Preprocess import my_scaler, get_batches
from audio_format import pcm2float

data_dir = '../Files'
file_data = open(os.path.normpath('/'.join([data_dir, 'Dataset_prepared_32.pickle'])), 'rb')
Z = pickle.load(file_data)
x = Z['x']
y = Z['y']
x_val = Z['x_val']
y_val = Z['y_val']
x_test = Z['x_test']
y_test = Z['y_test']
