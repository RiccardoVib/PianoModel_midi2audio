import pickle
import random
import os
import numpy as np
#import pretty_midi
import matplotlib.pyplot as plt
#from easyDataset import plot_piano_roll



def get_batches(data_dir, window, number_of_iterations, index,seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    I_O = open(os.path.normpath('/'.join([data_dir, 'Dataset_midi_2_wav.pickle'])), 'rb')
    I_O = pickle.load(I_O)

    N = np.array(I_O['midi']).shape[0]
    n_iteration = N // number_of_iterations

    indeces = [int(n_iteration * index), int((1 + index) * n_iteration)]

    if indeces[1] >= N:
        indeces[1] = N - 1

    midi = np.float32(np.array(I_O['midi'][indeces[0]: indeces[1]]))
    wav = np.float32(np.array(I_O['wav'][indeces[0]: indeces[1]]))

    scaler = np.array(I_O['scaler'])
    del I_O

    N = wav.shape[0]
    N_train = int(N/100 * 85)
    #N_val = (N - N_train)


    all_inp = []
    all_tar = []
    for i in range(N_train):
        for t in range(0, wav.shape[2] - window):
            inp_temp = np.array(midi[i, :, t:t + window])
            all_inp.append(inp_temp.T)

            tar_temp = np.array(wav[i, 0, t:t + window])
            all_tar.append(tar_temp.T)

    x = np.array(all_inp)
    y = np.array(all_tar)

    all_inp = []
    all_tar = []
    for i in range(N_train, N):
        for t in range(0, wav.shape[2] - window):
            inp_temp = np.array(midi[i, :, t:t + window])
            all_inp.append(inp_temp.T)

            tar_temp = np.array(wav[i, 0, t:t + window])
            all_tar.append(tar_temp.T)

    x_val = np.array(all_inp)
    y_val = np.array(all_tar)

    return x, y, x_val, y_val, scaler