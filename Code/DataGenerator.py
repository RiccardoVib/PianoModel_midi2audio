import numpy as np
import tensorflow as tf
import os
import pickle

def create_windows(data_dir, window):
    I_O = open(os.path.normpath('/'.join([data_dir, 'Dataset_midi_2_wav.pickle'])), 'rb')
    I_O = pickle.load(I_O)
    wav = np.float32(np.array(I_O['wav']))
    midi = np.float32(np.array(I_O['midi']))

    N = wav.shape[0]
    N_train = int(N/100 * 85)
    N_val = (N - N_train)

    x = []
    y = []

    for i in range(N):
        for t in range(wav.shape[2] - window):
            x_temp = np.array(midi[i, :, t :t + window])
            x.append(x_temp.T)

            y_temp = np.array(wav[i, 0, t :t + window])
            y.append(y_temp.T)

    x_train = np.array(x[:N_train])
    y_train = np.array(y[:N_train])
    x_val = np.array(x[N_val:])
    y_val = np.array(y[N_val:])
    return x_train, y_train, x_val, y_val



class dataGenerator(tf.data.Dataset):
    #def __init__(self, list_IDs, labels, batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True):
    def __init__(self, list_IDs, batch_size=32, dim=(32, 32, 32), n_channels=1):
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        #self.n_classes = n_classes
        #self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        #if self.shuffle == True:
            #np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __data_generation(self, list_IDs_temp):
        # Initialization-
        X = np.empty((self.batch_size, self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=np.float32)
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data /' + ID + '.npy')
            # Store class
            y[i] = self.labels[ID]

        return X, Y

    def __getitem__(self, index):

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y



# from my_classes import DataGenerator
#
# params = {‘dim’: (32,32,32),
#  ‘batch_size’: 128,
#  ‘n_classes’: 6,
#  ‘n_channels’: 1,
#  ‘shuffle’: True}
#
# partition = # IDs
# labels = # Labels
#
# training_generator = DataGenerator(partition[‘train’], labels, **params)
# validation_generator = DataGenerator(partition[‘validation’], labels, **params)
#
# model = Sequential()
# model.compile()
#
# model.fit_generator(generator=training_generator,
#  validation_data=validation_generator,
#  use_multiprocessing=True,
#  workers=6)