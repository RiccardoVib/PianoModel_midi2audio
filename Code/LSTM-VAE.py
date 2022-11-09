import os
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import RepeatVector, LSTM, Dense, TimeDistributed, Lambda
from scipy.io import wavfile
from tensorflow.keras import backend as K
from keras import Input, Model
from keras.utils import plot_model
import random

# TODO: make utils.py for the functions
def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

def vae_loss(input_x, original, out, z_log_sigma, z_mean):

    sequence_length = len(input_x)
    reconstruction = K.mean(K.square(original - out)) * sequence_length
    kl = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))

    return reconstruction + kl

def vae_loss2(input_x, decoder1, z_log_sigma, z_mean):
    """ Calculate loss = reconstruction loss + KL loss for each data in minibatch """
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(input_x, decoder1))
    # D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
    kl = 0.5 * K.sum(K.exp(z_log_sigma) + K.square(z_mean) - 1. - z_log_sigma)

    return recon + kl

def sampling(args):
    z_mean, z_log_sigma = args
    batch_size = tf.shape(z_mean)[0] # <================
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_sigma) * epsilon


model_save_dir = '../TrainedModels'  # '/scratch/users/riccarsi/TrainedModels',
save_folder = 'LSTM_VAE'
b_size = 0
learning_rate = 0.001
w_length = 32

data_dir = '../Files'
# Dataset
file_data = open(os.path.normpath('/'.join([data_dir, 'Dataset_prepared_32.pickle'])), 'rb')
data = pickle.load(file_data)
x = data['x'][0:10000]
y = data['y'][0:10000]
x_val = data['x_val']
y_val = data['y_val']
x_test = data['x_test']
y_test = data['y_test']
scaler = data['scaler']

del data
ckpt_flag = True

#input shape = [samples, timestep, features]
features = x.shape[2]
timesteps = x.shape[1]

inter_dim = 32*4
latent_dim = 32*4

# Model
input_x = Input(shape=(timesteps, features))

# Encoder
h = LSTM(inter_dim, activation='relu')(input_x)
#z_layer
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
#z = Lambda(sampling)([z_mean, z_log_sigma])

encoder = Model(input_x, [z_mean, z_log_sigma])
# Decoder
input_z = Input(shape=(latent_dim,))

dec = RepeatVector(timesteps)(input_z)
dec = LSTM(inter_dim, activation='relu', return_sequences=True)(dec)
#dec = TimeDistributed(Dense(features))(dec)
dec = TimeDistributed(Dense(1))(dec)

decoder = Model(input_z, dec)

### encoder + decoder ###

z_mean, z_log_sigma = encoder(input_x)
z = Lambda(sampling)([z_mean, z_log_sigma])
pred = decoder(z)

#vae = Model([inp, inp_original], pred)
vae = Model(input_x, pred)

#vae.add_loss(vae_loss(input_x, inp_original, pred, z_log_sigma, z_mean))
#vae.add_loss(vae_loss2(input_x, pred, z_log_sigma, z_mean)) #<===========
vae.compile(loss='mse', optimizer='adam')
vae.summary()

callbacks = []
#TODO: make a function for this
ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
ckpt_path_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best'))
ckpt_dir_latest = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest'))

if not os.path.exists(os.path.dirname(ckpt_dir)):
    os.makedirs(os.path.dirname(ckpt_dir))
if not os.path.exists(os.path.dirname(ckpt_dir_latest)):
    os.makedirs(os.path.dirname(ckpt_dir_latest))

ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                       save_best_only=True, save_weights_only=True, verbose=1)
ckpt_callback_latest = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path_latest, monitor='val_loss',
                                                              mode='min',
                                                              save_best_only=False, save_weights_only=True,
                                                              verbose=1)
callbacks += [ckpt_callback, ckpt_callback_latest]
latest = tf.train.latest_checkpoint(ckpt_dir_latest)
if latest is not None:
    print("Restored weights from {}".format(ckpt_dir))
    vae.load_weights(latest)
    # start_epoch = int(latest.split('-')[-1].split('.')[0])
    # print('Starting from epoch: ', start_epoch + 1)
else:
    print("Initializing random weights.")


#results = vae.fit(x, y, batch_size=b_size, epochs=300, verbose=0,
          #validation_data=(x_val, y_val), callbacks=callbacks)
#
# results = {
#     'Min_val_loss': np.min(results.history['val_loss']),
#     'Min_train_loss': np.min(results.history['loss']),
#     'b_size': b_size,
#     'learning_rate': learning_rate,
#     'w_length': w_length,
#     'lat_dim': latent_dim,
#     'inter_dim': inter_dim,
#     'Train_loss': results.history['loss'],
#     'Val_loss': results.history['val_loss']
#     }
# with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_it_.txt'])), 'w') as f:
#     for key, value in results.items():
#         print('\n', key, '  : ', value, file=f)
#         pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_it_.pkl'])),'wb'))


#plot_model(vae, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
yhat = vae.predict(x_test, verbose=0)
print(yhat[0, :, 0])

yhat = (scaler[0].inverse_transform(yhat)).reshape(-1)
y_test = (scaler[0].inverse_transform(y_test[:, -1])).reshape(-1)

# Define directories
pred_name = '_pred.wav'
tar_name = '_tar.wav'

pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

if not os.path.exists(os.path.dirname(pred_dir)):
    os.makedirs(os.path.dirname(pred_dir))


wavfile.write(pred_dir, 44100, yhat)
wavfile.write(tar_dir, 44100, y_test)
