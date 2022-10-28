import os
import pickle
import numpy as np
import tensorflow as tf
from keras.layers import RepeatVector, LSTM, Dense, TimeDistributed, Lambda, Flatten
from keras import backend as K
from keras import Input, Sequential, Model
from keras.utils import plot_model
from GetDataPiano_it import get_batches

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


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
    return z_mean + z_log_sigma * epsilon


seed = 422
model_save_dir = '../TrainedModels'  # '/scratch/users/riccarsi/TrainedModels',
save_folder = 'LSTM_VAE'
b_size = 32
learning_rate = 0.001
w_length = 32

data_dir = '../Files'
# Dataset
number_of_iterations = 3

window = 32

ckpt_flag = True

#input shape = [samples, timestep, features]
features = 25#x.shape[2]
timesteps = window#x.shape[1]

#sequence_in = sequence.reshape(1, timesteps, features)
#sequence_out = sequence.reshape(1, timesteps, features)

#####model

inter_dim = 32*4
latent_dim = 32*4

# timesteps, features
input_x = Input(shape=(timesteps, features))
#intermediate dimension
h = LSTM(inter_dim, activation='relu')(input_x)

#z_layer
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)
z = Lambda(sampling)([z_mean, z_log_sigma])

# Reconstruction decoder
decoder1 = RepeatVector(timesteps)(z)
decoder1 = LSTM(inter_dim, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(features))(decoder1)
decoder1 = Flatten()(decoder1)
decoder1 = Dense(1)(decoder1)

model = Model(input_x, decoder1)
model.add_loss(vae_loss2(input_x, decoder1, z_log_sigma, z_mean)) #<===========
model.compile(loss=None, optimizer='adam')
model.summary()

callbacks = []
if ckpt_flag:
    ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'best', 'best.ckpt'))
    ckpt_path_latest = os.path.normpath(
        os.path.join(model_save_dir, save_folder, 'Checkpoints', 'latest', 'latest.ckpt'))
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
        model.load_weights(latest)
        # start_epoch = int(latest.split('-')[-1].split('.')[0])
        # print('Starting from epoch: ', start_epoch + 1)
    else:
        print("Initializing random weights.")

epochs = 300
for ep in range(epochs):
    for n_iteration in range(number_of_iterations):
        print("Getting data")
        x, y, x_val, y_val, scaler = get_batches(data_dir=data_dir, window=w_length, index=n_iteration,
                                                    number_of_iterations=number_of_iterations, seed=seed)

        results = model.fit(x, y, batch_size=b_size, epochs=1, verbose=2, validation_data=(x_val, y_val), callbacks=callbacks)

        results = {
            'Min_val_loss': np.min(results.history['val_loss']),
            'Min_train_loss': np.min(results.history['loss']),
            'b_size': b_size,
            'learning_rate': learning_rate,
            'w_length': w_length,
            'lat_dim': latent_dim,
            'inter_dim': inter_dim,
            # 'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss']
            }
        # print(results)
        if ckpt_flag:
            with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_it_'+ str(epochs) + '_' + str(n_iteration) + '.txt'])), 'w') as f:
                for key, value in results.items():
                    print('\n', key, '  : ', value, file=f)
                    pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_it_'+ str(epochs) + '_' + str(n_iteration) + '.pkl'])), 'wb'))


#plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
# demonstrate recreation
#yhat = model.predict(x_test, verbose=0)
#print(yhat[0, :, 0])

