import numpy as np
import os
import tensorflow as tf
from TrainFunctionality import combinedLoss, STFT_loss_function
from scipy.io import wavfile
from scipy import signal
from keras.layers import Input, Dense, LSTM
from keras.models import Model
# from tensorflow.keras.optimizers import Adam, SGD
# from tensorflow.keras.layers import Attention, Activation

import pickle
import matplotlib.pyplot as plt

from GetDataPiano_it import get_batches


#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#

def trainED(data_dir, epochs, seed=422, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 32)
    learning_rate = kwargs.get('learning_rate', 0.001)
    encoder_units = kwargs.get('encoder_units', [64])
    decoder_units = kwargs.get('decoder_units', [64])
    dnn_units = kwargs.get('dnn_units', 64)

    if encoder_units[-1] != decoder_units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    model_save_dir = kwargs.get('model_save_dir', '/scratch/users/riccarsi/TrainedModels')
    save_folder = kwargs.get('save_folder', 'ED_Testing')
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    loss_type = kwargs.get('loss_type', 'mse')
    w_length = kwargs.get('w_length', 16)
    type_ = kwargs.get('type_', 'int')
    inference = kwargs.get('inference', False)
    generate_wav = kwargs.get('generate_wav', None)
    activation = kwargs.get('activation', 'sigmoid')

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    layers_enc = len(encoder_units)
    layers_dec = len(decoder_units)
    n_units_enc = ''
    for unit in encoder_units:
        n_units_enc += str(unit) + ', '

    n_units_dec = ''
    for unit in decoder_units:
        n_units_dec += str(unit) + ', '

    n_units_enc = n_units_enc[:-2]
    n_units_dec = n_units_dec[:-2]

    # T past values used to predict the next value
    features = 25  # x.shape[2]
    timesteps = 32*6  # x.shape[1]
    encoder_inputs = Input(shape=(timesteps//2, features), name='enc_input')

    encoder_dnn = Dense(dnn_units, name='Dense_enc')(encoder_inputs)

    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(encoder_dnn)
        for i, unit in enumerate(encoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(encoder_dnn)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(timesteps//2, features), name='dec_input')

    decoder_dnn = Dense(dnn_units, name='Dense_dec')(decoder_inputs)

    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        outputs = LSTM(first_unit_decoder, return_sequences=True, name='LSTM_De0', dropout=drop)(decoder_dnn,
                                                                                                 initial_state=encoder_states)
        for i, unit in enumerate(decoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_De' + str(i + 1), dropout=drop)(outputs)
        outputs, _, _ = LSTM(last_unit_decoder, return_sequences=True, return_state=True, name='LSTM_DeFin',
                             dropout=drop)(outputs)
    else:
        outputs, _, _ = LSTM(first_unit_decoder, return_sequences=True, return_state=True, name='LSTM_De',
                             dropout=drop)(
            decoder_dnn,
            initial_state=encoder_states)

    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    decoder_outputs = Dense(timesteps//2, activation=activation, name='DenseLay')(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=STFT_loss_function, optimizer=opt)

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

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001, patience=20,
                                                               restore_best_weights=True, verbose=0)
    callbacks += [early_stopping_callback]
    if not inference:
        # train
        number_of_iterations = 10

        for n_iteration in range(number_of_iterations):
            print("Getting data")

            x, y, x_val, y_val, scaler = get_batches(data_dir=data_dir, window=w_length, index=n_iteration,
                                                     number_of_iterations=number_of_iterations, seed=seed)

            results = model.fit([x[:, :-1, :], x[:, -1, :].reshape(x.shape[0], 1, x.shape[2])], y, batch_size=b_size,
                                epochs=epochs, verbose=0,
                                validation_data=(
                                [x_val[:, :-1, :], x_val[:, -1, :].reshape(x_val.shape[0], 1, x_val.shape[2])], y_val),
                                callbacks=callbacks)

            print(n_iteration)
            print("Training done")

            results = {
                'Min_val_loss': np.min(results.history['val_loss']),
                'Min_train_loss': np.min(results.history['loss']),
                'b_size': b_size,
                'learning_rate': learning_rate,
                'drop': drop,
                'opt_type': opt_type,
                'loss_type': loss_type,
                'activation': activation,
                'type_': type_,
                'layers_enc': layers_enc,
                'layers_dec': layers_dec,
                'n_units_enc': n_units_enc,
                'n_units_dec': n_units_dec,
                'w_length': w_length,
                # 'Train_loss': results.history['loss'],
                'Val_loss': results.history['val_loss']
            }
        if ckpt_flag:
            with open(os.path.normpath(
                    '/'.join([model_save_dir, save_folder, 'results_it_.txt'])), 'w') as f:
                for key, value in results.items():
                    print('\n', key, '  : ', value, file=f)
                pickle.dump(results,
                            open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results_it_.pkl'])),
                                 'wb'))

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)

    x, y, x_val, y_val, scaler = get_batches(data_dir=data_dir, window=w_length, index=0,
                                             number_of_iterations=10, seed=seed)

    x_test = x_val
    y_test = y_val
    test_loss = model.evaluate([x_test[:, :-1, :], x_test[:, -1, :].reshape(x_test.shape[0], 1, features)], y_test,
                               batch_size=b_size, verbose=0)

    print('Test Loss: ', test_loss)
    if generate_wav is not None:
        predictions = model.predict([x_test[:, :-1, :], x_test[:, -1, :].reshape(x_test.shape[0], 1, features)],
                                    batch_size=b_size)

        predictions = (scaler[0].inverse_transform(predictions)).reshape(-1)
        y_test = (scaler[0].inverse_transform(y_test)).reshape(-1)

        # Define directories
        pred_name = '_pred.wav'
        tar_name = '_tar.wav'

        pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
        tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

        if not os.path.exists(os.path.dirname(pred_dir)):
            os.makedirs(os.path.dirname(pred_dir))

        predictions = predictions.astype('int16')
        y_test = y_test.astype('int16')

        wavfile.write(pred_dir, 44100, predictions)
        wavfile.write(tar_dir, 44100, y_test)

    results = {'Test_Loss': test_loss}

    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    return results


if __name__ == '__main__':
    data_dir = '../Files'  # /scratch/users/riccarsi/Files'
    seed = 422
    trainED(data_dir=data_dir,
            model_save_dir='../TrainedModels',  # '/scratch/users/riccarsi/TrainedModels',
            save_folder='ED_piano',
            ckpt_flag=True,
            b_size=128,
            learning_rate=0.001,
            encoder_units=[64],
            decoder_units=[64],
            dnn_units=64,
            epochs=1,
            loss_type='STFT',
            activation='softmax',
            generate_wav=1,
            w_length=32*6,
            type_='int',
            inference=False)