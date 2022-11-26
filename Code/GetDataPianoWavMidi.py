import pickle
import random
import os
import numpy as np
#import pretty_midi
import matplotlib.pyplot as plt
import pretty_midi

from Preprocess import my_scaler, get_batches
from audio_format import pcm2float
#from easyDataset import plot_piano_roll



def create_I_O_data(data_dir, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    fs = 44100
    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    I_O = open(os.path.normpath('/'.join([data_dir, 'Dataset_midi_2_wav22050.pickle'])), 'rb')
    I_O = pickle.load(I_O)
    wav = np.float32(np.array(I_O['wav']))
    midi = np.float32(np.array(I_O['midi']))
    scaler = np.array(I_O['scaler'])

    N = wav.shape[0]
    N_train = int(N/100 * 70)
    N_val = (N - N_train) // 2

    all_inp = []
    all_tar = []
    window = 32
    for i in range(N_train):
        for t in range(wav.shape[2] - window):
            inp_temp = np.array(midi[i, :, t :t + window])
            all_inp.append(inp_temp.T)

            tar_temp = np.array(wav[i, 0, t :t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x = all_inp
    y = all_tar

    all_inp = []
    all_tar = []

    for i in range(N_train, N_train + N_val):
        for t in range(wav.shape[2] - window):
            inp_temp = np.array(midi[i, :, t :t + window])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(wav[i, 0, t :t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x_val = all_inp
    y_val = all_tar

    all_inp = []
    all_tar = []

    for i in range(N_train + N_val, N):
        for t in range(wav.shape[2] - window):
            inp_temp = np.array(midi[i, :, t:t + window])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(wav[i, 0, t:t + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    x_test = all_inp
    y_test = all_tar

    return x, y, x_val, y_val, x_test, y_test, scaler


def get_data(data_dir, seed=422):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    fs = 44100
    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    wav = open(os.path.normpath('/'.join([data_dir, 'NotesDatasetLong.pickle'])), 'rb')
    midi = open(os.path.normpath('/'.join([data_dir, 'MidiDatasetLong22050.pickle'])), 'rb')

    wav = pickle.load(wav)
    midi = pickle.load(midi)

    signals = np.array(wav['signal'])
    notes = np.array(wav['note'])
    #vels = np.array(Z['velocity'])

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler(feature_range=(-1, 1))
    scaler.fit(signals)
    signals = scaler.transform(signals)

    # scaler_note = my_scaler()
    # scaler_note.fit(notes)
    # notes = scaler_note.transform(notes)
    #
    # scaler_vel = my_scaler()
    # scaler_vel.fit(vels)
    # vels = scaler_vel.transform(vels)

    #scaler = [scaler_sig, scaler_note, scaler_vel]
    #signals = scaler.inverse_transform(signals)

    #zero_value = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    signals, notes, midi = get_batches(signals, notes, midi, 1)
    piano_roll = []
    for i in range(len(midi)):
        midi[0].end = midi[0].end - midi[0].start
        midi[0].start = 0.0

        new = pretty_midi.PrettyMIDI()
        # Create an Instrument instance for a cello instrument
        program = pretty_midi.instrument_name_to_program('Cello')
        piano = pretty_midi.Instrument(program=program)
        # Iterate over note names, which will be converted to note number later
        piano.notes.append(midi[0])
        # Add the cello instrument to the PrettyMIDI object
        new.instruments.append(piano)
        # Write out the MIDI data
        #new.write('cello-C-chord.mid')

        roll = new.get_piano_roll(fs)[47:72]
        dim = signals[0].shape[1] - roll.shape[1]
        roll = np.pad(roll, [(0, 0), (0, dim)], mode='constant', constant_values=0)
        piano_roll.append(roll)

    piano_roll = np.array(piano_roll)
    scaler_roll = my_scaler(feature_range=(0, 1))
    scaler_roll.fit(piano_roll)
    piano_roll = scaler_roll.transform(piano_roll)

    scaler = [scaler, scaler_roll]

    return np.array(signals), np.array(piano_roll), scaler

if __name__ == '__main__':

    data_dir = '../Files'

    signals, piano_roll, scaler = get_data(data_dir=data_dir, seed=422)
    data = {'wav': signals, 'midi': piano_roll, 'scaler': scaler}

    file_data = open(os.path.normpath('/'.join([data_dir, 'Dataset_midi_2_wav22050.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

    # x, y, x_val, y_val, x_test, y_test, scaler = create_I_O_data(data_dir=data_dir, seed=422)
    #
    # data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test,'scaler': scaler}
    #
    # file_data = open(os.path.normpath('/'.join([data_dir, 'Dataset_prepared_32.pickle'])), 'wb')
    # pickle.dump(data, file_data)
    # file_data.close()