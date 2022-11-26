import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy
from scipy.io import wavfile
from audio_format import pcm2float
import glob
import pretty_midi

import librosa.display

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))



data_dir = '../Files'

wav = glob.glob(os.path.normpath('/'.join([data_dir, 'PianoDatasetsSingleNoteLong.wav'])))
for file in wav:
    fs, wav = wavfile.read(file)

if wav.ndim != 1:
    wav = pcm2float(wav[:, 0] + wav[:, 1])

wav = scipy.signal.resample_poly(wav, 1, 2)
fs = fs/2
#plt.plot(wav)
#print(len(wav))
#print(len(wav)/fs/60)

mid = pretty_midi.PrettyMIDI(os.path.normpath('/'.join([data_dir, 'FileDisk.mid'])))

# plt.figure(figsize=(12, 4))
# plot_piano_roll(mid, 24, 84, fs=fs)

plot_piano_roll(mid, 24, 84, fs=fs)
matrix = mid.get_piano_roll(fs)[24:84]

msg_collector = []
notes_collector = []

midi_collector = {'note': [], 'velocity': [], 'start': [], 'end': []}

for inst in mid.instruments:
    msg_collector.append(inst)

for note in range(len(msg_collector[0].notes)):
    notes_collector.append(msg_collector[0].notes[note])

file_data = open(os.path.normpath('/'.join([data_dir, 'MidiDatasetLong22050.pickle'])), 'wb')
pickle.dump(notes_collector, file_data)
file_data.close()

# for i in range(len(notes_collector)):
#     midi_collector['start'].append(notes_collector[i].start)
#

matrix = mid.get_piano_roll(fs)[47:72]
Nwav = len(wav)
N = matrix.shape[1]

matrix = np.pad(matrix, (0, Nwav-N), mode='constant')


matrix1 = matrix[:, :N//2]
matrix3 = matrix[:, N//2:]

matrix1.shape[1] + matrix3.shape[1] == N

N1 = matrix1.shape[1]
matrix2 = matrix1[:, N1//2:]
matrix1 = matrix1[:, :N1//2]

N2 = matrix3.shape[1]
matrix4 = matrix3[:, N1//2:]
matrix3 = matrix3[:, :N1//2]

plt.plot(matrix1)
