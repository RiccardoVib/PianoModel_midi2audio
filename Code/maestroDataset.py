import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from scipy.io import wavfile
from audio_format import pcm2float
from mido import MidiFile
import glob
import pretty_midi

import librosa.display

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))



data_dir = '../Files'
wav = open(os.path.normpath('/'.join([data_dir, 'maestro/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav'])),'rb')
#mid = open(os.path.normpath('/'.join([data_dir, 'maestro/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi'])),'rb')

wav = glob.glob(os.path.normpath('/'.join([data_dir, 'maestro/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.wav'])))
for file in wav:
    fs, wav = wavfile.read(file)

wav = pcm2float(wav[:, 0] + wav[:, 1])
#plt.plot(wav)
#print(len(wav))
#print(len(wav)/fs/60)

mid = pretty_midi.PrettyMIDI(os.path.normpath('/'.join([data_dir, 'maestro/MIDI-Unprocessed_Chamber2_MID--AUDIO_09_R3_2018_wav--1.midi'])))

# plt.figure(figsize=(12, 4))
# plot_piano_roll(mid, 24, 84, fs=fs)

msg_collector = []
notes_collector = []
cc_collector = []

midi_collector = {'type': [], 'note/control': [], 'velocity/value': [], 'start': [], 'end': []}

for inst in mid.instruments:
    msg_collector.append(inst)

for note in range(len(msg_collector[0].notes)):
    notes_collector.append(msg_collector[0].notes[note])

for cc in range(len(msg_collector[0].control_changes)):
    cc_collector.append(msg_collector[0].control_changes[cc])

for i in range(len(notes_collector)):
    midi_collector['start'].append(notes_collector[i].start)

for i in range(max(len(cc_collector), len(notes_collector))):
    notes_collector.append(msg_collector[i][0].notes)
    cc_collector.append(msg_collector[i][0].control_changes)


matrix = mid.get_piano_roll(fs)[24:84]
# for msg in mid.tracks[1]:
#     msg_collector.append(msg)
#
# for i in range(1, len(msg_collector)):
#     if msg_collector[i].type == 'note_on':
#         midi_collector['type'].append(msg_collector[i].type)
#         midi_collector['note/control'].append(msg_collector[i].note)
#         midi_collector['velocity/value'].append(msg_collector[i].velocity)
#         midi_collector['time'].append(msg_collector[i].time)
#     elif msg_collector[i].type == 'control_change':
#         midi_collector['type'].append(msg_collector[i].type)
#         midi_collector['note/control'].append(msg_collector[i].control)
#         midi_collector['velocity/value'].append(msg_collector[i].value)
#         midi_collector['time'].append(msg_collector[i].time)

print(min(midi_collector['channel']))

#prima nota: 0.750 s

tempo= 500000 #length of a beat, in microseconds.
#clocks_per_click=24
ticks_per_beat = 384  #number of ticks per beat
#1 tick = microseconds per beat / 60
#1 tick = 1,000,000 / (24 * 100) = 416.66 microseconds
#32nd_notes_per_beat=8

delta_time = tempo * delta_ticks / ticks_per_beat

#36911/fs = 768,9791666666667 ms