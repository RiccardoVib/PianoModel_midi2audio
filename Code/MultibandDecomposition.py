import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_bandpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff_1 = cutoff[0] / nyq
    normal_cutoff_2 = cutoff[1] / nyq
    b, a = butter(order, [normal_cutoff_1, normal_cutoff_2], btype='bandpass', analog=False)
    return b, a

# def butter_lowpass_filter(data, cutoff, fs, order=2):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data)
#     return y


# Setting standard filter requirements.
order = 1
fs = 44100.0
cutoff_l = 7000.0
cutoff_h = 14000.0

b, a = butter_lowpass(cutoff_l, fs, order)
b1, a1 = butter_bandpass([cutoff_l, cutoff_h], fs, order)
b2, a2 = butter_highpass(cutoff_h, fs, order)

# Plotting the frequency response.
w, h = freqz(b, a, worN=8000)
w1, h1 = freqz(b1, a1, worN=8000)
w2, h2 = freqz(b2, a2, worN=8000)

# plt.subplot(2, 1, 1)
# plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
# plt.plot(0.5*fs*w/np.pi, np.abs(h1), 'b')
# plt.plot(0.5*fs*w/np.pi, np.abs(h2), 'b')
# plt.plot(cutoff_l, 0.5*np.sqrt(2), 'ko')
# plt.plot(cutoff_h, 0.5*np.sqrt(2), 'ko')
# plt.axvline(cutoff_l, color='k')
# plt.axvline(cutoff_h, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
# plt.subplots_adjust(hspace=0.35)
# plt.show()