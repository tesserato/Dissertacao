from numpy.fft import rfft, irfft
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
import os

def read_wav(path): # returns signal & fps
  wav = wave.open(path , 'r')
  signal = np.fromstring(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps

def save_wav(signal, name = 'test.wav', fps = 44100): #save .wav file to program folder
    o = wave.open(name, 'wb')
    o.setframerate(fps)
    o.setnchannels(1)
    o.setsampwidth(2)
    o.writeframes(np.int16(signal)) # Int16
    o.close()

def create_signal(N, fps, real_frequency, phase = 0, amplitude = 1, decay = 0):
  frequency = real_frequency * N / fps
  f = lambda x: amplitude * np.exp(-decay * x) * np.cos(phase + 2 * np.pi * frequency * x / N)
  X = np.arange(0,N,1)
  return f(X)

def pad(X, n, value = 0):
  l = X.shape[0]
  if l >= n:
    padded = X[0:n]
  else:
    zeros = np.zeros(n - l) #.fill(np.nan)
    padded = np.hstack([X, zeros])
  # print('p:', padded.shape[0])
  return padded

##########################################
##########################################
################ Settings ################
harmonics = 10
##########################################
##########################################
##########################################

filenames = os.listdir()
filenames = [fn for fn in filenames if '.wav' in fn]
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))
names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

save_path = 'Peaks_plots/'
os.makedirs(save_path, exist_ok=True)

# extract frequencies, amplitudes and decays
AMPLITUDES = []
MAX_amps = []
FREQUENCIES = []
DECAYS = []
for name in names:
  print('\nsample:', name)
  s, fps = read_wav(name + '.wav')
  n = s.shape[0]
  F = rfft(s) * 2 / n
  P = np.abs(F) # P(f); f in {0,1,2,...,n}
  f0 = np.power(2, (int(name) - 49) / 12) * 440

  local_f0 = int(round(f0 / fps * n))
  local_T = int(round( (n / (f0 / fps * n))))
  idxs = argrelmax(P, order=local_T * 2)[0]
  print(idxs * fps / n)
  idx0 = idxs[np.argmin(np.abs(idxs - local_f0))]

  print('T f0:', f0,'| M f0:', idx0 * fps / n)

  idxs = idxs[np.where(idxs >= idx0)][:harmonics]

  naive_idxs = np.argsort(P)[::-1][ : harmonics]
  higher_local_freq = np.max(idxs)
  X = np.arange(higher_local_freq) * fps / n

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(X, P[ : higher_local_freq], '0.5')
  plt.plot(idxs * fps / n, P[idxs],'ro', label='Peaks')
  plt.plot(naive_idxs * fps / n, P[naive_idxs],'kx', label='Max(|FFT|)')
  
  plt.legend()
  plt.savefig(save_path + name + '.png')
  plt.close()