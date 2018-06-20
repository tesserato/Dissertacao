import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft
import wave
import numpy as np
import os
# from Hlib import read_wav, save_wav, create_signal


def read_wav(path): # returns signal & fps
  wav = wave.open(path , 'r')
  signal = np.fromstring(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps



path = 'FT_plots/'
os.makedirs(path, exist_ok=True)

filenames = os.listdir()

print(filenames)

filenames = [fn for fn in filenames if '.wav' in fn]
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))



for name in names:
  s, fps = read_wav(name + '.wav')
  n = s.shape[0]
  local_max_freq = int(round(20000 / fps * n))
  FT = rfft(s)[ :local_max_freq]

  
  Y_pwr = np.abs(FT)
  X = np.linspace(0, 20000, Y_pwr.shape[0])
  M = np.max(Y_pwr)

  Y_real = FT.real / M
  Y_imag = FT.imag / M
  Y_pwr = Y_pwr / M


  fig = plt.figure(1)
  fig.set_size_inches((n / 50) / fig.dpi, 1200 / fig.dpi)
  
  plt.subplot(311)
  plt.plot(X, Y_pwr, '0.3', label='Max(|F|) =' + str(M), linewidth=.5)
  plt.legend()
  plt.ylabel('Power', fontsize=14, color='k')
  plt.axis([20, 20000, 0, 1.01 ])

  plt.subplot(312)
  plt.plot(X, Y_real, '0.3', linewidth=.5)
  plt.ylabel('Real', fontsize=14, color='k')
  plt.axis([20, 20000,-1.01, 1.01 ])

  plt.subplot(313)
  plt.plot(X, Y_imag, '0.3', linewidth=.5)
  plt.ylabel('Imaginary', fontsize=14, color='k')
  plt.axis([20, 20000,-1.01, 1.01 ])


  
  plt.savefig(
    path + name.replace('.wav','.png'),
    frameon=False,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0
    )

  plt.close('all')







