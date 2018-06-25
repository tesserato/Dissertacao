import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft, fft, ifft
import wave


def read_wav(path): # returns signal & fps
  wav = wave.open(path , 'r')
  signal = np.fromstring(wav.readframes(-1) , np.int16)
  fps = wav.getframerate()
  return signal, fps

save_path = 'Decay_plots/'
os.makedirs(save_path, exist_ok=True)

filenames = os.listdir()

print(filenames)

filenames = [fn for fn in filenames if '.wav' in fn]
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))
# names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

for name in names:
  s, fps = read_wav(name + '.wav')
  n = s.shape[0]
  local_frequency = np.argmax(np.abs(rfft(s)))
  x = np.arange(n)
  y = np.exp(-2 * np.pi * x * local_frequency / n * 1j)
  Z = np.multiply(y, s)
  middle = int(round(n/2))

  a1 = np.sum(np.abs(Z[ : middle])) / middle
  a2 = np.sum(np.abs(Z[middle : ])) / middle
  decay = 2 * np.log(a1 / a2) / n
  a = a1 / np.exp(-decay * n / 4)
  D = a * np.exp(-decay * x)

  fig = plt.figure()
  fig.set_size_inches(1000 / fig.dpi, 400 / fig.dpi)
  plt.plot(s,'0.5', label = 'wave: ' + name)
  plt.plot(D, 'k', label='d =' + str(decay) + ', f =' + str(local_frequency * fps / n))
  plt.legend()
  plt.savefig(save_path + name + '.png')
  plt.close()