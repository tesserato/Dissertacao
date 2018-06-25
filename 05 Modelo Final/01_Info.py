from numpy.fft import rfft, irfft
import numpy as np
import matplotlib.pyplot as plt
from Hlib import read_wav, save_wav, create_signal
from scipy.signal import argrelmax
from scipy.optimize import curve_fit
import os

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
piece_name = 'piano'
partials = 100
##########################################
##########################################
##########################################

filenames = os.listdir('00_samples/' + piece_name + '/')
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))
names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

save_path = '01_Info/' + piece_name + '/'
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path + 'graphs/', exist_ok=True)

# extract frequencies, amplitudes and decays per partials
max_amps = []
max_decays = []
fractions_of_max_amps = []
fractions_of_max_decays = []
Mfreqs_over_Tfreqs = []
phases = []

for name in names:
  print('\nsample:', name)
  s, fps = read_wav('00_samples/' + piece_name + '/' + name + '.wav')
  n = s.shape[0]
  F = rfft(s) * 2 / n
  P = np.abs(F)
  max_amp = np.max(P)                                     #<| <| <| <| <|
  f0 = np.power(2, (int(name) - 49) / 12) * 440

  local_f0 = int(round(f0 / fps * n))
  idxs = argrelmax(P, order=local_f0 // 2)[0] #Local Frequencies
  idx0 = idxs[np.argmin(np.abs(idxs - local_f0))]
  idxs = idxs[np.where(idxs >= idx0)][:partials]
  fs_x_partials = idxs * fps / n                                   #<| <| <| <| <|
  as_x_partials = P[idxs]                                          #<| <| <| <| <|
  ps_x_partials = np.angle(F)                                      #<| <| <| <| <|

  middle = int(round(n/2))
  x = np.arange(n)
  ds_x_partials = np.zeros(partials)
  for i, local_frequency in enumerate(idxs):
    y = np.exp(-2.0 * np.pi * local_frequency * x / n * 1j)
    Z = y * s
    a1 = np.abs(np.sum(Z[ : middle])) / middle
    a2 = np.abs(np.sum(Z[middle : ])) / middle
    ds_x_partials[i] = 2 * np.log(a1 / a2) / n                     #<| <| <| <| <|

  max_decay = np.max(np.abs(ds_x_partials))                                #<| <| <| <| <|
  max_amps.append(max_amp)
  max_decays.append(max_decay)
  fractions_of_max_amps.append(pad(as_x_partials / max_amp, partials))
  fractions_of_max_decays.append(pad(ds_x_partials / max_decay, partials))
  inharms = fs_x_partials / (f0 * np.arange(1, fs_x_partials.shape[0] + 1))
  Mfreqs_over_Tfreqs.append(pad(inharms, partials))
  phases.append(pad(ps_x_partials, partials))



  higher_local_freq = np.max(idxs)
  X = np.arange(higher_local_freq) * fps / n
  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(X, P[ : higher_local_freq], '0.5')
  plt.plot(idxs * fps / n, P[idxs],'ro', label='Peaks')
  plt.legend()
  plt.savefig(save_path + 'graphs/01_peaks_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(fractions_of_max_amps[-1], 'k.-')
  plt.savefig(save_path + 'graphs/02_fractions_of_max_amps_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(fractions_of_max_decays[-1], 'k.-')
  plt.savefig(save_path + 'graphs/03_fractions_of_max_decays_' + name + '.png')
  plt.close()

  fig = plt.figure()
  fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
  plt.plot(Mfreqs_over_Tfreqs[-1], 'k.-')
  plt.savefig(save_path + 'graphs/04_Mfreqs_over_Tfreqs_' + name + '.png')
  plt.close()


Mfreqs_over_Tfreqs = np.array(Mfreqs_over_Tfreqs)
fractions_of_max_amps = np.array(fractions_of_max_amps)
fractions_of_max_decays = np.array(fractions_of_max_decays)
phases = np.array(phases)

max_amps = np.array(max_amps)
max_decays = np.array(max_decays)

np.savetxt(save_path + 'Mfreqs_over_Tfreqs.csv', Mfreqs_over_Tfreqs, delimiter=',')
np.savetxt(save_path + 'fractions_of_max_amps.csv', fractions_of_max_amps, delimiter=',')
np.savetxt(save_path + 'fractions_of_max_decays.csv', fractions_of_max_decays, delimiter=',')
np.savetxt(save_path + 'phases.csv', phases, delimiter=',')

np.savetxt(save_path + 'max_amps.csv', max_amps, delimiter=',')
np.savetxt(save_path + 'max_decays.csv', max_decays, delimiter=',')

keys = names.astype(np.int)

fig = plt.figure()
fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
plt.plot(keys, max_amps, 'k.-')
plt.savefig(save_path + 'graphs/05_max_amps_' + name + '.png')
plt.close()

fig = plt.figure()
fig.set_size_inches(2000 / fig.dpi, 400 / fig.dpi)
plt.plot(keys, max_decays, 'k.-')
plt.savefig(save_path + 'graphs/06_max_decays_' + name + '.png')
plt.close()

exit()

X = np.arange(1, partials + 1, 1)
XY = np.tile(X, (AMPLITUDES.shape[0], 1))
keys = [int(n) for n in names]

# Keys x Max Amplitude
plt.plot(keys, MAX_amps, 'k.')
plt.savefig(save_path + 'graphs/Keys x Max Amplitude.png')
plt.close()

# KxMA = np.vstack([keys, MAX_amps]).T
# np.savetxt(save_path + 'Keys x Max Amplitude.csv', KxMA, delimiter=',')


# Partials x Fraction of Max Amplitude
plt.plot(XY, AMPLITUDES, 'k.')
plt.savefig(save_path + 'graphs/Partials x Fraction of Max Amplitude.png')
plt.close()

# PxFMA = np.vstack([np.ndarray.flatten(XY),np.ndarray.flatten(AMPLITUDES)]).T
# np.savetxt(save_path + 'Partials x Fraction of Max Amplitude.csv', PxFMA, delimiter=',')



# Partials x Average Fraction of Max Amplitude
avgAmp = np.average(AMPLITUDES, 0)
plt.plot(X, avgAmp, 'k.')
plt.savefig(save_path + 'graphs/Partials x Average Fraction of Max Amplitude.png')
plt.close()

# PxavgA = np.vstack([np.ndarray.flatten(X),np.ndarray.flatten(avgAmp)]).T
# np.savetxt(save_path + 'Partials x Average Fraction of Max Amplitude.csv', PxavgA, delimiter=',')



# Frequencies x Fractions of Max Decay
MD = np.max(DECAYS)
print('Max Decay =', MD)
plt.plot(FREQUENCIES[np.where(DECAYS>=0)], DECAYS[np.where(DECAYS>=0)] / MD, 'k.')
plt.savefig(save_path + 'graphs/Frequencies x Fractions of Max Decay.png')
plt.close()

# FxD = np.vstack([np.ndarray.flatten(FREQUENCIES[np.where(DECAYS>=0)]),np.ndarray.flatten(DECAYS[np.where(DECAYS>=0)] / MD)]).T
# np.savetxt(save_path + 'Frequencies x Fractions of Max Decay.csv', FxD, delimiter=',')



# Partials x Harmonics_over_Partials
T_freqs = np.zeros((len(names), partials))
for i, name in enumerate(names):
  k = int(name)
  for j in range(partials):    
    T_freqs[i, j] = np.power(2, (k - 49) / 12) * 440 * (j + 1)

ratio = FREQUENCIES / T_freqs
nonzero = np.where(FREQUENCIES > 0)
plt.plot(XY[nonzero], ratio[nonzero], 'k.')
plt.savefig(save_path + 'graphs/Partials x Harmonics_over_Partials.png')
plt.close()

# PxHoP = np.vstack([np.ndarray.flatten(XY[nonzero]),np.ndarray.flatten(ratio[nonzero])]).T
# np.savetxt(save_path + 'Partials x Harmonics_over_Partials.csv', PxHoP, delimiter=',')





# avg power: 52991.72999130899




#     w += create_signal(n, fps, real_f, np.random.uniform(0, 2 * np.pi), 5000 * a, d)
#     parameters += ',' + str(real_f) + ',' + str(a) + ',' + str(d)
#     loss = np.mean((s-w)**2)
#     print('real freq:', real_f, 'loss:', loss)
#     amps.append(a)

#     plt.figure(4)
#     plt.subplot(211)
#     plt.title('Frequency x Decay')
#     plt.plot(real_f, d,'k.')
#     plt.subplot(212)
#     plt.title('Amplitude x Decay')
#     plt.plot(a, d,'k.')

#   amps = np.array(amps)
#   amps = amps / np.max(amps)
#   amplitudes.append(amps)
#   plt.figure(3)
#   plt.plot(original_amps, '-o')
#   plt.plot(amps, '-x')
#   plt.savefig(save_path + 'graphs/03_TxF_' + name + '.png')
#   plt.close()
#   # original_amplitudes.append(np.hstack([original_amps, np.zeros(harmonics - original_amps.shape[0])]))
#   original_amplitudes.append(pad(original_amps, harmonics))
#   parameters += '\n'
#   save_wav(w, save_path + name + '.wav')

# plt.savefig(save_path + 'graphs/04_D.png')
# plt.close()

# #   save_wav(s - w, save_path + 'r_' + name + '.wav')

# file = open(save_path + '#parameters.csv','w')
# file.writelines([line for line in parameters])
# file.close()

# # amplitudes = np.vstack(amplitudes)
# original_amplitudes = np.array(original_amplitudes)
# np.savetxt(save_path + '#amplitudes.csv', original_amplitudes, delimiter=',')
# # print(amplitudes.shape)