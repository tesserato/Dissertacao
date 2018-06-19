import keras as K
import numpy as np
import time

from Hlib import read_wav, normalize_signal, get_sinusoid, save_wav, save_model

# Read the samples
np.random.seed(0)
max_amp = 0
names = []
signals = []

for i in range(1, 6):
  for j in range(1, 6):
    for k in range(1, 5):
      name = str(i) + str(j) + str(k)
      names.append(name)
      Y, _ = read_wav('samples/' + name + '.wav')
      Y, a = normalize_signal(Y)
      Y = np.array(Y + [0])
      signals.append(Y)
      if a > max_amp:
        max_amp = a

names = np.array(names)
signals = np.array(signals)

##
subsample = [0]
subsample = np.array(subsample)
names = names[subsample]
signals = signals[subsample]
##

print(signals.shape)

####################
#### Parameters ####
####################
samples = signals.shape[0]
timesteps = signals.shape[1]
channels = 441
path = './Log/Dense_0.01_dpout_0.01_noise/'
hneurons = 75
epochs = 500
####################
####################
####################

# Model
input = K.layers.Input(shape=(channels,))
layer = K.layers.Dropout(0.01)(input)
layer = K.layers.GaussianNoise(0.01)(layer)
layer = K.layers.Dense(75, activation='tanh')(layer)
layer = K.layers.Dense(75, activation='tanh')(layer)
layer = K.layers.Dense(75, activation='tanh')(layer)
output = K.layers.Dense(1, activation='tanh')(layer)

model = K.models.Model(input, output)
K.utils.print_summary(model)

model.compile(loss='mean_squared_error', optimizer='nadam')

# Epochs
initial_time = time.time()
loss = []
for E in range(epochs):
  print('Epoch', E)
  shuffled_idx = np.random.permutation(np.arange(samples))
  signals = signals[shuffled_idx]
  names = names[shuffled_idx]
  for i in range(samples): # for every sample
    batch_ipt = []
    batch_tgt = []
    for step in range(timesteps - channels - 1):
      batch_ipt.append(signals[i, step: step + channels])
      batch_tgt.append([signals[i, step + channels + 1]])
    batch_ipt = np.array(batch_ipt)
    batch_tgt = np.array(batch_tgt)
    # batch_ipt = np.expand_dims(batch_ipt, 0)
    # batch_tgt = np.expand_dims(batch_tgt, 0)
    batch_loss = model.train_on_batch(batch_ipt, batch_tgt) ### TRAINNING
    loss.append([time.time() - initial_time, batch_loss])
    print('LOSS:', loss[-1][1], 'Time:', loss[-1][0])

save_model(model, path, 'model')
np.savetxt(path + 'loss.csv', np.array(loss), fmt='%.18e', delimiter=',')

for i in range(samples):
  seed = np.array(signals[i][0:channels])
  save_wav(max_amp * seed, name = path + 'seed_' + names[i] + '.wav')
  seed = np.expand_dims(seed, 0)
  # seed = np.expand_dims(seed, 0)
  pred_wave = np.zeros(timesteps)
  for j in range(channels, timesteps):
    predicted = model.predict(seed)
    pred_wave[j] = predicted[0]
    seed = np.roll(seed, -1)
    seed[0, channels - 1] = predicted[0]
  save_wav(max_amp * pred_wave, name=path + 'wave_' + names[i] + '.wav')

