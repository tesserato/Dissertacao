import numpy as np
import keras as K
# import os
import itertools

from scipy.fftpack import fft, ifft

from Hlib import sample
from Hlib import normalize_signal
from Hlib import save_wav
from Hlib import save_model

np.random.seed(0)



max_amp = 0
tgt = []
ipt = []
labels = []

for i in range(1, 6):
  for j in range(1, 6):
    for k in range(1, 5):
      ipt.append([(i - 1) / 4,(j - 1) / 4,(k - 1) / 3])
      name =  str(i) + str(j) + str(k)
      s = sample('samples/' + name + '.wav')
      labels.append(name)
      T, Y = s.get_signal()
      Y, a = normalize_signal(Y)
      tgt.append(Y)
      if a > max_amp:
        max_amp = a

tgt = np.array(tgt, dtype='float32')
ipt = np.array(ipt, dtype='float32')

act = 'tanh' # softsign
opt = K.optimizers.nadam()
# opt = K.optimizers.adagrad()


def train(hlayers, logpath, batch, epchs):
  model = K.Sequential()
  if len(hlayers) == 0:
    model.add(K.layers.Dense(tgt.shape[1], activation=act, input_shape=(ipt.shape[1],)))
  else:
    model.add(K.layers.Dense(hlayers[0], activation=act, input_shape=(ipt.shape[1],)))
    # model.add(K.layers.Dropout(0.01))
    for hn in hlayers[1:]:
      model.add(K.layers.Dense(hn, activation=act))
      # model.add(K.layers.Dropout(0.01))
    model.add(K.layers.Dense(tgt.shape[1], activation=act))
    # model.add(K.layers.Dropout(0.01))
  model.compile(loss='mean_squared_error', optimizer=opt)
  tb = K.callbacks.TensorBoard(logpath)
  K.utils.print_summary(model)
  history = model.fit(ipt, tgt, batch_size=batch, epochs=epchs, verbose=0, callbacks=[tb], shuffle=True)
  
  save_model(model, logpath, 'model')
  
  return model



path = './Log/'

m = train([75,75,75], path, batch = 100, epchs = 10)

for i, input in enumerate(ipt):
  input = np.expand_dims(input, 0)
  pred = m.predict(input)
  save_wav(pred * max_amp, name = path + 'wave-' + labels[i] + '.wav')

# %USERPROFILE%/.keras/keras.json
