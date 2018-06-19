import numpy as np
import keras as K

from Hlib import read_wav, save_wav
from Hlib import save_model
from Hlib import normalize_cols, denormalize_cols
from Hlib import create_signal
import os


np.random.seed(0)
# 44100 x 100 = 4 410 000
#                  54 031

piece_name = 'samples'
path = 'Predicted_no_overlap_'+ piece_name +'/'
epcs = 1000
timesteps = 10
overlap = 0
dt='float32'

names = []
ipt = []
tgt = []
os.makedirs(path, exist_ok=True)

# |> Generating inputs and names. Reading and splitting targets
for i in range(1, 5 + 1):
  for j in range(1, 5 + 1):
    for k in range(1, 4 + 1):
      name = str(i) + str(j) + str(k)
      names.append(name)
      ipt.append([i, j, k])
      s, _ = read_wav('samples/' + name + '.wav')
      n = s.shape[0]
      l = n // timesteps
      s = np.hstack([s, np.zeros((overlap))])
      l_tgt = []
      for q in range(0, timesteps):
        idx_0 = q * l
        idx_1 = idx_0 + l + overlap
        l_tgt.append(s[idx_0 : idx_1])
      tgt.append(l_tgt)

tgt = np.array(tgt, dt)
amp = np.max(np.abs(tgt))
tgt = tgt / amp

n_samples = tgt.shape[0]

ipt = np.array(ipt, dt)
ipt, maxs_ipt, mins_ipt = normalize_cols(ipt)
# ipt[:,2] = 0

print('samples x timesteps x features:', ipt.shape, tgt.shape, amp, overlap)

# |> Keras Model
act = 'tanh'
opt = K.optimizers.nadam()
hneurons1 = 100
hneurons2 = 100
hneurons3 = 100

input = K.layers.Input(batch_shape=(None, ipt.shape[1]))

layer = K.layers.Dense(hneurons1, activation=act)(input)

layer = K.layers.Reshape((1,-1))(layer)

# layer = K.layers.GaussianNoise(0.1)(layer)
# layer = K.layers.GaussianNoise(0.2)(layer)
# layer = K.layers.GaussianNoise(0.3)(layer)

layer = K.layers.ZeroPadding1D((0,9))(layer)

layer = K.layers.GRU( 
  hneurons2,
  activation=act,
  recurrent_activation=act,
  return_sequences=True,
  dropout=0.0,
  recurrent_dropout=0.0
)(layer)

dense = K.layers.Dense(hneurons3, activation=act)
layer = K.layers.TimeDistributed(dense)(layer)

dense = K.layers.Dense(tgt.shape[2], activation=act)
output = K.layers.TimeDistributed(dense)(layer)



model = K.models.Model(input, output)
K.utils.print_summary(model)
model.compile(loss='mean_squared_error', optimizer=opt)

tb = K.callbacks.TensorBoard(path)
history = model.fit(ipt, tgt, batch_size=n_samples, epochs=epcs, verbose=1, callbacks=[tb], shuffle=True)

save_model(model, path, 'model')


# def blend(a, b):
#   steps = a.shape[0]
#   X = np.linspace(-3 , 3 , steps)
#   Y = (np.tanh(X) + 1) / 2
#   return a * (1 - Y) + b * Y

def blend(a, b): # smoothstep
  steps = a.shape[0]
  X = np.linspace(0 , 1 , steps)
  Y = 3 * X ** 2 - 2 * X** 3
  return a * (1 - Y) + b * Y

extended_ipt = []
extended_names = []
for i in np.arange(1, 5 + 0.1, 0.5):
  for j in np.arange(1, 5 + 0.1, 0.5):
    for k in np.arange(1, 4 + 0.1, 0.5):
      name = str(i) + '_' + str(j) + '_' + str(k)
      extended_names.append(name)
      extended_ipt.append([i, j, k])

extended_ipt = np.array(extended_ipt, dt)
extended_ipt, maxs_ipt, mins_ipt = normalize_cols(extended_ipt)

predictions = model.predict(extended_ipt)

if overlap == 0:
  predictions = np.reshape(predictions, (extended_ipt.shape[0],-1))
  for i, w in enumerate(predictions):
    save_wav(w * amp, path + extended_names[i] + '.wav')
else:
  for i, p in enumerate(predictions):
    w = p[0]
    for j in range(1, timesteps):
      x = p[j]
      b = blend(w[-overlap:], x[:overlap])
      w = np.hstack([w[:-overlap], b ,x[overlap:]])
    save_wav(w[:-overlap] * amp, path + extended_names[i] + '.wav')