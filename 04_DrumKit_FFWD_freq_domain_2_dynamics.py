import numpy as np
import keras as K
from numpy.fft import rfft, irfft
from Hlib import read_wav, save_wav, save_model, normalize_cols, denormalize_cols, create_signal
import os

def scaled_tanh(x):
  return K.activations.tanh(3 * x)

np.random.seed(0)

piece_name = 'toms'
max_dynamic1 = 5
max_dynamic2 = 5
round_robins = 4

path = 'Predicted_5k/' + piece_name + '/'
epcs = 1000
inf_norm = -1
sup_norm = 1



os.makedirs(path, exist_ok=True)

names = []
ipt = []
tgt = []


xnames = []
xipt = []
for i in np.linspace(1, max_dynamic1, max_dynamic1 * 2 - 1):
  for j in np.linspace(1, max_dynamic2, max_dynamic2 * 2 - 1):
    for k in np.linspace(1, round_robins, round_robins * 2 - 1):
      xnames.append(str(i) + '-' + str(j) + '-' + str(k))
      xipt.append([i, j, k])
xipt = np.array(xipt)
xipt, maxs_xipt, mins_xipt = normalize_cols(xipt,inf_norm,sup_norm)



original_len = 0


for i in range(1, max_dynamic1 + 1, 1):
  for j in range(1, max_dynamic2 + 1, 1):
    for k in range(1, round_robins + 1, 1):
      name = str(i) + '-' + str(j) + '-' + str(k)
      names.append(name)
      ipt.append([i, j, k])
      s, fps = read_wav('samples/' + piece_name + '/' + name + '.wav')
      original_len = s.shape[0]
      max = np.max(np.abs(s))
      F = rfft(s) * 2 / original_len
      local_max_freq = int(round(20000 / fps * original_len))
      F = F[:local_max_freq]
      n = F.shape[0]
      F_real = F.real
      F_imag = F.imag
      tgt.append(np.hstack([F_real, F_imag]))


tgt = np.array(tgt)
M = np.max(np.abs(tgt))
tgt = tgt / M
# tgt = np.cbrt(tgt)


print('Max:',np.max(tgt), 'Min:', np.min(tgt))


ipt = np.array(ipt)
ipt, maxs_ipt, mins_ipt = normalize_cols(ipt,inf_norm,sup_norm)


print('samples x timesteps x features:', ipt.shape, tgt.shape)

# print(ipt)

# print(xipt)

# exit()

# |> Keras Model
opt = K.optimizers.nadam()

input = K.layers.Input(batch_shape=(None, ipt.shape[1])) # 0.0098

# layer = K.layers.GaussianNoise(0.5)(input)
# layer = K.layers.GaussianDropout(0.1)(input)
# layer = K.layers.Dropout(0.1)(input)

# layer = K.layers.Dense(
#   units=3,
#   activation=scaled_tanh,
# )(input)

# layer = K.layers.Dense(
#   units=2,
#   activation=scaled_tanh,
# )(layer)

layer = K.layers.Dense(
  units=2,
  activation=None,
)(input)


output = K.layers.Dense(
  tgt.shape[1],
  activation=None
)(layer)


model = K.models.Model(input, output)
K.utils.print_summary(model)
model.compile(loss='mean_squared_error', optimizer=opt)

tb = K.callbacks.TensorBoard(path)
history = model.fit(
  ipt, tgt, 
  batch_size=tgt.shape[0],
  epochs=epcs, 
  verbose=1, 
  callbacks=[tb],
  validation_split=0.0,
  shuffle=True
  )

save_model(model, path, 'model')


predictions = model.predict(xipt) * M
# predictions = np.power(predictions, 3)

N = predictions.shape[1] // 2

# d = (- np.log(0.1)) / original_len
# X = np.arange(0, original_len, 1)
# D = np.exp(-d * X)
for i, p in enumerate(predictions):
  real = p[ : N]
  imag = p[N : ]
  w = irfft(real + imag * 1j, original_len) / 2 * original_len
  # w = np.multiply(D, w)
  save_wav(w, path + xnames[i] + '.wav')