import numpy as np
import keras as K
import matplotlib.pyplot as plt
# from Hlib import save_wav
from Hlib import save_model
from Hlib import normalize_cols, denormalize_cols
from Hlib import create_signal
import os

def scaled_tanh(x):
  return K.activations.tanh(6 * x - 3) / 2 + 1 / 2


piece_name = 'piano'
path = '02_predictions/' + piece_name + '/'
epcs = 10000


os.makedirs(path, exist_ok=True)
os.makedirs(path + '/amplitudes', exist_ok=True)

filenames = os.listdir('00_samples/' + piece_name + '/')
names = []
for filename in filenames:
  names.append(filename.replace('.wav', ''))
names = np.sort(np.array(names, dtype=np.int)).astype(np.str)

# preparing targets
amplitudes = np.genfromtxt('01_info/' + piece_name + '/fractions_of_max_amps.csv', delimiter=',')
partials = amplitudes.shape[1]
tgt = np.reshape(amplitudes, (-1, 1))

# preparing inputs
ipt = []
for name in names:
  key = (float(name) - 1) / 87
  for p in np.linspace(0, 1, partials):
    ipt.append([key, p])
ipt = np.array(ipt)

# removing zeros from tgt, and the respective inputs
nonzero_idxs = np.where(tgt > 0)
ipt = ipt[nonzero_idxs[0], :]
tgt = tgt[nonzero_idxs[0], :]

M = np.max(tgt)
m = np.min(tgt)
tgt = (tgt - m) / (M - m)

# preparing complete inputs
complete_ipt = []
for name in range(1, 88 + 1, 1):
  key = (float(name) - 1) / 87
  for p in np.linspace(0, 1, partials):
    complete_ipt.append([key, p])
complete_ipt = np.array(complete_ipt)


print(ipt.shape, tgt.shape)

act = scaled_tanh
opt = K.optimizers.nadam()
init = 'lecun_normal'   #BESTFIT: 0.0048

input = K.layers.Input(batch_shape=(None, ipt.shape[1]))

layer = K.layers.Dense(
  units = 50,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init
)(input)

layer = K.layers.Dense(
  units = 50,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init
)(layer)

layer = K.layers.Dense(
  units = 50,
  activation=act,
  kernel_initializer=init,
  bias_initializer=init
)(layer)


output = K.layers.Dense(
  units = tgt.shape[1],
  activation=act,
  kernel_initializer=init,
  bias_initializer=init
)(layer)

model = K.models.Model(input, output)
K.utils.print_summary(model)
model.compile(loss='mean_squared_error', optimizer=opt)

tb = K.callbacks.TensorBoard(path + '/amplitudes')
history = model.fit(ipt, tgt, batch_size=ipt.shape[0], epochs=epcs, verbose=1, callbacks=[tb], shuffle=True)

save_model(model, path + '/amplitudes/', 'model')



pred = np.reshape(model.predict(complete_ipt), (-1, partials)) * (M - m) + m

print(m, M)

X = np.reshape(complete_ipt[:, 1], (-1, partials))
plt.plot(X, amplitudes ,'ko')
plt.plot(X, pred ,'r.')
plt.savefig(path + 'fractions_of_max_amps.png')
plt.close()

np.savetxt(path + 'fractions_of_max_amps.csv', pred, delimiter=',')
