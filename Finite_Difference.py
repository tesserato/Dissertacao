import numpy as np
import matplotlib.pyplot as plt
import os
from Hlib import save_wav

path = 'Demo Finite Difference/'
os.makedirs(path, exist_ok=True)



amplitude = 0.005 # meters
pluck_position = .3
pickup_position = .7

fps = 44100
frequency = 440
duration = 1 # seconds
L = 0.6 # meters

N = int(duration * fps)# number of time points
dt = 1 / fps
M = int(fps / (2 * frequency))# number of position points
dx = L / M
c = frequency * 2 * L # meters / second
C = c * dt / dx


x = np.arange(0, M + 1, 1) * dx
t = np.arange(0, N + 1, 1) * dt

print(C, M, N)

F = lambda X: 3 * X**2 - 2 * X**3

pickup = int(round(M * pickup_position))
asc = int(round(M * pluck_position))
dsc = M + 2 - asc
X_asc = np.linspace(0, 1, asc)
X_dsc = np.linspace(0, 1, dsc)[::-1][1: ]

y = np.zeros(M + 1)
y_1 = np.hstack([X_asc,X_dsc]) * amplitude * np.random.normal(1, .01, M + 1) # initial shape
y_2 = np.zeros(M + 1)


# plt.plot(y_1)
# plt.show()
# exit()
plot = False
ctr=0
w = []
for i in range(1, M):
  y[i] = y_1[i] + 0.5 * C**2 *(y_1[i+1] - 2*y_1[i] + y_1[i-1])
y[0] = 0
y[M] = 0
w.append(y[pickup])
y_2[:] = y_1
y_1[:] = y

if plot:
  plt.plot(x, y, 'k')
  plt.axis([0, L, -1.1 * amplitude, 1.1 * amplitude])
  plt.axvline(L * pickup_position)
  plt.savefig(path + str(ctr) + '.png')
  plt.close()
ctr += 1


for j in range(1, N):
  print(ctr)
  for i in range(1, M):# Update all inner mesh points at time t[n+1]
    y[i] = 2 * y_1[i] - y_2[i] + C**2 * (y_1[i+1] - 2*y_1[i] + y_1[i-1])

  # Insert boundary conditions
  y[0] = 0
  y[M] = 0
  w.append(y[pickup])
  y_2[:] = y_1 
  y_1[:] = y

  if plot:
    loop = 0
    plt.plot(x, y, 'k')
    plt.axis([0, L, -1.1 * amplitude, 1.1 * amplitude])
    plt.axvline(L * pickup_position)
    plt.savefig(path + str(ctr) + '.png')
    plt.close()
  ctr += 1

w = np.array(w) * 20000

save_wav(
  w,
  path + str(frequency) + 
  '_Pluck=' + str(pluck_position) + 
  '_Pick=' + str(pickup_position) + '.wav'
  )


# plot = True
# sustain = .99
# smoothing = 3
# w = np.zeros(n)
# zeros = np.zeros(L)
# for i in range(n):
#   print('step ', i + 1,' of ', n)

#   if plot:
#     plt.subplot(311)
#     plt.axis([0, L, -1.1, 1.1])
#     plt.plot(delay_r,'.')
#     plt.plot(zeros,'k')    
#     plt.subplot(312)
#     plt.axis([0, L, -1.1, 1.1])
#     plt.plot(delay_l,'.')
#     plt.plot(zeros,'k')
#     plt.subplot(313)
#     plt.axis([0, L, -1.1, 1.1])
#     plt.plot(delay_r + delay_l,'k')
#     plt.axvline(pickup)
#     plt.savefig('Demo Digital Waveguide/' + str(i) + '.png')
#     plt.close()


#   w[i] = delay_r[pickup] + delay_l[pickup]
#   to_l = -1 * np.average(delay_r[-smoothing:]) * sustain # to add in delay_l[L-1] AFTER rolling
#   delay_r = np.roll(delay_r, 1)

#   delay_r[0] = -1 * delay_l[0]

#   delay_l = np.roll(delay_l, -1)
#   delay_l[L-1] = to_l



# save_wav(
#   w * amplitude,
#   'Demo Digital Waveguide/#F=' + str(frequency) + 
#   '_Pluck=' + str(pluck_position) + 
#   '_Pick=' + str(pickup_position) + '.wav'
#   )

