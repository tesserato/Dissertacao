import wave
import numpy as np


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

f = 10000
p = 0
a = 5000
d = 0.01


w = create_signal(500, 44100, f, p, a, d)
save_wav(w, str(f) + 'hz_' + str(d) + 'd.wav')