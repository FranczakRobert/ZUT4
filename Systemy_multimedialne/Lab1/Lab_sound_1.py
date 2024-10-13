import numpy as np
import matplotlib.pyplot as plt
import scipy
import sounddevice as sd
import soundfile as sf

# wczytywanie pliku dźwiekowego
data, fs = sf.read('SOUND_INTRO/sound1.wav', dtype='float32')
 #FS  ----> Drugi parametr to częstotliwość próbkowania
#data ----> tablica zawierająca wartości próbek sygnału,

print(data.dtype)
print(data.shape)


#odpowiada za rozpoczęcie odtwarzania.
# sd.play(data, fs)

#zapewnia, że program poczeka z wykonaniem kolejnych
#instrukcji do momentu, aż dźwięk skończy się odtwarzać.
# status = sd.wait()


# -------------- Zadanie 1 ---------------------
graj_to = False

data, fs = sf.read('SOUND_INTRO/sound1.wav', dtype='float32')

kanaly = []
for i in range(data.shape[1]):
    kanaly.append(data[:, i])
    sf.write("kanal" + str(i) + ".wav", kanaly[i], fs)

if(graj_to):
    for i in range(len(kanaly)):
        data_kanal_i, fs_i = sf.read("kanal" + str(i) + ".wav", dtype='float32')
        sd.play(data_kanal_i, fs_i)
        status = sd.wait()

# plt.subplot(2,1,1)
# plt.plot(data[:,0])

# Wyswietl kanały
time = np.arange(0, len(data)) / fs

for i in range(data.shape[1]):
    # Wykres dla kanału 1
    plt.subplot(2, 1, i + 1)
    if (i == 0):
        plt.plot(time, data[:, i], color='blue')

    if(i == 1):
        plt.plot(time, data[:, i], color='red')
    plt.title("Kanal" + str(i + 1))
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')

# Wyświetlenie wykresów
plt.tight_layout()
plt.show()

# WIDMO
data, fs = sf.read('SIN/sin_440Hz.wav', dtype=np.int32)
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(0,data.shape[0])/fs,data)

plt.subplot(2,1,2)
yf = scipy.fftpack.fft(data)
plt.plot(np.arange(0,fs,1.0*fs/(yf.size)),np.abs(yf))
plt.show()
