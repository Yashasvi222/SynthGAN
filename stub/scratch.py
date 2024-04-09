"""
We have WAV files that are to be loaded in librosa. Some kind of dependency issue/malware issue is present.
1. WE ARE ASSUMING THAT THE FILES ARE CLEAN.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import wave

import glob


directory = 'C:\\Users\\yasha\\OneDrive\\Desktop\\music21\\music21datasetwav\\*.wav'
wav_files = glob.glob(directory)
for fp in wav_files:
    # Load the WAV file
    wav_file = wave.open(fp, 'r')
    signal = wav_file.readframes(-1)
    signal = np.frombuffer(signal, dtype=np.int16)
    sampling_rate = wav_file.getframerate()

    # Define parameters
    window_size = 0.025  # Window size in seconds
    hop_size = 0.010  # Hop size in seconds

    # Convert window size and hop size from seconds to samples
    window_size_samples = int(window_size * sampling_rate)
    hop_size_samples = int(hop_size * sampling_rate)

    # Perform STFT
    frequencies, times, Zxx = stft(signal, fs=sampling_rate, window='hamming', nperseg=window_size_samples, noverlap=hop_size_samples)

    # Plot the magnitude spectrogram
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title('STFT Magnitude Spectrogram')
    plt.colorbar(label='Magnitude')
    plt.tight_layout()
    plt.show()



# import scipy.io.wavfile as wav
# from scipy import signal  # This is the issue
# import numpy as np
# import matplotlib.pyplot as plt
#
# import glob
#
# dset = []
# directory = 'C:\\Users\\yasha\\OneDrive\\Desktop\\music21\\music21datasetwav\\*.wav'
# wav_files = glob.glob(directory)
# for fp in wav_files:
#     (rate, sig) = wav.read(fp)
#     dset.append((rate, sig))
#     print(f"""
# PATH: {fp}
# SIG : {sig}
# RATE: {rate}
#     """)
#
#
# for file in dset:
#     rate, sig = file
#
#     rate, sig = wav.read(fp)
#     frequencies, times, spectrogram = signal.spectrogram(sig, rate)
#
#     # Plot the spectrogram
#     plt.figure(figsize=(10, 5))
#     plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram), shading='gouraud')  # Convert to dB
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [s]')
#     plt.title('Spectrogram of the audio signal')
#     plt.colorbar(label='Intensity [dB]')
#     plt.tight_layout()
#     plt.show()

    # # Define parameters
    # window_size = 1024  # Size of the FFT window
    # overlap = 512  # Overlap between consecutive windows
    #
    # # Calculate spectrogram using FFT
    # n_fft = window_size
    # hop_length = window_size - overlap
    # spectrogram = np.abs(
    #     np.array([np.fft.fft(sig[i:i + window_size], n=n_fft) for i in range(0, len(sig) - window_size, hop_length)]))
    #
    # # Calculate frequencies and times
    # frequencies = np.fft.fftfreq(window_size, d=1 / rate)[:n_fft // 2]
    # times = np.arange(0, len(sig) / rate - window_size / rate, hop_length / rate)
    #
    # # Plot the spectrogram
    # plt.figure(figsize=(10, 5))
    # plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram[:, :n_fft // 2].T), shading='gouraud')  # Convert to dB
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [s]')
    # plt.title('Spectrogram of the audio signal')
    # plt.colorbar(label='Intensity [dB]')
    # plt.tight_layout()
    # plt.show()

