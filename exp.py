from pydub import AudioSegment
import librosa
import scipy
import scipy.stats


def trim_wav(input_wav, output_wav, start_ms, end_ms):
    audio = AudioSegment.from_wav(input_wav)
    trimmed_audio = audio[start_ms:end_ms]
    trimmed_audio.export(output_wav, format="wav")


def trim_wav_librosa(input_wav, output_wav, start_sec, end_sec):
    # Load the audio file
    y, sr = librosa.load(input_wav, sr=None)

    # Convert start and end times to sample indices
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)

    # Trim the audio
    trimmed_audio = y[start_sample:end_sample]

    # Save the trimmed audio
    librosa.output.write_wav(output_wav, trimmed_audio, sr)


inpath = "C:\\Users\\yasha\\OneDrive\\Desktop\\music21\\music21datasetwav\\a5_flat_piano.wav"
outpath = "C:\\Users\\yasha\\OneDrive\\Desktop\\music21\\music21datasetwav\\a5_flat_piano_trimmed.wav"
start = 0
end = 1
trim_wav_librosa(inpath, outpath, start, end)
