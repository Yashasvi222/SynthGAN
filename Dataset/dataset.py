from music21 import *
from midi2audio import FluidSynth
from pydub import AudioSegment


def collect_notes(notes_dict, path):
    for key, value in notes_dict.items():
        n = note.Note(f"{key}")
        n.write('midi', fp=f'{path}\\{value}.mid')


def midi_to_wav(input_midi, output_wav):
    fs = FluidSynth()
    fs.midi_to_audio(input_midi, output_wav)


def trim_wav(input_wav, output_wav, start_ms, end_ms):

    audio = AudioSegment.from_wav(input_wav)
    trimmed_audio = audio[start_ms:end_ms]
    trimmed_audio.export(output_wav, format="wav")


if __name__ == '__main__':
    notes = ["c5", "c#5", "d-5", "d5", "d#5", "e-5", "e5", "f5", "f#5", "f-5", "g5", "g#5", "a-5", "a5", "a#5", "b-5",
             "b5"]
    notes_names = ["c5_piano", "c5_sharp_piano", "d5_flat_piano", "d5_piano", "d5_sharp_piano", "e5_flat_piano",
                   "e5_piano",
                   "f5_piano", "f5_sharp_piano", "f5_flat_piano", "g5_piano", "g5_sharp_piano", "a5_flat_piano",
                   "a5_piano",
                   "a5_sharp_piano", "b5_flat_piano", "b5_piano"]

    notes_dict = dict(zip(notes, notes_names))

    path = 'C:\\Users\\yasha\\OneDrive\\Desktop\\music21dataset'

    collect_notes(notes_dict, path)
