import glob
import os

from pydub import AudioSegment

temp_dir = "/home/dev/Documents/audiobook/extracted_pdf/HPI_Methodology-June-2022-rev-ENG/audio/"

wav_files = sorted(glob.glob(os.path.join(temp_dir, "*.wav")))
print(wav_files)

combined = AudioSegment.empty()

for wav_file in wav_files:
    audio = AudioSegment.from_wav(wav_file)
    combined += audio
output_file = os.path.join(temp_dir, "audio.wav")
combined.export(output_file, format="wav")

print(f"Combined file is saved as: {output_file}")
