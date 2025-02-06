import os
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import write

from Kokoro.kokoro import generate
from Kokoro.models import build_model

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device:{device}")
# Load the model
MODEL = build_model("/home/dev/Documents/audiobook/models/kokoro-v0_19.pth", device)


# Read the text to synthesize from a file
input_dir = Path(
    "/home/dev/Documents/audiobook/extracted_pdf/houn"
)  # Path to your text file
audio_dir = "audio"
voice_name = "af_sarah"
audio_path = os.path.join(input_dir, audio_dir)
os.makedirs(audio_path, exist_ok=True)
files = [file for file in input_dir.iterdir() if file.is_file()]
print(f"Loaded voice: {voice_name}")
for i, file in enumerate(files):
    print(f"Progress: {i + 1}/{len(files)}")
    with open(file, "r") as f:
        text = f.read()

    # # Loop through each voice and generate audio
    if not os.path.exists(os.path.join(input_dir, audio_dir, f"{file.stem}.wav")):
        try:
            # Load the voicepack
            voicepack_path = f"Kokoro/voices/{voice_name}.pt"
            VOICEPACK = torch.load(voicepack_path, weights_only=True).to(device)

            # Generate audio
            audio_chunks, phoneme_chunks = generate(
                MODEL, text, VOICEPACK, lang=voice_name[0]
            )

            # Combine and save the normalized audio
            combined_audio = np.concatenate(audio_chunks)
            normalized_audio = (
                combined_audio / np.max(np.abs(combined_audio)) * 32767
            ).astype("int16")
            output_path = os.path.join(input_dir, audio_dir, f"{file.stem}.wav")
            write(output_path, 24000, normalized_audio)
            print(f"Audio saved to {output_path}")

            # Debug: Print audio stats
            print(f"Audio stats for {voice_name}:")
            print(
                "  Audio waveform preview:", combined_audio[:10]
            )  # Show the first 10 samples
            print(
                "  Max amplitude:",
                max(combined_audio),
                "Min amplitude:",
                min(combined_audio),
            )
            print("  Audio data type:", combined_audio.dtype)

        except Exception as e:
            print(f"Failed to generate audio for {voice_name}: {e}")
    else:
        print(f"{file.stem} is already converted and saved.....!")
