
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import os
import glob
import argparse
import numpy as np
from scipy.signal import butter, lfilter
from jiwer import wer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

# Noise filtering: Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')
    return b, a

def bandpass_filter(data, lowcut=100.0, highcut=3000.0, fs=16000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)

# Preprocess: normalize and filter
def preprocess(waveform, sample_rate):
    waveform = waveform.numpy().squeeze()
    waveform = bandpass_filter(waveform, fs=sample_rate)
    waveform = torch.tensor(waveform).unsqueeze(0)
    return waveform

# Transcribe one audio file
def transcribe(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = preprocess(waveform, sample_rate)
    input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(pred_ids[0])
    return transcription.lower()

# Batch transcription with WER
def transcribe_folder(audio_dir, transcript_file=None):
    files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
    ground_truths = open(transcript_file).readlines() if transcript_file else None
    total_wer = 0
    for i, file in enumerate(files):
        prediction = transcribe(file)
        print(f"[{os.path.basename(file)}] -> {prediction}")
        if ground_truths:
            ref = ground_truths[i].strip().lower()
            err = wer(ref, prediction)
            total_wer += err
            print(f"   REF: {ref}")
            print(f"   WER: {err:.2f}")
    if ground_truths:
        print(f"\nAverage WER: {total_wer / len(files):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", required=True, help="Directory with .wav files")
    parser.add_argument("--transcripts", help="Optional text file with reference transcriptions")
    args = parser.parse_args()
    transcribe_folder(args.audio_dir, args.transcripts)
