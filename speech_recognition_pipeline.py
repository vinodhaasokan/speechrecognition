
import os
import librosa
import torch
import torchaudio
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from jiwer import wer
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 16000
N_MELS = 80
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpeechDataset(Dataset):
    def __init__(self, audio_paths, transcripts):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        path = self.audio_paths[idx]
        transcript = self.transcripts[idx]
        waveform, _ = torchaudio.load(path)
        mel = self.processor(waveform).squeeze(0).transpose(0, 1)
        return mel, transcript

class SpeechModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(SpeechModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x)

class SimpleTokenizer:
    def __init__(self):
        self.chars = list("abcdefghijklmnopqrstuvwxyz '")
        self.char2idx = {ch: i + 1 for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}

    def encode(self, text):
        return [self.char2idx[ch] for ch in text.lower() if ch in self.char2idx]

    def decode(self, indices):
        return ''.join([self.idx2char.get(i, '') for i in indices])

tokenizer = SimpleTokenizer()

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lengths = [x.shape[0] for x in inputs]
    target_lengths = [len(tokenizer.encode(y)) for y in targets]
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_encoded = [torch.tensor(tokenizer.encode(y)) for y in targets]
    targets_padded = nn.utils.rnn.pad_sequence(targets_encoded, batch_first=True)
    return inputs_padded, targets_padded, input_lengths, target_lengths

def train(model, dataloader, criterion, optimizer):
    model.train()
    for inputs, targets, input_lengths, target_lengths in dataloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        output = model(inputs)
        output = output.permute(1, 0, 2)
        log_probs = nn.functional.log_softmax(output, dim=2)
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, dataloader):
    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in dataloader:
            inputs = inputs.to(DEVICE)
            output = model(inputs)
            pred = output.argmax(dim=2).cpu().numpy()
            for p, t in zip(pred, targets):
                decoded = tokenizer.decode(p)
                target_text = tokenizer.decode(t.tolist())
                predictions.append(decoded.strip())
                references.append(target_text.strip())
    return wer(references, predictions)

def load_data(audio_dir):
    audio_paths = []
    transcripts = []
    for fname in os.listdir(audio_dir):
        if fname.endswith('.wav'):
            base = os.path.splitext(fname)[0]
            txt_path = os.path.join(audio_dir, base + ".txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    transcript = f.read().strip()
                audio_paths.append(os.path.join(audio_dir, fname))
                transcripts.append(transcript)
    return audio_paths, transcripts

def main():
    audio_dir = "data/"
    audio_paths, transcripts = load_data(audio_dir)
    train_paths, test_paths, train_transcripts, test_transcripts = train_test_split(
        audio_paths, transcripts, test_size=0.2
    )
    train_set = SpeechDataset(train_paths, train_transcripts)
    test_set = SpeechDataset(test_paths, test_transcripts)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    model = SpeechModel(N_MELS, 128, len(tokenizer.char2idx) + 1).to(DEVICE)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(EPOCHS):
        train(model, train_loader, criterion, optimizer)
        wer_score = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}, WER: {wer_score:.4f}")

if __name__ == "__main__":
    main()
