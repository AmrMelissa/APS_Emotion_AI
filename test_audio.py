import torch, torchaudio, numpy as np
from transformers import AutoModel, AutoFeatureExtractor
import torch.nn as nn
from pathlib import Path

class AttentivePooling(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.attn = nn.Linear(h, 1)
    def forward(self, x):
        w = torch.softmax(self.attn(x), dim=1)
        return (x * w).sum(dim=1)

class DualBackboneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2    = AutoModel.from_pretrained("facebook/wav2vec2-base")
        self.hubert      = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        self.pooling_w2v = AttentivePooling(768)
        self.pooling_hub = AttentivePooling(768)
        self.classifier  = nn.Sequential(nn.Linear(1536,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256,7))
    def forward(self, x):
        w = self.pooling_w2v(self.wav2vec2(x).last_hidden_state)
        h = self.pooling_hub(self.hubert(x).last_hidden_state)
        return self.classifier(torch.cat([w,h], dim=-1))

LABELS = ["neutral","happy","sad","angry","fear","disgust","surprised"]

MODEL_PATH = "checkpoints/both/best_model.pt"
AUDIO_PATH = "data/processed/audio/sample_0.wav"

device = "cpu"
model = DualBackboneModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model = model.to(device).eval()
extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

waveform, sr = torchaudio.load(AUDIO_PATH)
if waveform.shape[0] > 1: waveform = waveform.mean(0, keepdim=True)
if sr != 16000: waveform = torchaudio.functional.resample(waveform, sr, 16000)
waveform = waveform.squeeze(0).numpy()
waveform = waveform[:64000] if len(waveform) > 64000 else np.pad(waveform, (0, 64000 - len(waveform)))

inp = extractor(waveform, sampling_rate=16000, return_tensors="pt").input_values
with torch.no_grad():
    logits = model(inp)
    probs  = torch.softmax(logits, dim=-1)[0]
    pred   = logits.argmax(-1).item()

print(f"Émotion prédite : {LABELS[pred]}")
for label, prob in zip(LABELS, probs):
    print(f"  {label:12s} : {prob:.3f}")
