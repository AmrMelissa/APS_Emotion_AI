import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from transformers import AutoTokenizer
import os

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4

# =========================
# LABELS
# =========================
label_map = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6
}

NUM_CLASSES = len(label_map)

# =========================
# DATASET
# =========================
class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        path = row["path"].replace("\\", "/")
        label = label_map[row["emotion"]]
        text = row["text"]

        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio not found: {path}")

        waveform, sr = torchaudio.load(path)

        return waveform, text, label


# =========================
# LOAD DATA
# =========================
dataset = EmotionDataset("data/processed/full_dataset_with_text.csv")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =========================
# LOAD MODELS
# =========================
print("=== Loading models ===")

bert_model = torch.load(
    "checkpoints/best_model.pt",
    map_location=DEVICE
)

audio_model = torch.load(
    "checkpoints/both/best_model.pt",
    map_location=DEVICE
)

bert_model.to(DEVICE)
audio_model.to(DEVICE)

bert_model.eval()
audio_model.eval()

print("=== Models loaded ===")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =========================
# FEATURE EXTRACTION
# =========================
def get_audio_vec(audio):
    with torch.no_grad():
        audio = audio.squeeze(1).to(DEVICE)

        out = audio_model(audio)

        # 🔥 compat HuggingFace
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state

        mean = out.mean(dim=1)
        std = out.std(dim=1)

        return torch.cat([mean, std], dim=1)


def get_text_vec(texts):
    inputs = tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        out = bert_model(**inputs)

        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state.mean(dim=1)
        else:
            return out


# =========================
# MODELS
# =========================
class AttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, audio_vec, text_vec):
        Q = self.query(audio_vec)
        K = self.key(text_vec)
        V = self.value(text_vec)

        attn = torch.softmax(Q @ K.T, dim=-1)
        return attn @ V


class FusionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.proj_audio = nn.Linear(input_dim * 2, input_dim)

        self.attention = AttentionFusion(input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, audio_vec, text_vec):
        audio_vec = self.proj_audio(audio_vec)
        fusion = self.attention(audio_vec, text_vec)
        return self.classifier(fusion)


# =========================
# INIT
# =========================
D = 768
model = FusionModel(D).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# =========================
# TRAIN
# =========================
def train():
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for audio, text, label in loader:
            label = label.to(DEVICE)

            audio_vec = get_audio_vec(audio)
            text_vec = get_text_vec(text)

            logits = model(audio_vec, text_vec)

            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")


# =========================
# EVAL
# =========================
def evaluate():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for audio, text, label in loader:
            label = label.to(DEVICE)

            audio_vec = get_audio_vec(audio)
            text_vec = get_text_vec(text)

            logits = model(audio_vec, text_vec)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == label).sum().item()
            total += label.size(0)

    print(f"Accuracy: {correct/total:.4f}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("=== Test CUDA ===")
    print(torch.cuda.is_available())

    print("=== Start training ===")
    train()

    print("=== Evaluation ===")
    evaluate()

    print("=== END ===")