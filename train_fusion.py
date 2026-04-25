"""
train_fusion.py — Late fusion : logits audio (DualBackbone) + logits texte (BERT fine-tuné)
Stratégie : chaque modèle produit ses propres logits → petit réseau de fusion apprend à les combiner.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor

# ─── Config ───────────────────────────────────────────────────────────────────

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 8
EPOCHS      = 10
LR          = 1e-4
SAMPLE_RATE = 16_000
MAX_LENGTH  = SAMPLE_RATE * 4  # 4 secondes

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprised"]
LABEL2ID = {e: i for i, e in enumerate(EMOTION_LABELS)}

# ─── Modèle audio (identique à train_audio.py) ───────────────────────────────

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        weights = torch.softmax(self.attn(hidden_states), dim=1)
        return (hidden_states * weights).sum(dim=1)


class DualBackboneModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.wav2vec2    = AutoModel.from_pretrained("facebook/wav2vec2-base")
        self.hubert      = AutoModel.from_pretrained("facebook/hubert-base-ls960")
        hidden_size      = 768
        self.pooling_w2v = AttentivePooling(hidden_size)
        self.pooling_hub = AttentivePooling(hidden_size)
        self.classifier  = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values):
        out_w2v    = self.wav2vec2(input_values).last_hidden_state
        out_hub    = self.hubert(input_values).last_hidden_state
        pooled_w2v = self.pooling_w2v(out_w2v)
        pooled_hub = self.pooling_hub(out_hub)
        fused      = torch.cat([pooled_w2v, pooled_hub], dim=-1)
        return self.classifier(fused)  # (B, 7) logits


# ─── Modèle BERT complet (backbone + tête de classification) ─────────────────

class BertClassifier(nn.Module):
    """
    Reproduit BertForSequenceClassification :
      pooler_output = tanh(Linear(CLS))  →  classifier
    Le checkpoint d'Asma a été entraîné avec cette même structure.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.bert       = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None, **kwargs):
        out    = self.bert(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        pooled = out.pooler_output      # (B, 768) — Linear(768,768) + tanh sur le CLS
        return self.classifier(pooled)  # (B, 7) logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FusionDataset(Dataset):
    def __init__(self, df, audio_extractor, data_root=""):
        self.df              = df.reset_index(drop=True)
        self.audio_extractor = audio_extractor
        self.data_root       = data_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        path  = os.path.join(self.data_root, row["path"]) if self.data_root else row["path"]
        label = LABEL2ID[row["emotion"]]
        text  = str(row["text"])

        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        waveform = waveform.squeeze(0).numpy()

        if len(waveform) > MAX_LENGTH:
            waveform = waveform[:MAX_LENGTH]
        else:
            waveform = np.pad(waveform, (0, MAX_LENGTH - len(waveform)))

        inputs = self.audio_extractor(
            waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=False
        )
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "text":         text,
            "label":        torch.tensor(label, dtype=torch.long),
        }


# ─── Modèle de fusion (late fusion sur probabilités) ─────────────────────────

class FusionModel(nn.Module):
    """
    Prend softmax(audio_logits) + softmax(text_logits) → 14 valeurs
    Un petit MLP apprend à pondérer les deux modèles par émotion.
    """
    def __init__(self, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, audio_logits, text_logits):
        audio_probs = torch.softmax(audio_logits, dim=-1)
        text_probs  = torch.softmax(text_logits,  dim=-1)
        x = torch.cat([audio_probs, text_probs], dim=-1)  # (B, 14)
        return self.net(x)                                  # (B, 7)


# ─── Extraction des logits (backbones gelés) ─────────────────────────────────

@torch.no_grad()
def extract_features(batch, audio_backbone, bert_model, tokenizer, device):
    input_values = batch["input_values"].to(device)
    texts        = batch["text"]

    audio_logits = audio_backbone(input_values)  # (B, 7)

    tok = tokenizer(list(texts), return_tensors="pt",
                    padding=True, truncation=True, max_length=128)
    tok = {k: v.to(device) for k, v in tok.items()}
    text_logits = bert_model(**tok)              # (B, 7)

    return audio_logits, text_logits


# ─── Boucle train / eval ──────────────────────────────────────────────────────

def run_epoch(model, loader, audio_backbone, bert_model, tokenizer,
              criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    for batch in loader:
        labels = batch["label"].to(device)
        audio_logits, text_logits = extract_features(
            batch, audio_backbone, bert_model, tokenizer, device
        )

        if train:
            optimizer.zero_grad()

        logits = model(audio_logits, text_logits)
        loss   = criterion(logits, labels)

        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), correct / total, f1, all_preds, all_labels


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device : {DEVICE}")

    # ── Données ──────────────────────────────────────────────────────────────
    # Labels corrects depuis full_dataset.csv, texte depuis full_dataset_with_text.csv
    audio_df = pd.read_csv("data/processed/full_dataset.csv")
    text_df  = pd.read_csv("data/processed/transcriptions/full_dataset_with_text.csv")[["path", "text"]]
    df = audio_df.merge(text_df, on="path")
    df["emotion"] = df["emotion"].replace({"fearful": "fear"})
    df = df[df["emotion"].isin(EMOTION_LABELS)].reset_index(drop=True)

    splits_dir = Path("data/splits_fusion")
    splits_dir.mkdir(parents=True, exist_ok=True)

    if (splits_dir / "train.csv").exists():
        train_df = pd.read_csv(splits_dir / "train.csv")
        val_df   = pd.read_csv(splits_dir / "val.csv")
        test_df  = pd.read_csv(splits_dir / "test.csv")
        print("Splits chargés depuis data/splits_fusion/")
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.20, random_state=42, stratify=df["emotion"]
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, random_state=42, stratify=temp_df["emotion"]
        )
        train_df.to_csv(splits_dir / "train.csv", index=False)
        val_df.to_csv(  splits_dir / "val.csv",   index=False)
        test_df.to_csv( splits_dir / "test.csv",  index=False)
        print("Splits créés dans data/splits_fusion/")

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Modèle audio ─────────────────────────────────────────────────────────
    print("Chargement modèle audio...")
    audio_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    audio_backbone  = DualBackboneModel(num_classes=len(EMOTION_LABELS))
    audio_backbone.load_state_dict(
        torch.load("checkpoints/audio/both/best_model.pt",
                   map_location=DEVICE, weights_only=False)
    )
    audio_backbone = audio_backbone.to(DEVICE).eval()
    for p in audio_backbone.parameters():
        p.requires_grad = False
    print("Modèle audio chargé.")

    # ── Modèle BERT (backbone + tête de classification) ──────────────────────
    print("Chargement modèle BERT...")
    tokenizer  = AutoTokenizer.from_pretrained("checkpoints/texte/bert_finetuned")
    bert_model = BertClassifier(num_classes=len(EMOTION_LABELS))
    checkpoint = torch.load("checkpoints/texte/bert_finetuned/best_model.pt",
                             map_location=DEVICE, weights_only=False)
    bert_sd = checkpoint["model_state_dict"]
    result  = bert_model.load_state_dict(bert_sd, strict=False)
    if result.missing_keys:
        print(f"  Clés manquantes  : {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Clés inattendues : {[k for k in result.unexpected_keys if not k.startswith('cls.')]}")
    bert_model = bert_model.to(DEVICE).eval()
    for p in bert_model.parameters():
        p.requires_grad = False
    print("Modèle BERT chargé.")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = FusionDataset(train_df, audio_extractor)
    val_ds   = FusionDataset(val_df,   audio_extractor)
    test_ds  = FusionDataset(test_df,  audio_extractor)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Diagnostic : vérification des logits avant fusion ────────────────────
    print("\n=== Diagnostic ===")
    diag_batch = next(iter(val_loader))
    with torch.no_grad():
        al, tl = extract_features(diag_batch, audio_backbone, bert_model, tokenizer, DEVICE)
        lab    = diag_batch["label"]
        print(f"Audio acc (1 batch) : {(al.argmax(-1).cpu() == lab).float().mean():.3f}")
        print(f"Texte acc (1 batch) : {(tl.argmax(-1).cpu() == lab).float().mean():.3f}")
    print("=================\n")

    # ── Fusion model ──────────────────────────────────────────────────────────
    model     = FusionModel(num_classes=len(EMOTION_LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    output_dir = Path("checkpoints/fusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}

    # ── Entraînement ──────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, _, _, _ = run_epoch(
            model, train_loader, audio_backbone, bert_model,
            tokenizer, criterion, optimizer, DEVICE, train=True
        )
        with torch.no_grad():
            val_loss, val_acc, val_f1, _, _ = run_epoch(
                model, val_loader, audio_backbone, bert_model,
                tokenizer, criterion, optimizer, DEVICE, train=False
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f} acc: {val_acc:.4f} f1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  → Meilleur modèle sauvegardé (f1={best_f1:.4f})")

    # ── Évaluation finale ────────────────────────────────────────────────────
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    with torch.no_grad():
        _, test_acc, test_f1, preds, labels = run_epoch(
            model, test_loader, audio_backbone, bert_model,
            tokenizer, criterion, optimizer, DEVICE, train=False
        )

    report = classification_report(labels, preds, target_names=EMOTION_LABELS, zero_division=0)
    cm     = confusion_matrix(labels, preds).tolist()

    print(f"\nTest — Accuracy: {test_acc:.4f} | F1 macro: {test_f1:.4f}")
    print(report)

    results = {
        "test_acc": test_acc, "test_f1": test_f1,
        "report": report, "cm": cm, "history": history,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résultats sauvegardés dans {output_dir}/results.json")


if __name__ == "__main__":
    main()
