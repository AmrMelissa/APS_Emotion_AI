# 🎭 Détection d'Émotions Multimodale — Audio + Texte

> Pipeline bimodal combinant **HuBERT / Wav2Vec2** (audio) et **BERT** (texte via Whisper) pour la reconnaissance automatique des émotions.

---

## 👥 Équipe

| Membre | Responsabilité principale |
|--------|--------------------------|
| **Melissa** | Pipeline audio — HuBERT + Wav2Vec2, fine-tuning SER |
| **Asma** | Transcription Whisper, pipeline texte BERT, fine-tuning NLP |
| **Kawther** | Module de fusion, déploiement, intégration temps réel |

---

## 🗂️ Structure du Projet

```
APS_emotion_AI/
│
├── data/
│   ├── raw/
│   │   ├── RAVDESS/          # Corpus audio brut (téléchargé)
│   │   └── CREMA-D/          # Corpus audio brut (téléchargé)
│   ├── processed/
│   │   ├── audio/            # Fichiers audio prétraités (16kHz, normalisés)
│   │   └── transcriptions/   # Transcriptions Whisper (.csv)
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── notebooks/
│   ├── 00_download_data.ipynb        ← Ce notebook
│   ├── 01_preprocessing_audio.ipynb
│   ├── 02_transcription_whisper.ipynb
│   ├── 03_finetune_audio_model.ipynb
│   ├── 04_finetune_bert.ipynb
│   └── 05_fusion_evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── audio_model.py      # HuBERT / Wav2Vec2 + tête de classification
│   │   ├── text_model.py       # BERT + tête de classification
│   │   └── fusion.py           # Late Fusion & Attention-based Fusion
│   ├── training/
│   │   ├── train_audio.py
│   │   ├── train_text.py
│   │   └── train_fusion.py
│   └── inference/
│       ├── predict.py
│       └── realtime.py         # Capture micro + prédiction live
│
├── configs/
│   ├── audio_model.yaml
│   ├── text_model.yaml
│   └── fusion.yaml
│
├── app/
│   ├── api.py                  # FastAPI — endpoint REST
│   └── streamlit_app.py        # Interface démo temps réel
│
├── checkpoints/                # Modèles sauvegardés (.pt)
├── results/                    # Métriques, matrices de confusion
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 🎯 Émotions Cibles

| ID | RAVDESS | CREMA-D | Classe Unifiée |
|----|---------|---------|----------------|
| 1  | Neutral  | Neutral | `neutral`  |
| 2  | Calm     | —       | `calm`     |
| 3  | Happy    | Happy   | `happy`    |
| 4  | Sad      | Sad     | `sad`      |
| 5  | Angry    | Anger   | `angry`    |
| 6  | Fearful  | Fear    | `fearful`  |
| 7  | Disgust  | Disgust | `disgust`  |
| 8  | Surprised| —       | `surprised`|

---

## 🏗️ Architecture

```
#ancinne
Signal Audio Brut
       │
   ┌───┴──────────────────────────────┐
   │                                  │
   ▼                                  ▼
[Prétraitement Audio]          [Whisper (ASR)]
       │                              │
       ▼                              ▼
[HuBERT / Wav2Vec2]            [BERT fine-tuné]
   (fine-tuné)                        │
       │                              │
       ▼                              ▼
  Logits Audio               Logits Texte
  P_a(y|x_audio)             P_t(y|x_texte)
       │                              │
       └──────────┬───────────────────┘
                  ▼
          [Module de Fusion]
       ┌──────────┴──────────┐
       ▼                     ▼
 Late Fusion         Attention Fusion
 (pondération)       (cross-modal)
       └──────────┬──────────┘
                  ▼
         Émotion Prédite 🎭
```

```
#nouvelle             
🎧 Signal Audio Brut
        │
        ▼
[Prétraitement Audio]
        │
        ▼
        ┌───────────────────────────────┬────────────────────────────────┐
        │                               │                                │
        ▼                               ▼                                ▼
🔊 BRANCHE AUDIO                 📝 TRANSCRIPTION                  📝 BRANCHE TEXTE
(Wav2Vec2 / HuBERT)              (Whisper)                         (BERT)
        │                               │                                │
        ▼                               ▼                                ▼
Représentation audio (T, d)     Texte transcrit                  Tokenisation
        │                               │                                │
        ▼                               │                                ▼
⭐ Statistical Pooling                  │                    [BERT fine-tuné (GoEmotions)]
(mean + std)                           │                                │
        │                               │                                ▼
        ▼                               │                    Représentation texte (T, d)
audio_vec (2d)                         │                                │
        │                               │                                ▼
        │                               │                     Pooling (mean ou CLS)
        │                               │                                │
        ▼                               ▼                                ▼
(OPTIONNEL) MLP_audio         Texte → BERT input               text_vec (d)
        │                                                                │
        ▼                                                                ▼
logits_audio = P_a(y|x_audio)                                 (OPTIONNEL) MLP_text
        │                                                                │
        ▼                                                                ▼
                                                    logits_text = P_t(y|x_text)
        │                                                                │
        └───────────────────────────────┬────────────────────────────────┘
                                        ▼
                              🔗 MODULE DE FUSION
                                        │
                ┌───────────────────────┴────────────────────────┐
                ▼                                                ▼
        🟢 Late Fusion                                    🔵 Feature Fusion
   (fusion des probabilités)                       (fusion des représentations)
                │                                                │
P_final = αP_audio + (1-α)P_text              fusion_vec = concat(audio_vec, text_vec)
                │                                                │
                ▼                                                ▼
        Émotion finale 🎭                          ⭐ MLP (classification finale)
                                                          │
                                                          ▼
                                                   logits_final
                                                          │
                                                          ▼
                                                   Émotion finale 🎭


1 er choix:
audio → MLP → logits_audio
texte → MLP → logits_text

→ combinaison

2eme choix: Feature Fusion
audio_vec + text_vec → concat → MLP → prédiction

3eme choix: Attention Fusion
audio_vec ↔ text_vec
        ↓
    attention
        ↓
   fusion_vec
        ↓
      MLP




```

---

## 🔧 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-repo>/emotion-detection.git
cd emotion-detection
```

### 2. Créer l'environnement conda

```bash
conda env create -f environment.yml
conda activate emotion-detection
```

Ou avec pip :

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Télécharger les données

```bash
jupyter notebook notebooks/00_download_data.ipynb
```

Ou directement via le script :

```bash
python src/data/download.py --dataset ravdess crema-d --output data/raw/
```

---

## 📦 Dépendances Principales

```
torch>=2.0.0
transformers>=4.38.0
datasets
openai-whisper
librosa
soundfile
torchaudio
pyaudio              # pour le mode temps réel
fastapi
uvicorn
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
```

---

## 🚀 Utilisation Rapide

### Prédiction sur un fichier audio

```python
from src.inference.predict import EmotionPredictor

predictor = EmotionPredictor(
    audio_model_path="checkpoints/hubert_finetuned/",
    text_model_path="checkpoints/bert_finetuned/",
    fusion_strategy="attention"  # ou "late"
)

result = predictor.predict("chemin/vers/audio.wav")
print(result)
# {'emotion': 'happy', 'confidence': 0.87, 'probabilities': {...}}
```

### Mode Temps Réel (microphone)

```bash
python src/inference/realtime.py --strategy attention
```

### Lancer l'API REST

```bash
uvicorn app.api:app --reload --port 8000
# POST http://localhost:8000/predict  (form-data: file=audio.wav)
```

### Lancer la démo Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Résultats (à compléter)

| Configuration | WA (%) | UA (%) | F1 Macro |
|---------------|--------|--------|----------|
| Baseline Audio (Wav2Vec2) | — | — | — |
| Baseline Texte (BERT) | — | — | — |
| Late Fusion | — | — | — |
| **Attention Fusion** | — | — | — |

---

## 📚 Références

- **Wav2Vec2** : Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*, NeurIPS 2020.
- **HuBERT** : Hsu et al., *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units*, TASLP 2021.
- **Whisper** : Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, ICML 2023.
- **BERT** : Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019.
- **RAVDESS** : Livingstone & Russo, *The Ryerson Audio-Visual Database of Emotional Speech and Song*, PLOS ONE 2018.
- **CREMA-D** : Cao et al., *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset*, IEEE TAFFC 2014.

---

## 📄 Licence

Ce projet est réalisé dans le cadre d'un projet académique de Master 1.
