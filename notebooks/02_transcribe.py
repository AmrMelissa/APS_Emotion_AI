import os
import pandas as pd
import whisper

#chemin
input_csv = "data/processed/full_dataset.csv"
output_csv = "data/processed/transcriptions/full_dataset_with_text.csv"

os.makedirs("data/processed/transcriptions", exist_ok=True)

#LOAD DATA

df = pd.read_csv(input_csv)

# Fix chemins Windows à Linux
df["path"] = df["path"].str.replace("\\", "/", regex=False)

# Ajouter colonne text si pas existante
if "text" not in df.columns:
    df["text"] = None


#Chargement WHISPER

print("Chargement modèle Whisper")
model = whisper.load_model("base")  # rapide


#RANSCRIPTION

print("Début transcription")

for i in range(len(df)):

    # Skip si déjà traité
    if pd.notna(df.loc[i, "text"]):
        continue

    audio_path = df.loc[i, "path"]

    try:
        result = model.transcribe(audio_path)
        df.loc[i, "text"] = result["text"]

    except Exception as e:
        print(f"erreur sur : {audio_path}")
        df.loc[i, "text"] = ""

    # Sauvegarde tous les 50 fichiers
    if i % 50 == 0:
        df.to_csv(output_csv, index=False)
        print(f"sauvegarde : {i}/{len(df)}")

    print(f"{i}/{len(df)}")

# SAVE FINAL
df.to_csv(output_csv, index=False)

print("\nTranscription terminée !")
print(f"Fichier : {output_csv}")