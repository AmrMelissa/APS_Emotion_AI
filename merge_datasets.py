import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# 📁 CONFIG
# =========================
output_audio_dir = "data/audio"
os.makedirs(output_audio_dir, exist_ok=True)

data = []
counter = 0

# =========================
# 🎧 RAVDESS
# =========================
def get_ravdess_emotion(filename):
    try:
        code = int(filename.split("-")[2])
    except:
        return None

    mapping = {
        1: "neutral",
        2: "neutral",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fear",
        7: "disgust",
        8: "surprised"
    }
    return mapping.get(code)


print("🔍 RAVDESS...")

for root, _, files in os.walk("audio_speech_actors_01-24"):
    for file in files:
        if file.endswith(".wav"):
            emotion = get_ravdess_emotion(file)

            if emotion:
                src_path = os.path.join(root, file)

                new_name = f"sample_{counter}.wav"
                dst_path = os.path.join(output_audio_dir, new_name)

                shutil.copy(src_path, dst_path)

                data.append([dst_path, emotion])
                counter += 1

print("✅ RAVDESS terminé")


# =========================
# 🎤 CREMA-D
# =========================
def get_cremad_emotion(filename):
    try:
        code = filename.split("_")[2]
    except:
        return None

    mapping = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad"
    }
    return mapping.get(code)


print("🔍 CREMA-D...")

for root, _, files in os.walk("AudioWAV"):
    for file in files:
        if file.endswith(".wav"):
            emotion = get_cremad_emotion(file)

            if emotion:
                src_path = os.path.join(root, file)

                new_name = f"sample_{counter}.wav"
                dst_path = os.path.join(output_audio_dir, new_name)

                shutil.copy(src_path, dst_path)

                data.append([dst_path, emotion])
                counter += 1

print("✅ CREMA-D terminé")


# =========================
# 📊 DATAFRAME
# =========================
df = pd.DataFrame(data, columns=["path", "emotion"])

print("\n📊 Distribution globale :")
print(df["emotion"].value_counts())


# =========================
# 💾 SAVE FULL DATASET
# =========================
full_path = "data/full_dataset.csv"
df.to_csv(full_path, index=False)
print(f"\n📁 Dataset complet sauvegardé : {full_path}")


# =========================
# 🔀 SPLITS
# =========================
print("\n🔀 Création des splits...")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["emotion"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["emotion"], random_state=42)

os.makedirs("data/splits", exist_ok=True)

train_df.to_csv("data/splits/train.csv", index=False)
val_df.to_csv("data/splits/val.csv", index=False)
test_df.to_csv("data/splits/test.csv", index=False)

print("✅ Splits créés : train / val / test")


# =========================
# 📊 CHECK SPLITS
# =========================
print("\n📊 Train distribution :")
print(train_df["emotion"].value_counts())

print("\n📊 Val distribution :")
print(val_df["emotion"].value_counts())

print("\n📊 Test distribution :")
print(test_df["emotion"].value_counts())


print("\n🎉 TOUT EST PRÊT 🔥")
print("➡️ Audio :", output_audio_dir)
print("➡️ CSV :", full_path)
print("➡️ Splits : data/splits/")