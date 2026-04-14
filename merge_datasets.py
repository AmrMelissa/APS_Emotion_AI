import os
import shutil
import pandas as pd

# =========================
# 📁 PATHS
# =========================
ravdess_path = "data/raw/RAVDESS"
cremad_path = "data/raw/CREMA-D"
output_audio_dir = "data/processed/audio"



os.makedirs(output_audio_dir, exist_ok=True)

print("RAVDESS existe ?", os.path.exists(ravdess_path))
print("CREMA-D existe ?", os.path.exists(cremad_path))
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

for root, _, files in os.walk(ravdess_path):
    for file in files:
        if file.endswith(".wav"):
            emotion = get_ravdess_emotion(file)

            if emotion:
                src = os.path.join(root, file)
                new_name = f"sample_{counter}.wav"
                dst = os.path.join(output_audio_dir, new_name)

                shutil.copy(src, dst)

                data.append([dst, emotion])
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

for root, _, files in os.walk(cremad_path):
    for file in files:
        if file.endswith(".wav"):
            emotion = get_cremad_emotion(file)

            if emotion:
                src = os.path.join(root, file)
                new_name = f"sample_{counter}.wav"
                dst = os.path.join(output_audio_dir, new_name)

                shutil.copy(src, dst)

                data.append([dst, emotion])
                counter += 1

print("✅ CREMA-D terminé")


# =========================
# 📊 DATAFRAME FINAL
# =========================
df = pd.DataFrame(data, columns=["path", "emotion"])

print("\n📊 Distribution :")
print(df["emotion"].value_counts())


# =========================
# 💾 SAVE CSV
# =========================
output_csv = "data/processed/full_dataset.csv"
df.to_csv(output_csv, index=False)

print(f"\n🎉 Dataset sauvegardé : {output_csv}")
print("Total samples :", len(df))