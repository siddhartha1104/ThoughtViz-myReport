# move this code to outside of this folder and keep it inside 0Report foler 
import os
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIG ==========
SPECTROGRAM_DIR = '/home/sidx/myDrive/shimlaInternship/githubRepos/8ThoughtViz/3D_spectrogram_Conversoin/generated_3d_spectrograms/char'
LABEL_PATH = os.path.join(SPECTROGRAM_DIR, 'char_labels.npy')
OUTPUT_DIR = 'original_spectrogram_plots'
CHAR_CLASSES = ['A', 'C', 'F', 'H', 'J', 'M', 'P', 'S', 'T', 'Y']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD LABELS ==========
labels = np.load(LABEL_PATH)
spec_files = sorted([f for f in os.listdir(SPECTROGRAM_DIR) if f.endswith('.npy') and f.startswith('sample_')])

print(f"[INFO] Loaded {len(labels)} labels and {len(spec_files)} spectrograms")

# ========== SELECT ONE SAMPLE PER CHARACTER ==========
used_classes = set()
selected = []
for i, fname in enumerate(spec_files):
    label = int(labels[i])
    if label not in used_classes:
        used_classes.add(label)
        selected.append((fname, label))
    if len(used_classes) == 10:
        break

print(f"[INFO] Selected {len(selected)} unique spectrograms")

# ========== PLOT ==========
for fname, label in selected:
    spec_path = os.path.join(SPECTROGRAM_DIR, fname)
    spec = np.load(spec_path)  # Shape: [14, F, T]

    plt.figure(figsize=(16, 10))
    for ch in range(spec.shape[0]):
        plt.subplot(4, 4, ch + 1)
        plt.imshow(spec[ch], aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Ch {ch + 1}')
        plt.axis('off')

    char_name = CHAR_CLASSES[label]
    plt.suptitle(f"Spectrogram for Character {char_name} (Label {label})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(OUTPUT_DIR, f'spec_{char_name}_{label}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved: {save_path}")
