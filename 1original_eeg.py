# move this code to outside of this folder and keep it inside 0Report foler 
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# ========== CONFIG ==========
PICKLE_PATH = '../data/eeg/char/data.pkl'
OUTPUT_DIR = 'original_eeg_plots'
CHAR_CLASSES = ['A', 'C', 'F', 'H', 'J', 'M', 'P', 'S', 'T', 'Y']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD EEG DATA ==========
with open(PICKLE_PATH, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

x_test = np.squeeze(data['x_test'])  # Shape: [N, 14, 32]
y_test = data['y_test']              # Shape: [N, 10] (one-hot)

print(f"[INFO] Loaded EEG test data: {x_test.shape} samples")

# ========== FIND ONE EXAMPLE PER CLASS ==========
used_classes = set()
selected_indices = []

for idx in range(len(y_test)):
    label_index = int(np.argmax(y_test[idx]))  # Convert one-hot to index
    if label_index not in used_classes:
        used_classes.add(label_index)
        selected_indices.append((idx, label_index))
    if len(used_classes) == 10:
        break

print(f"[INFO] Found {len(selected_indices)} unique character EEG samples")

# ========== PLOT EACH EEG SAMPLE ==========
for i, (idx, label) in enumerate(selected_indices):
    eeg = x_test[idx]  # Shape: [14, time]
    plt.figure(figsize=(10, 6))
    for ch in range(eeg.shape[0]):
        plt.plot(eeg[ch], label=f'Ch {ch + 1}', linewidth=1)

    plt.title(f'EEG Signal for Character: {CHAR_CLASSES[label]} (Label {label})')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f'eeg_{CHAR_CLASSES[label]}_{label}.png')
    plt.savefig(save_path)
    plt.close()

    print(f"[\u2713] Saved: {save_path}")
