# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # === CONFIGURATION ===
# sample_name = "image"  # can be 'char', 'digit', 'image'
# mask_type = "channel"  # 'channel', 'frequency', etc.
# channel_to_check = 13 # which channel to verify (e.g., 0, 1, 2)

# # === PATHS ===
# original_path = f"../3D_spectrogram_Conversoin/generated_3d_spectrograms/{sample_name}/{sample_name}_spectrogram.npy"
# masked_path = f"{mask_type}/{sample_name}_spectrogram_masked.npy"

# # === LOAD ===
# orig = np.load(original_path)
# masked = np.load(masked_path)

# assert orig.shape == masked.shape, "Original and masked shapes must match"
# print(f"[INFO] Loaded spectrogram shape: {orig.shape}")

# # === 1. VISUAL COMPARISON ===
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(orig[channel_to_check], aspect='auto', cmap='viridis')
# plt.title(f"Original - Channel {channel_to_check}")
# plt.colorbar()

# plt.subplot(1, 2, 2)
# plt.imshow(masked[channel_to_check], aspect='auto', cmap='viridis')
# plt.title(f"Masked - Channel {channel_to_check}")
# plt.colorbar()

# plt.suptitle(f"Visual Comparison - {sample_name.capitalize()} - Channel {channel_to_check}", fontsize=14)
# plt.tight_layout()
# plt.show()

# # === 2. STATS CHECK ===
# print("\n[STATS CHECK]")
# print(f"→ Original channel {channel_to_check} sum: {np.sum(orig[channel_to_check]):.4f}")
# print(f"→ Masked channel  {channel_to_check} sum: {np.sum(masked[channel_to_check]):.4f}")
# print(f"→ Masked channel {channel_to_check} mean: {np.mean(masked[channel_to_check]):.4f}")
# print(f"→ Non-zero elements: {np.count_nonzero(masked[channel_to_check])} / {masked[channel_to_check].size}")

# # === 3. DIFFERENCE HEATMAP ===
# diff = orig[channel_to_check] - masked[channel_to_check]
# plt.figure(figsize=(5, 4))
# plt.imshow(diff, aspect='auto', cmap='hot')
# plt.title(f"Difference Heatmap (Original - Masked)\nChannel {channel_to_check}")
# plt.colorbar(label="Power Difference")
# plt.tight_layout()
# plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
CHAR_CLASSES = ['A', 'C', 'F', 'H', 'J', 'M', 'P', 'S', 'T', 'Y']
SPEC_DIR = '/home/sidx/myDrive/shimlaInternship/githubRepos/8ThoughtViz/masked_spectrograms/masked_spectrograms/channel/char'
LABEL_PATH = '/home/sidx/myDrive/shimlaInternship/githubRepos/8ThoughtViz/masked_spectrograms/masked_spectrograms/channel/char_labels.npy'
OUTPUT_DIR = 'channel_masked_spectrogram_channelwise_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load label file ===
labels = np.load(LABEL_PATH)

# === Get list of files ===
spec_files = sorted([f for f in os.listdir(SPEC_DIR) if f.endswith('.npy') and f.startswith('sample')])
print(f"[INFO] Found {len(spec_files)} spectrogram files")

# === Select one sample per class ===
used_classes = set()
selected = []

for i, fname in enumerate(spec_files):
    label = int(labels[i])
    if label not in used_classes:
        used_classes.add(label)
        selected.append((fname, label))
    if len(used_classes) == 10:
        break

# === Plot each selected sample ===
for fname, label in selected:
    fpath = os.path.join(SPEC_DIR, fname)
    spec = np.load(fpath)  # shape: [14, F, T]

    fig, axes = plt.subplots(2, 7, figsize=(18, 6))
    axes = axes.flatten()

    for ch in range(14):
        ax = axes[ch]
        ch_spec = spec[ch]
        is_masked = np.all(ch_spec == 0)
        cmap = 'gray_r' if is_masked else 'inferno'
        ax.imshow(ch_spec, aspect='auto', origin='lower', cmap=cmap)
        ax.set_title(f"Ch {ch} - {'Masked ✅' if is_masked else 'Active ❌'}")
        ax.axis('off')

    fig.suptitle(f"Channel-wise Spectrogram (Masked) for Character: {CHAR_CLASSES[label]} (Label {label})", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(OUTPUT_DIR, f"masked_channels_{CHAR_CLASSES[label]}_{label}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"[✓] Saved: {save_path}")
