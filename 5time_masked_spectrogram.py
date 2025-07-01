# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import stft

# # ==== CONFIG ====
# task = 'char'
# masked_dir = f'masked_spectrograms/time/{task}'
# sample_ids = [0, 1, 2]  # change these to check different samples

# # ==== PLOT FUNCTION ====
# def plot_masked_spectrogram(spec, sample_id):
#     fig, axes = plt.subplots(4, 4, figsize=(12, 10))
#     axes = axes.flatten()
#     for ch in range(spec.shape[0]):
#         ax = axes[ch]
#         ax.imshow(spec[ch], aspect='auto', origin='lower', cmap='viridis')
#         ax.set_title(f"Ch {ch}")
#         ax.axis('off')
#     for i in range(spec.shape[0], len(axes)):
#         axes[i].axis('off')
#     plt.suptitle(f"Masked Spectrogram - Sample {sample_id}")
#     plt.tight_layout()
#     plt.show()

# # ==== LOAD AND VISUALIZE ====
# for i in sample_ids:
#     path = os.path.join(masked_dir, f"sample_{i:04d}.npy")
#     if os.path.exists(path):
#         spec = np.load(path)
#         print(f"[INFO] Loaded: {path} - shape: {spec.shape}")
#         plot_masked_spectrogram(spec, i)
#     else:
#         print(f"[WARNING] File not found: {path}")
import numpy as np
import matplotlib.pyplot as plt
import os

# ==== CONFIGURATION ====
char_class = 'char'
sample_idx = 0  # change if you want to visualize another sample
spec_dir_original = f'../3D_spectrogram_Conversoin/generated_3d_spectrograms/{char_class}'
spec_dir_masked = f'masked_spectrograms/time/{char_class}'
output_dir = 'time_masking_verification_plots'
os.makedirs(output_dir, exist_ok=True)

# ==== LOAD FILES ====
original_path = os.path.join(spec_dir_original, f'sample_{sample_idx:04d}.npy')
masked_path = os.path.join(spec_dir_masked, f'sample_{sample_idx:04d}.npy')

original = np.load(original_path)  # shape: [14, F, T]
masked = np.load(masked_path)

# ==== COMPUTE DIFFERENCE MASK ====
diff_mask = (original != masked).astype(int)  # 1 where masked

# ==== PLOTTING ====
def plot_spectrogram(spec, title, save_name):
    fig, axs = plt.subplots(2, 7, figsize=(20, 6))
    axs = axs.flatten()
    for ch in range(14):
        axs[ch].imshow(spec[ch], aspect='auto', origin='lower', cmap='magma')
        axs[ch].set_title(f'Ch {ch}')
        axs[ch].axis('off')
    fig.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

# ==== PLOT ORIGINAL ====
plot_spectrogram(original, "Original Spectrogram (All 14 Channels)", "original_sample.png")

# ==== PLOT MASKED ====
plot_spectrogram(masked, "Time-Masked Spectrogram", "masked_sample.png")

# ==== PLOT DIFFERENCE HIGHLIGHT ====
plot_spectrogram(diff_mask, "Masked Region Highlight (Difference Mask)", "mask_highlight.png")

print(f"[✓] Plots saved to: {output_dir}")
