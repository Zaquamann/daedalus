#!/usr/bin/env python3
"""Preprocess WAV files into a log mel spectrogram dataset saved as .pt files."""

import os
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal
import torch


# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.expanduser("~/Downloads/audio")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed")

SAMPLE_RATE = 44100
N_MELS = 80
N_FFT = 2048
HOP_LENGTH = 441       # ~10ms at 44.1kHz
WIN_LENGTH = 1103      # ~25ms at 44.1kHz
MAX_DURATION_S = 1.0   # pad/truncate all audio to this length
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_S)


# ── Mel filterbank ──────────────────────────────────────────────────────────
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    n_freqs = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_freqs))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)
    return filterbank


# ── Spectrogram ─────────────────────────────────────────────────────────────
def wav_to_log_mel_spectrogram(audio, sr, mel_fb):
    # Compute STFT magnitude (window=25ms, hop=10ms, zero-pad to n_fft=2048)
    _, _, Zxx = scipy.signal.stft(
        audio, fs=sr, nperseg=WIN_LENGTH, noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT, boundary=None,
    )
    power = np.abs(Zxx) ** 2

    # Apply mel filterbank
    mel_spec = mel_fb @ power

    # Log scale with floor to avoid log(0)
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    return log_mel


# ── Label extraction ────────────────────────────────────────────────────────
def extract_label(filename):
    """Extract the word label from filename like '01_01-002_REPLY-C.wav'."""
    name_no_ext = filename.rsplit(".wav", 1)[0]   # '01_01-002_REPLY-C'
    parts = name_no_ext.split("_", 2)             # ['01', '01-002', 'REPLY-C']
    label_part = parts[2]                          # 'REPLY-C'
    label = label_part.rsplit("-C", 1)[0]          # 'REPLY'
    return label.strip()


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    mel_fb = create_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS)

    spectrograms = []
    labels = []
    file_paths = []

    # Walk all speaker-* folders (flat structure: speaker-NN/*.wav)
    speaker_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith("speaker-") and os.path.isdir(os.path.join(DATA_DIR, d))
    )

    for speaker_id in speaker_dirs:
        speaker_path = os.path.join(DATA_DIR, speaker_id)
        wav_files = sorted(
            f for f in os.listdir(speaker_path)
            if f.endswith("-C.wav")
            and not f.startswith("._")
            and "_SENT-END-" not in f
        )
        print(f"  {speaker_id}: {len(wav_files)} files")

        for fname in wav_files:
            fpath = os.path.join(speaker_path, fname)
            sr, audio = wavfile.read(fpath)

            # Convert to float32 normalized to [-1, 1]
            audio = audio.astype(np.float32) / 32768.0

            # Pad (random position) or truncate to MAX_SAMPLES
            if len(audio) < MAX_SAMPLES:
                pad_total = MAX_SAMPLES - len(audio)
                pad_left = np.random.randint(0, pad_total + 1)
                pad_right = pad_total - pad_left
                audio = np.pad(audio, (pad_left, pad_right))
            else:
                audio = audio[:MAX_SAMPLES]

            log_mel = wav_to_log_mel_spectrogram(audio, sr, mel_fb)
            spectrograms.append(log_mel)
            labels.append(extract_label(fname))
            file_paths.append(fpath)

    # Build label-to-index mapping (sorted for determinism)
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    label_indices = [label_to_idx[l] for l in labels]

    # Convert to tensors
    # spectrograms shape: (N, n_mels, n_frames)
    X = torch.tensor(np.array(spectrograms), dtype=torch.float32)
    y = torch.tensor(label_indices, dtype=torch.long)

    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Unique labels: {len(unique_labels)}")
    print(f"Spectrogram: {N_MELS} mel bands x {X.shape[2]} frames")
    print(f"Label mapping ({len(label_to_idx)} classes):")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        count = label_indices.count(idx)
        print(f"  {idx:3d}: {label} ({count} samples)")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "dataset.pt")
    torch.save(
        {
            "spectrograms": X,
            "labels": y,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "file_paths": file_paths,
            "config": {
                "sample_rate": SAMPLE_RATE,
                "n_mels": N_MELS,
                "n_fft": N_FFT,
                "hop_length": HOP_LENGTH,
                "win_length": WIN_LENGTH,
                "max_duration_s": MAX_DURATION_S,
            },
        },
        output_path,
    )
    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()
