#!/usr/bin/env python3
"""Analyze per-phoneme-class accuracy of the trained WordResNet model."""

import os
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add script dir to path so we can import from train.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train import WordResNet, stratified_split, TEST_SIZE, RANDOM_SEED

DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "processed", "model.pt")
ANALYSIS_DIR = os.path.join(SCRIPT_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ── Grouping helpers ───────────────────────────────────────────────────────

VOWELS = set("AEIOU")

# Words with special onset mapping (pronunciation-based)
SPECIAL_ONSETS = {"ONE": "w"}


def get_onset(word: str) -> str:
    """Map a word to its onset consonant group."""
    word_upper = word.upper()

    # Check special cases first
    if word_upper in SPECIAL_ONSETS:
        return f"/{SPECIAL_ONSETS[word_upper]}/"

    first = word_upper[0]
    two = word_upper[:2] if len(word_upper) >= 2 else ""

    # Vowel-initial
    if first in VOWELS:
        return "vowel"

    # Specific consonant onsets
    onset_map = {
        "S": "/s/",
        "C": "/k/",
        "K": "/k/",
        "Q": "/k/",
        "T": "/t/",
        "F": "/f/",
        "M": "/m/",
        "P": "/p/",
        "B": "/b/",
        "D": "/d/",
        "N": "/n/",
        "R": "/r/",
        "H": "/h/",
        "W": "/w/",
    }

    if first in onset_map:
        return onset_map[first]

    # Everything else: CH, J, G, L, V, Z, Y, SH, etc.
    return "other"


def count_syllables(word: str) -> int:
    """Approximate syllable count by counting vowel clusters."""
    word_upper = word.upper()
    count = len(re.findall(r"[AEIOU]+", word_upper))
    return max(1, count)  # at least 1 syllable


def get_syllable_group(word: str) -> str:
    """Bin words by syllable count."""
    n = count_syllables(word)
    if n >= 4:
        return "4+"
    return str(n)


def get_length_group(word: str) -> str:
    """Bin words by character count."""
    n = len(word)
    if n <= 4:
        return "Short (<=4)"
    elif n <= 7:
        return "Medium (5-7)"
    else:
        return "Long (8+)"


def get_vowel_group(word: str) -> str:
    """Classify word by its primary (stressed) vowel sound based on spelling patterns."""
    w = word.upper().replace("-", "").replace(".", "")

    # /aɪ/ — "I_E", "IGH", "Y_E" patterns
    if (re.search(r"I[A-Z]E$", w) or re.search(r"I[A-Z]E[SD]$", w)
            or "IGH" in w or w in {"FIVE", "NINE", "LINE", "TIME", "TYPE", "RIGHT",
                                    "WIFE", "DIAL", "FILE", "HIBERNATE", "FRIDAY",
                                    "MINIMIZE", "JULY", "NINETY"}):
        return "/aɪ/"

    # /iː/ — "EE", "EA", "E_E", long E patterns
    if ("EE" in w or "EA" in w or re.search(r"E[A-Z]E$", w)
            or w in {"THREE", "SCREEN", "DELETE", "EMAIL", "PREVIOUS",
                      "BEGIN", "REPEAT", "SLEEP"}):
        return "/iː/"

    # /eɪ/ — "A_E", "AY", "AI" patterns
    if ("AY" in w or "AI" in w or re.search(r"A[A-Z]E$", w)
            or re.search(r"A[A-Z]E[SD]$", w)
            or w in {"SAVE", "PLAY", "CHANGE", "PAGE", "PASTE", "EIGHT",
                      "EIGHTEEN", "EIGHTY", "FAVORITES", "FACEBOOK", "APRIL",
                      "PLAYER", "NAME", "MAKE"}):
        return "/eɪ/"

    # /ɔː/ — "ALL", "OR", "OUR", "AW", "AL" patterns
    if ("ALL" in w or "OUR" in w or "OR" in w or "AW" in w
            or w in {"CALL", "FOUR", "FORWARD", "PAUSE", "AUGUST",
                      "FOURTEEN", "FORTY", "QUARTER"}):
        return "/ɔː/"

    # /ʌ/ — short U sounds
    if w in {"ONE", "RUN", "CUT", "SHUT", "MUM", "SON", "HUSBAND",
             "MONDAY", "SUNDAY", "DOUBLE", "HUNDRED", "SUBJECT",
             "BROTHER", "MONTH", "UP", "SUNDAY", "NOVEMBER", "VOLUME"}:
        return "/ʌ/"

    # /æ/ — short A
    if (re.search(r"^[^AEIOU]*A[^AEIOUY]", w) and not re.search(r"A[A-Z]E$", w)
            and w not in {"ALARM", "MARCH"}
            or w in {"BACK", "ADD", "FLASH", "CANCEL", "TAB", "ATTACH",
                      "CAMERA", "CALENDAR", "SATURDAY", "JANUARY",
                      "CALCULATOR", "PARAGRAPH", "ATTACHMENT"}):
        return "/æ/"

    return "other"


def get_frequency_group(total_samples: int) -> str:
    """Bin words by total sample count in the dataset."""
    if total_samples <= 50:
        return "Low (<=50)"
    elif total_samples <= 100:
        return "Medium (51-100)"
    elif total_samples <= 200:
        return "High (101-200)"
    else:
        return "Very High (200+)"


def get_speaker(filepath: str) -> str:
    """Extract speaker ID from file path."""
    m = re.search(r"speaker-(\d+)", filepath)
    return f"speaker-{m.group(1)}" if m else "unknown"


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_horizontal_bars(group_accs, group_counts, overall_acc, title, xlabel, save_path):
    """Create a publication-quality horizontal bar chart of per-group accuracy."""
    # Sort by accuracy (worst at top)
    items = sorted(group_accs.items(), key=lambda x: x[1])
    labels = [k for k, _ in items]
    accs = [v for _, v in items]
    counts = [group_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(labels) + 1.5)))

    colors = plt.cm.RdYlGn(np.array(accs))
    bars = ax.barh(range(len(labels)), [a * 100 for a in accs], color=colors, edgecolor="gray", linewidth=0.5)

    # Annotate each bar
    for i, (bar, acc, n) in enumerate(zip(bars, accs, counts)):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{acc * 100:.1f}% (N={n})", va="center", fontsize=9, fontfamily="sans-serif")

    # Overall accuracy reference line
    ax.axvline(overall_acc * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
               label=f"Overall: {overall_acc * 100:.1f}%")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, fontfamily="sans-serif")
    ax.set_xlabel(xlabel, fontsize=11, fontfamily="sans-serif")
    ax.set_title(title, fontsize=13, fontfamily="sans-serif", fontweight="bold")
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()  # worst at top already from sort, but invert so first item is at top

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_top_bottom_words(word_accs, word_counts, overall_acc, save_path, n=20):
    """Plot the N worst and N best words as two stacked subplots."""
    sorted_words = sorted(word_accs.items(), key=lambda x: x[1])
    worst = sorted_words[:n]
    best = sorted_words[-n:][::-1]  # best first (reversed so best at top)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 14))

    for ax, items, title in [(ax_top, worst, f"{n} Hardest Words"),
                              (ax_bot, best, f"{n} Easiest Words")]:
        labels = [w for w, _ in items]
        accs = [a for _, a in items]
        ns = [word_counts[w] for w in labels]

        colors = plt.cm.RdYlGn(np.array(accs))
        bars = ax.barh(range(len(labels)), [a * 100 for a in accs],
                       color=colors, edgecolor="gray", linewidth=0.5)

        for bar, acc, count in zip(bars, accs, ns):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{acc * 100:.1f}% (N={count})", va="center", fontsize=8, fontfamily="sans-serif")

        ax.axvline(overall_acc * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Overall: {overall_acc * 100:.1f}%")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9, fontfamily="sans-serif")
        ax.set_xlabel("Accuracy (%)", fontsize=10, fontfamily="sans-serif")
        ax.set_title(title, fontsize=12, fontfamily="sans-serif", fontweight="bold")
        ax.set_xlim(0, 112)
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Per-Phoneme-Class Accuracy Analysis")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    data = torch.load(DATA_PATH, weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    file_paths = data["file_paths"]
    num_classes = len(label_to_idx)
    print(f"  {len(labels)} samples, {num_classes} classes")

    # Load model
    print("Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model = WordResNet(num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', '?')}, "
          f"val acc: {checkpoint.get('best_val_acc', 0):.1%}")

    # Stratified split (same as training)
    _, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
    val_specs = spectrograms[val_idx].unsqueeze(1)  # (N, 1, 80, T)
    val_labels = labels[val_idx]
    val_file_paths = [file_paths[i] for i in val_idx]
    print(f"  Validation set: {len(val_labels)} samples")

    # Run inference
    print("\nRunning inference...")
    all_preds = []
    with torch.no_grad():
        # Process in batches
        batch_size = 64
        for i in range(0, len(val_specs), batch_size):
            batch = val_specs[i:i + batch_size]
            logits = model(batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)

    all_preds = torch.cat(all_preds).numpy()
    val_labels_np = val_labels.numpy()

    # Per-word accuracy
    word_correct = defaultdict(int)
    word_total = defaultdict(int)
    for pred, true in zip(all_preds, val_labels_np):
        word = idx_to_label[int(true)]
        word_correct[word] += int(pred == true)
        word_total[word] += 1

    word_accs = {w: word_correct[w] / word_total[w] for w in word_total}
    overall_correct = (all_preds == val_labels_np).sum()
    overall_acc = overall_correct / len(val_labels_np)
    print(f"  Overall accuracy: {overall_acc:.1%} ({overall_correct}/{len(val_labels_np)})")

    # ── A. By onset consonant ──────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("A. Accuracy by Onset Consonant")
    print("-" * 60)

    onset_correct = defaultdict(int)
    onset_total = defaultdict(int)
    for word in word_total:
        onset = get_onset(word)
        onset_correct[onset] += word_correct[word]
        onset_total[onset] += word_total[word]

    onset_accs = {o: onset_correct[o] / onset_total[o] for o in onset_total}
    for onset in sorted(onset_accs, key=onset_accs.get):
        print(f"  {onset:>8s}: {onset_accs[onset]:6.1%} (N={onset_total[onset]})")

    plot_horizontal_bars(onset_accs, onset_total, overall_acc,
                         "Accuracy by Onset Consonant",
                         "Accuracy (%)",
                         os.path.join(ANALYSIS_DIR, "accuracy_by_onset.png"))

    # ── B. By syllable count ───────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("B. Accuracy by Syllable Count")
    print("-" * 60)

    syl_correct = defaultdict(int)
    syl_total = defaultdict(int)
    for word in word_total:
        grp = get_syllable_group(word)
        syl_correct[grp] += word_correct[word]
        syl_total[grp] += word_total[word]

    syl_accs = {g: syl_correct[g] / syl_total[g] for g in syl_total}
    for grp in sorted(syl_accs, key=syl_accs.get):
        print(f"  {grp:>4s} syl: {syl_accs[grp]:6.1%} (N={syl_total[grp]})")

    plot_horizontal_bars(syl_accs, syl_total, overall_acc,
                         "Accuracy by Syllable Count",
                         "Accuracy (%)",
                         os.path.join(ANALYSIS_DIR, "accuracy_by_syllables.png"))

    # ── C. By word length ──────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("C. Accuracy by Word Length")
    print("-" * 60)

    len_correct = defaultdict(int)
    len_total = defaultdict(int)
    for word in word_total:
        grp = get_length_group(word)
        len_correct[grp] += word_correct[word]
        len_total[grp] += word_total[word]

    len_accs = {g: len_correct[g] / len_total[g] for g in len_total}
    for grp in sorted(len_accs, key=len_accs.get):
        print(f"  {grp:>14s}: {len_accs[grp]:6.1%} (N={len_total[grp]})")

    plot_horizontal_bars(len_accs, len_total, overall_acc,
                         "Accuracy by Word Length",
                         "Accuracy (%)",
                         os.path.join(ANALYSIS_DIR, "accuracy_by_length.png"))

    # ── D. Per-word accuracy (top/bottom 20) ───────────────────────────────
    print("\n" + "-" * 60)
    print("D. Per-Word Accuracy (20 hardest / 20 easiest)")
    print("-" * 60)

    sorted_words = sorted(word_accs.items(), key=lambda x: x[1])
    print("  20 Hardest:")
    for w, a in sorted_words[:20]:
        print(f"    {w:>15s}: {a:6.1%} ({word_correct[w]}/{word_total[w]})")
    print("  20 Easiest:")
    for w, a in sorted_words[-20:][::-1]:
        print(f"    {w:>15s}: {a:6.1%} ({word_correct[w]}/{word_total[w]})")

    plot_top_bottom_words(word_accs, word_total, overall_acc,
                          os.path.join(ANALYSIS_DIR, "accuracy_per_word.png"))

    # ── E. By speaker ─────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("E. Accuracy by Speaker")
    print("-" * 60)

    spk_correct = defaultdict(int)
    spk_total = defaultdict(int)
    for i, (pred, true) in enumerate(zip(all_preds, val_labels_np)):
        spk = get_speaker(val_file_paths[i])
        spk_correct[spk] += int(pred == true)
        spk_total[spk] += 1

    spk_accs = {s: spk_correct[s] / spk_total[s] for s in spk_total}
    for spk in sorted(spk_accs, key=spk_accs.get):
        print(f"  {spk:>12s}: {spk_accs[spk]:6.1%} (N={spk_total[spk]})")

    plot_horizontal_bars(spk_accs, spk_total, overall_acc,
                         "Accuracy by Speaker",
                         "Accuracy (%)",
                         os.path.join(ANALYSIS_DIR, "accuracy_by_speaker.png"))

    # ── F. By vowel sound ─────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("F. Accuracy by Vowel Sound")
    print("-" * 60)

    vow_correct = defaultdict(int)
    vow_total = defaultdict(int)
    for word in word_total:
        grp = get_vowel_group(word)
        vow_correct[grp] += word_correct[word]
        vow_total[grp] += word_total[word]

    vow_accs = {g: vow_correct[g] / vow_total[g] for g in vow_total}
    for grp in sorted(vow_accs, key=vow_accs.get):
        print(f"  {grp:>6s}: {vow_accs[grp]:6.1%} (N={vow_total[grp]})")

    plot_horizontal_bars(vow_accs, vow_total, overall_acc,
                         "Accuracy by Vowel Sound",
                         "Accuracy (%)",
                         os.path.join(ANALYSIS_DIR, "accuracy_by_vowel.png"))

    # ── G. By sample frequency ────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("G. Accuracy by Sample Frequency (total dataset samples per word)")
    print("-" * 60)

    # Count total samples per word across entire dataset (not just val)
    all_labels_np = labels.numpy()
    word_total_dataset = defaultdict(int)
    for lbl in all_labels_np:
        word_total_dataset[idx_to_label[int(lbl)]] += 1

    freq_correct = defaultdict(int)
    freq_total = defaultdict(int)
    for word in word_total:
        grp = get_frequency_group(word_total_dataset[word])
        freq_correct[grp] += word_correct[word]
        freq_total[grp] += word_total[word]

    freq_accs = {g: freq_correct[g] / freq_total[g] for g in freq_total}
    for grp in sorted(freq_accs, key=freq_accs.get):
        print(f"  {grp:>18s}: {freq_accs[grp]:6.1%} (N={freq_total[grp]})")

    plot_horizontal_bars(freq_accs, freq_total, overall_acc,
                         "Accuracy by Training Frequency",
                         "Accuracy (%)",
                         os.path.join(ANALYSIS_DIR, "accuracy_by_frequency.png"))

    # ── Original combined figure (A-C) ─────────────────────────────────────
    print("\n" + "-" * 60)
    print("Generating combined A-C figure...")
    print("-" * 60)

    analyses_abc = [
        ("By Onset Consonant", onset_accs, onset_total),
        ("By Syllable Count", syl_accs, syl_total),
        ("By Word Length", len_accs, len_total),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, max(len(d) for _, d, _ in analyses_abc) * 0.5 + 2)))

    for ax, (title, accs, counts) in zip(axes, analyses_abc):
        items = sorted(accs.items(), key=lambda x: x[1])
        labels_list = [k for k, _ in items]
        acc_vals = [v for _, v in items]
        ns = [counts[k] for k in labels_list]

        colors = plt.cm.RdYlGn(np.array(acc_vals))
        bars = ax.barh(range(len(labels_list)), [a * 100 for a in acc_vals],
                       color=colors, edgecolor="gray", linewidth=0.5)

        for i, (bar, acc, n) in enumerate(zip(bars, acc_vals, ns)):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{acc * 100:.1f}% (N={n})", va="center", fontsize=8, fontfamily="sans-serif")

        ax.axvline(overall_acc * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Overall: {overall_acc * 100:.1f}%")

        ax.set_yticks(range(len(labels_list)))
        ax.set_yticklabels(labels_list, fontsize=9, fontfamily="sans-serif")
        ax.set_xlabel("Accuracy (%)", fontsize=10, fontfamily="sans-serif")
        ax.set_title(title, fontsize=11, fontfamily="sans-serif", fontweight="bold")
        ax.set_xlim(0, 108)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Per-Phoneme-Class Accuracy Analysis", fontsize=14, fontweight="bold",
                 fontfamily="sans-serif", y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(ANALYSIS_DIR, "rev_accuracy_by_phoneme.png")
    fig.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {combined_path}")

    # ── H. Combined mega-figure (all 7 analyses) ──────────────────────────
    print("\n" + "-" * 60)
    print("H. Generating combined mega-figure (all breakdowns)...")
    print("-" * 60)

    all_analyses = [
        ("A. By Onset Consonant", onset_accs, onset_total),
        ("B. By Syllable Count", syl_accs, syl_total),
        ("C. By Word Length", len_accs, len_total),
        ("E. By Speaker", spk_accs, spk_total),
        ("F. By Vowel Sound", vow_accs, vow_total),
        ("G. By Training Frequency", freq_accs, freq_total),
    ]

    # Compute max bars for sizing — D (per-word) gets its own 2-row panel
    max_bars = max(len(d) for _, d, _ in all_analyses)
    row_height = 0.4
    fig_height = sum(max(2.5, len(d) * row_height + 1.2) for _, d, _ in all_analyses) + 10  # +10 for per-word panel

    fig = plt.figure(figsize=(10, fig_height))
    n_rows = 7  # 6 bar charts + 1 per-word (which has 2 subplots)
    gs = fig.add_gridspec(n_rows, 1, hspace=0.45)

    # Rows 0-2: A, B, C
    # Row 3: D (per-word top/bottom)
    # Rows 4-6: E, F, G

    analysis_idx = 0
    row_map = [0, 1, 2, 4, 5, 6]  # which gridspec rows for the 6 standard analyses

    for row, (title, accs, counts) in zip(row_map, all_analyses):
        ax = fig.add_subplot(gs[row])
        items = sorted(accs.items(), key=lambda x: x[1])
        labels_list = [k for k, _ in items]
        acc_vals = [v for _, v in items]
        ns = [counts[k] for k in labels_list]

        colors = plt.cm.RdYlGn(np.array(acc_vals))
        bars = ax.barh(range(len(labels_list)), [a * 100 for a in acc_vals],
                       color=colors, edgecolor="gray", linewidth=0.5)

        for bar, acc, n in zip(bars, acc_vals, ns):
            ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{acc * 100:.1f}% (N={n})", va="center", fontsize=7, fontfamily="sans-serif")

        ax.axvline(overall_acc * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                   label=f"Overall: {overall_acc * 100:.1f}%")
        ax.set_yticks(range(len(labels_list)))
        ax.set_yticklabels(labels_list, fontsize=8, fontfamily="sans-serif")
        ax.set_xlabel("Accuracy (%)", fontsize=9, fontfamily="sans-serif")
        ax.set_title(title, fontsize=10, fontfamily="sans-serif", fontweight="bold")
        ax.set_xlim(0, 112)
        ax.legend(loc="lower right", fontsize=7)

    # Row 3: D — per-word hardest/easiest (split into two halves within the subplot)
    ax_d = fig.add_subplot(gs[3])
    # Show top 10 worst in this single subplot (compact version for mega-figure)
    worst_10 = sorted_words[:10]
    labels_w = [w for w, _ in worst_10]
    accs_w = [a for _, a in worst_10]
    ns_w = [word_total[w] for w in labels_w]
    colors_w = plt.cm.RdYlGn(np.array(accs_w))
    bars = ax_d.barh(range(len(labels_w)), [a * 100 for a in accs_w],
                     color=colors_w, edgecolor="gray", linewidth=0.5)
    for bar, acc, n in zip(bars, accs_w, ns_w):
        ax_d.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                  f"{acc * 100:.1f}% (N={n})", va="center", fontsize=7, fontfamily="sans-serif")
    ax_d.axvline(overall_acc * 100, color="black", linestyle="--", linewidth=1.0, alpha=0.7,
                 label=f"Overall: {overall_acc * 100:.1f}%")
    ax_d.set_yticks(range(len(labels_w)))
    ax_d.set_yticklabels(labels_w, fontsize=8, fontfamily="sans-serif")
    ax_d.set_xlabel("Accuracy (%)", fontsize=9, fontfamily="sans-serif")
    ax_d.set_title("D. 10 Hardest Words", fontsize=10, fontfamily="sans-serif", fontweight="bold")
    ax_d.set_xlim(0, 112)
    ax_d.legend(loc="lower right", fontsize=7)

    fig.suptitle("Complete Accuracy Breakdown", fontsize=14, fontweight="bold",
                 fontfamily="sans-serif")
    mega_path = os.path.join(ANALYSIS_DIR, "rev_accuracy_all_breakdowns.png")
    fig.savefig(mega_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {mega_path}")

    # ── Full per-word breakdown ────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("Per-Word Accuracy (sorted worst to best)")
    print("-" * 60)
    for word in sorted(word_accs, key=word_accs.get):
        print(f"  {word:>15s}: {word_accs[word]:6.1%} ({word_correct[word]}/{word_total[word]})")

    print(f"\nDone. All figures saved to {ANALYSIS_DIR}/")


if __name__ == "__main__":
    main()
