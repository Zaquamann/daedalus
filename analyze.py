#!/usr/bin/env python3
"""Model analysis: confusion, embeddings, filters, Grad-CAM, RDMs, and speaker invariance."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from matplotlib.lines import Line2D
from collections import defaultdict
from PIL import Image

from train import ResBlock, WordResNet, stratified_split

# ── Global plot style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# ── Config ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "processed", "dataset.pt")
MODEL_PATH = os.path.join(SCRIPT_DIR, "processed", "model.pt")
OUT_DIR = os.path.join(SCRIPT_DIR, "analysis")
TEST_SIZE = 0.33
RANDOM_SEED = 42

# Phonetic categories and their display colors
CATEGORIES = {
    "Numbers": {
        "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
        "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN",
        "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY",
        "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY",
        "HUNDRED", "THOUSAND", "MILLION", "HALF", "QUARTER",
    },
    "Months": {
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
        "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    },
    "Days": {
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
        "SUNDAY", "TODAY", "TOMORROW", "YESTERDAY",
    },
    "Family": {
        "MUM", "DAD", "SON", "DAUGHTER", "WIFE", "HUSBAND", "BROTHER",
        "SISTER",
    },
    "Commands": {
        "OPEN", "CLOSE", "SAVE", "DELETE", "COPY", "CUT", "PASTE", "SEND",
        "REPLY", "CLICK", "SCROLL", "MOVE", "GO", "STOP", "RUN", "PLAY",
        "PAUSE", "RECORD", "SEARCH", "CHECK", "ADD", "EDIT", "EXPORT",
        "PRINT", "INSERT", "CHANGE", "BEGIN", "END", "RESTART",
    },
    "Navigation": {
        "UP", "DOWN", "LEFT", "RIGHT", "BACK", "FORWARD", "HOME", "PAGE",
    },
    "Media/Tech": {
        "BROWSER", "GOOGLE", "FACEBOOK", "YOUTUBE", "CALCULATOR", "CAMERA",
        "COMPUTER", "E-MAIL", "SCREEN", "WINDOW", "FLASH", "ALARM",
        "CALENDAR", "MUSIC", "HOMEPAGE",
    },
}

CAT_COLORS = {
    "Numbers": "#1f77b4",
    "Months": "#2ca02c",
    "Days": "#ff7f0e",
    "Commands": "#d62728",
    "Family": "#9467bd",
    "Navigation": "#8c564b",
    "Media/Tech": "#17becf",
    "Other": "#999999",
}

# Canonical ordering for categories
CAT_ORDER = ["Numbers", "Months", "Days", "Commands", "Family",
             "Navigation", "Media/Tech", "Other"]


def get_category(word):
    for cat, words in CATEGORIES.items():
        if word in words:
            return cat
    return "Other"


def savefig(fig, name):
    """Save figure as both PNG and PDF."""
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


def get_sorted_class_order(idx_to_label):
    """Sort class indices by category then word name."""
    class_cats = [(idx, idx_to_label[idx], get_category(idx_to_label[idx]))
                  for idx in sorted(idx_to_label.keys())]
    class_cats.sort(key=lambda x: (CAT_ORDER.index(x[2])
                                   if x[2] in CAT_ORDER else len(CAT_ORDER),
                                   x[1]))
    return class_cats


def get_cat_boundaries(sorted_class_cats):
    """Return boundary indices between categories for drawing separator lines."""
    boundaries = []
    prev_cat = None
    for i, (_, _, cat) in enumerate(sorted_class_cats):
        if cat != prev_cat and prev_cat is not None:
            boundaries.append(i - 0.5)
        prev_cat = cat
    return boundaries


# ── Data loading ────────────────────────────────────────────────────────────
def load_data_and_model():
    data = torch.load(DATA_PATH, weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    file_paths = data["file_paths"]
    num_classes = len(label_to_idx)

    _, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
    val_X = spectrograms[val_idx].unsqueeze(1)
    val_y = labels[val_idx]
    val_paths = [file_paths[i] for i in val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordResNet(num_classes)
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Also return raw spectrograms for input RDM
    val_specs = spectrograms[val_idx]  # (N, 80, 99) without channel dim

    print(f"Loaded {len(val_y)} val samples, {num_classes} classes")
    print(f"Model from epoch {ckpt['epoch']}, val acc {ckpt['best_val_acc']:.1%}")
    print(f"Device: {device}")

    return (model, val_X, val_specs, val_y, val_paths,
            idx_to_label, label_to_idx, device)


def extract_speaker(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("speaker-"):
            return p
    return "unknown"


def batch_forward(model, val_X, device, batch_size=64):
    """Run forward pass in batches, return predictions."""
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(val_X), batch_size):
            batch = val_X[i:i+batch_size].to(device)
            preds = model(batch).argmax(1).cpu()
            all_preds.append(preds)
    return torch.cat(all_preds).numpy()


def extract_layer_activations(model, val_X, device, batch_size=64):
    """Extract activations at block1, block2, and GAP layers."""
    layer_acts = {"block1": [], "block2": [], "gap": []}

    def make_hook(name):
        def hook_fn(module, inp, out):
            layer_acts[name].append(out.detach().cpu())
        return hook_fn

    handles = [
        model.block1.register_forward_hook(make_hook("block1")),
        model.block2.register_forward_hook(make_hook("block2")),
        model.gap.register_forward_hook(make_hook("gap")),
    ]

    with torch.no_grad():
        for i in range(0, len(val_X), batch_size):
            batch = val_X[i:i+batch_size].to(device)
            model(batch)

    for h in handles:
        h.remove()

    return {k: torch.cat(v).flatten(1).numpy() for k, v in layer_acts.items()}


# ── Fig 1a: Confusion Matrix ───────────────────────────────────────────────
def fig1a_confusion_matrix(model, val_X, val_y, idx_to_label, device):
    print("\n[1/9] Fig 1a: Confusion matrix...")
    from sklearn.metrics import confusion_matrix

    all_preds = batch_forward(model, val_X, device)
    all_true = val_y.numpy()
    num_classes = len(idx_to_label)

    cm_raw = confusion_matrix(all_true, all_preds, labels=np.arange(num_classes))

    # Sort by category
    sorted_cats = get_sorted_class_order(idx_to_label)
    order = [c[0] for c in sorted_cats]
    cm_sorted = cm_raw[np.ix_(order, order)]

    # Row-normalize
    row_sums = cm_sorted.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    cm_norm = cm_sorted / row_sums

    # Category boundaries
    boundaries = get_cat_boundaries(sorted_cats)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest",
                   vmin=0, vmax=1)

    # Draw category boundary lines
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=0.5, alpha=0.7)
        ax.axvline(b, color="black", linewidth=0.5, alpha=0.7)

    # Show every 10th class label
    tick_positions = list(range(0, num_classes, 10))
    sorted_labels = [c[1] for c in sorted_cats]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([sorted_labels[i] for i in tick_positions],
                       rotation=90, fontsize=6)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([sorted_labels[i] for i in tick_positions], fontsize=6)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion Matrix (row-normalized, sorted by phonetic category)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion", fontsize=10)

    savefig(fig, "fig1a_confusion_matrix")

    # Return data for fig1b
    return cm_raw, all_true, all_preds


# ── Fig 1b: Top-20 Confused Pairs ──────────────────────────────────────────
def fig1b_top_confusions(cm_raw, idx_to_label):
    print("[2/9] Fig 1b: Top-20 confused pairs...")
    num_classes = cm_raw.shape[0]
    cm_off = cm_raw.copy()
    np.fill_diagonal(cm_off, 0)

    # Compute confusion rate (proportion of true class confused as pred)
    row_sums = cm_raw.sum(axis=1)

    confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_off[i, j] > 0:
                rate = cm_off[i, j] / max(row_sums[i], 1)
                confused.append((rate, cm_off[i, j], idx_to_label[i],
                                 idx_to_label[j]))
    confused.sort(reverse=True)
    top20 = confused[:20]

    # Phonetic annotations
    ANNOTATIONS = {
        ("TO", "TWO"): "homophones",
        ("TWO", "TO"): "homophones",
        ("SEVENTEEN", "SEVENTY"): "shared prefix",
        ("SEVENTY", "SEVENTEEN"): "shared prefix",
        ("LINE", "NINE"): "rhyme (-ine/-ine)",
        ("NINE", "LINE"): "rhyme (-ine/-ine)",
        ("FORTY", "FOURTEEN"): "shared prefix",
        ("FOURTEEN", "FORTY"): "shared prefix",
        ("THIRTY", "THIRTEEN"): "shared prefix",
        ("THIRTEEN", "THIRTY"): "shared prefix",
        ("FIFTEEN", "FIFTY"): "shared prefix",
        ("FIFTY", "FIFTEEN"): "shared prefix",
        ("NINETEEN", "NINETY"): "shared prefix",
        ("NINETY", "NINETEEN"): "shared prefix",
        ("FILE", "FIVE"): "shared onset (f-)",
        ("FIVE", "FILE"): "shared onset (f-)",
        ("NEW", "VIEW"): "rhyme (-ew/-iew)",
        ("VIEW", "NEW"): "rhyme (-ew/-iew)",
        ("TEN", "TURN"): "shared onset (t-)",
        ("TURN", "TEN"): "shared onset (t-)",
        ("HALF", "PAST"): "time-related",
        ("FOUR", "ALL"): "phonetic overlap",
        ("EIGHT", "PAGE"): "rhyme (-ate/-age)",
    }

    labels = []
    rates = []
    colors = []
    annotations = []
    for rate, count, true_l, pred_l in top20:
        labels.append(f"{true_l} -> {pred_l}")
        rates.append(rate * 100)
        annot = ANNOTATIONS.get((true_l, pred_l), "")
        annotations.append(annot)
        # Green if phonetically expected, red if unexpected
        colors.append("#2ca02c" if annot else "#d62728")

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(labels))[::-1]
    bars = ax.barh(y_pos, rates, color=colors, edgecolor="black",
                   linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace")
    ax.set_xlabel("Confusion rate (%)")
    ax.set_title("Top-20 Most Confused Word Pairs")

    # Add annotations
    for i, (bar, annot) in enumerate(zip(bars, annotations)):
        if annot:
            ax.text(bar.get_width() + 0.3, y_pos[i], annot,
                    va="center", fontsize=7, fontstyle="italic", color="#555555")

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#2ca02c", lw=6, label="Phonetically expected"),
        Line2D([0], [0], color="#d62728", lw=6, label="Unexpected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    savefig(fig, "fig1b_top_confusions")


# ── Fig 2a/2b: UMAP Embeddings ─────────────────────────────────────────────
def fig2_umap(model, val_X, val_y, val_paths, idx_to_label, device):
    print("[3/9] Fig 2a/2b: UMAP embeddings...")

    # Extract GAP embeddings
    embeddings = []
    def hook_fn(module, input, output):
        embeddings.append(output.flatten(1).cpu())

    handle = model.gap.register_forward_hook(hook_fn)
    with torch.no_grad():
        for i in range(0, len(val_X), 64):
            batch = val_X[i:i+64].to(device)
            model(batch)
    handle.remove()

    emb = torch.cat(embeddings).numpy()
    labels_np = val_y.numpy()

    # UMAP (fall back to t-SNE)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED,
                            n_neighbors=15, min_dist=0.1)
        emb_2d = reducer.fit_transform(emb)
        method = "UMAP"
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30,
                       random_state=RANDOM_SEED, max_iter=1000)
        emb_2d = reducer.fit_transform(emb)
        method = "t-SNE"
    print(f"  Using {method}, shape: {emb_2d.shape}")

    # ── Fig 2a: by category ──
    word_labels = [idx_to_label[l] for l in labels_np]
    cat_labels = np.array([get_category(w) for w in word_labels])

    fig, ax = plt.subplots(figsize=(10, 8))
    for cat in CAT_ORDER:
        mask = cat_labels == cat
        if not mask.any():
            continue
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=CAT_COLORS[cat], s=15, alpha=0.6, label=cat,
                   edgecolors="none")

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9,
              markerscale=2, framealpha=0.9)
    ax.set_title(f"{method} of 128-dim GAP Embeddings (by phonetic category)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    savefig(fig, "fig2a_umap_by_category")

    # ── Fig 2b: by speaker ──
    speakers = np.array([extract_speaker(p) for p in val_paths])
    unique_speakers = sorted(set(speakers))
    speaker_cmap = plt.get_cmap("tab20", len(unique_speakers))

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, spk in enumerate(unique_speakers):
        mask = speakers == spk
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                   c=[speaker_cmap(i)], s=12, alpha=0.5,
                   label=spk.replace("speaker-", "S"),
                   edgecolors="none")

    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7,
              markerscale=2, ncol=2, framealpha=0.9)
    ax.set_title(f"{method} of GAP Embeddings (by speaker)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    savefig(fig, "fig2b_umap_by_speaker")


# ── Fig 3: Conv1 Filters ───────────────────────────────────────────────────
def fig3_filters(model):
    print("[4/9] Fig 3: First-layer filters...")
    filters = model.block1.conv1.weight.data.cpu().numpy()  # (64, 1, 3, 3)
    n_filters = filters.shape[0]
    vmax = np.abs(filters).max()

    fig, axes = plt.subplots(8, 8, figsize=(12, 6))
    for i in range(n_filters):
        ax = axes[i // 8, i % 8]
        im = ax.imshow(filters[i, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

    # Shared axis labels on outer edges
    for i in range(8):
        axes[7, i].set_xlabel("Time", fontsize=7)
        axes[i, 0].set_ylabel("Freq", fontsize=7)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Weight", fontsize=9)

    fig.suptitle("Learned Spectro-Temporal Receptive Fields (Conv1)\n"
                 "64 filters (3x3); blue = inhibitory, red = excitatory",
                 fontsize=12, fontweight="bold")
    fig.subplots_adjust(right=0.90, hspace=0.3, wspace=0.15)

    savefig(fig, "fig3_conv1_filters")


# ── Fig 4: Grad-CAM ────────────────────────────────────────────────────────
def fig4_gradcam(model, val_X, val_y, idx_to_label, label_to_idx, device):
    print("[5/9] Fig 4: Grad-CAM...")
    target_words = [
        "FIVE", "JANUARY", "MONDAY", "CLICK", "BROTHER", "BACK",
        "PLAY", "CALCULATOR", "GO", "THIRTY", "THIRTEEN", "TO",
    ]
    # Find examples in val set
    labels_np = val_y.numpy()
    examples = []
    for word in target_words:
        if word not in label_to_idx:
            continue
        target_idx = label_to_idx[word]
        matches = np.where(labels_np == target_idx)[0]
        if len(matches) > 0:
            examples.append((word, matches[0]))
    if len(examples) < 12:
        # Fill remaining slots with available words
        used = {w for w, _ in examples}
        for word in ["PAUSE", "GOOGLE", "DAUGHTER", "EIGHT", "STOP", "MUSIC"]:
            if len(examples) >= 12:
                break
            if word in label_to_idx and word not in used:
                idx = label_to_idx[word]
                matches = np.where(labels_np == idx)[0]
                if len(matches) > 0:
                    examples.append((word, matches[0]))
                    used.add(word)

    n_examples = len(examples)
    ncols = 4
    nrows = (n_examples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    # Flatten for easy indexing
    ax_flat = axes.flatten()

    for i, (word, sample_idx) in enumerate(examples):
        x = val_X[sample_idx:sample_idx+1].to(device).requires_grad_(True)

        activations = {}
        gradients = {}

        def fwd_hook(module, inp, out):
            activations["block2"] = out

        def bwd_hook(module, grad_in, grad_out):
            gradients["block2"] = grad_out[0]

        h1 = model.block2.register_forward_hook(fwd_hook)
        h2 = model.block2.register_full_backward_hook(bwd_hook)

        logits = model(x)
        pred_class = logits.argmax(1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_class].item()
        pred_label = idx_to_label[pred_class]

        model.zero_grad()
        logits[0, pred_class].backward()
        h1.remove()
        h2.remove()

        act = activations["block2"].detach().cpu().numpy()[0]
        grad = gradients["block2"].detach().cpu().numpy()[0]
        weights = grad.mean(axis=(1, 2))
        cam = np.sum(weights[:, None, None] * act, axis=0)
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        spec = val_X[sample_idx, 0].numpy()
        cam_resized = np.array(Image.fromarray(cam).resize(
            (spec.shape[1], spec.shape[0]), Image.BILINEAR))

        ax = ax_flat[i]
        ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        ax.imshow(cam_resized, aspect="auto", origin="lower",
                  cmap="jet", alpha=0.4)
        ax.set_title(f"{word} ({confidence:.0%})", fontsize=10)
        if i % ncols == 0:
            ax.set_ylabel("Mel band")
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel("Time (frames)")

    # Hide unused axes
    for j in range(len(examples), len(ax_flat)):
        ax_flat[j].set_visible(False)

    fig.suptitle("Grad-CAM Attention Maps (Block 2 activations)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig4_gradcam_examples")


# ── Fig 5a: Layer-wise RDMs ────────────────────────────────────────────────
def fig5a_rdm(layer_acts, val_specs, val_y, idx_to_label):
    print("[6/9] Fig 5a: Layer-wise RDMs...")
    from scipy.spatial.distance import cosine

    labels_np = val_y.numpy()
    num_classes = len(idx_to_label)
    sorted_cats = get_sorted_class_order(idx_to_label)
    order = [c[0] for c in sorted_cats]
    boundaries = get_cat_boundaries(sorted_cats)

    # Include input spectrogram as baseline
    input_acts = val_specs.numpy().reshape(len(val_specs), -1)
    all_layers = {"Input": input_acts}
    all_layers.update(layer_acts)
    layer_names = ["Input", "block1", "block2", "gap"]
    display_names = ["Input", "Block 1", "Block 2", "GAP"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    rdms = {}

    for ax_idx, (layer_name, display_name) in enumerate(
            zip(layer_names, display_names)):
        acts = all_layers[layer_name]

        # Class-mean activations
        class_means = np.zeros((num_classes, acts.shape[1]))
        for cls in range(num_classes):
            mask = labels_np == cls
            if mask.sum() > 0:
                class_means[cls] = acts[mask].mean(axis=0)

        # Reorder by category
        class_means = class_means[order]

        # Cosine distance RDM
        rdm = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                d = cosine(class_means[i], class_means[j])
                rdm[i, j] = d
                rdm[j, i] = d
        rdms[layer_name] = rdm

        im = axes[ax_idx].imshow(rdm, cmap="viridis", interpolation="nearest")
        for b in boundaries:
            axes[ax_idx].axhline(b, color="white", linewidth=0.3, alpha=0.8)
            axes[ax_idx].axvline(b, color="white", linewidth=0.3, alpha=0.8)
        axes[ax_idx].set_title(display_name)
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Word class (sorted by category)")

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Dissimilarity (1 - cosine sim)")

    fig.suptitle("Representational Dissimilarity Matrices Across Layers",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "fig5a_rdm_layers")

    return rdms


# ── Fig 5b: Dendrograms ────────────────────────────────────────────────────
def fig5b_dendrograms(rdms, idx_to_label):
    print("[7/9] Fig 5b: Dendrograms...")
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform

    sorted_cats = get_sorted_class_order(idx_to_label)
    order = [c[0] for c in sorted_cats]
    sorted_labels = [c[1] for c in sorted_cats]
    sorted_cat_names = [c[2] for c in sorted_cats]

    layer_names = ["block1", "block2", "gap"]
    display_names = ["Block 1", "Block 2", "GAP"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax_idx, (layer_name, display_name) in enumerate(
            zip(layer_names, display_names)):
        rdm = rdms[layer_name]
        # Ensure symmetry and zero diagonal
        rdm = (rdm + rdm.T) / 2
        np.fill_diagonal(rdm, 0)
        # Clamp small negative values from floating point
        rdm = np.maximum(rdm, 0)

        condensed = squareform(rdm)
        Z = linkage(condensed, method="ward")

        # Color branches by category (leaf colors)
        leaf_colors = {}
        for i, cat in enumerate(sorted_cat_names):
            leaf_colors[i] = CAT_COLORS.get(cat, "#999999")

        dendrogram(Z, ax=axes[ax_idx], no_labels=True,
                   above_threshold_color="#cccccc")
        axes[ax_idx].set_title(display_name)
        axes[ax_idx].set_xlabel("Word classes")
        if ax_idx == 0:
            axes[ax_idx].set_ylabel("Ward distance")

    # Add category legend
    legend_elements = [Line2D([0], [0], color=CAT_COLORS[cat], lw=3, label=cat)
                       for cat in CAT_ORDER if cat in CAT_COLORS]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.9)

    fig.suptitle("Hierarchical Clustering of Word Representations",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    savefig(fig, "fig5b_dendrograms")


# ── Fig 6: Speaker Invariance ──────────────────────────────────────────────
def fig6_speaker_invariance(layer_acts, val_specs, val_y, val_paths):
    print("[8/9] Fig 6: Speaker invariance...")
    labels_np = val_y.numpy()
    speakers = [extract_speaker(p) for p in val_paths]
    unique_speakers = sorted(set(speakers))
    unique_words = sorted(set(labels_np))
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    speaker_ids = np.array([speaker_to_idx[s] for s in speakers])

    input_acts = val_specs.numpy().reshape(len(val_specs), -1)
    all_layers = {"Input": input_acts}
    all_layers.update(layer_acts)
    layer_names = ["Input", "block1", "block2", "gap"]
    display_names = ["Input", "Block 1", "Block 2", "GAP"]

    word_vars = []
    speaker_vars = []

    for layer_name in layer_names:
        acts = all_layers[layer_name]
        total_var = np.var(acts, axis=0).mean()

        # Between-word variance
        word_means = []
        for w in unique_words:
            mask = labels_np == w
            if mask.sum() > 0:
                word_means.append(acts[mask].mean(axis=0))
        word_means = np.array(word_means)
        wv = np.var(word_means, axis=0).mean()

        # Between-speaker variance
        speaker_means = []
        for s_idx in range(len(unique_speakers)):
            mask = speaker_ids == s_idx
            if mask.sum() > 0:
                speaker_means.append(acts[mask].mean(axis=0))
        speaker_means = np.array(speaker_means)
        sv = np.var(speaker_means, axis=0).mean()

        # Normalize to fractions of total
        wv_frac = wv / max(total_var, 1e-10)
        sv_frac = sv / max(total_var, 1e-10)

        word_vars.append(wv_frac)
        speaker_vars.append(sv_frac)
        print(f"  {layer_name}: word_frac={wv_frac:.3f}, "
              f"speaker_frac={sv_frac:.3f}, "
              f"ratio={wv / max(sv, 1e-10):.1f}")

    x = np.arange(len(layer_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width/2, word_vars, width, label="Between-word variance",
                   color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, speaker_vars, width,
                   label="Between-speaker variance",
                   color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel("Fraction of total variance")
    ax.set_xlabel("Layer")
    ax.set_title("Variance Decomposition Across Layers")
    ax.legend(fontsize=9)

    # Add ratio annotations
    for i, (wv, sv) in enumerate(zip(word_vars, speaker_vars)):
        ratio = wv / max(sv, 1e-10)
        ax.text(x[i], max(wv, sv) + 0.01, f"ratio: {ratio:.1f}",
                ha="center", fontsize=8, fontstyle="italic", color="#555555")

    savefig(fig, "fig6_speaker_invariance")


# ── Comparison: Confusion Matrix (side-by-side) ────────────────────────────
def fig_cmp_confusion(model_u, model_t, val_X, val_y, idx_to_label, device):
    print("\n[cmp 1] Confusion matrix comparison...")
    from sklearn.metrics import confusion_matrix

    num_classes = len(idx_to_label)
    sorted_cats = get_sorted_class_order(idx_to_label)
    order = [c[0] for c in sorted_cats]
    boundaries = get_cat_boundaries(sorted_cats)
    sorted_labels = [c[1] for c in sorted_cats]
    tick_positions = list(range(0, num_classes, 10))

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    titles = ["Untrained (random weights)", "Trained"]

    for ax, mdl, title in zip(axes, [model_u, model_t], titles):
        preds = batch_forward(mdl, val_X, device)
        true = val_y.numpy()
        cm_raw = confusion_matrix(true, preds, labels=np.arange(num_classes))
        cm_sorted = cm_raw[np.ix_(order, order)]
        row_sums = np.maximum(cm_sorted.sum(axis=1, keepdims=True), 1)
        cm_norm = cm_sorted / row_sums

        acc = (preds == true).mean()

        im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest",
                       vmin=0, vmax=1)
        for b in boundaries:
            ax.axhline(b, color="black", linewidth=0.5, alpha=0.7)
            ax.axvline(b, color="black", linewidth=0.5, alpha=0.7)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([sorted_labels[i] for i in tick_positions],
                           rotation=90, fontsize=5)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([sorted_labels[i] for i in tick_positions], fontsize=5)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{title} (acc: {acc:.1%})")

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Proportion")
    fig.suptitle("Confusion Matrix: Untrained vs Trained", fontweight="bold")
    plt.tight_layout()
    savefig(fig, "cmp_confusion_matrix")


# ── Comparison: UMAP ──────────────────────────────────────────────────────
def fig_cmp_umap(model_u, model_t, val_X, val_y, val_paths, idx_to_label,
                 device):
    print("[cmp 2] UMAP comparison...")
    labels_np = val_y.numpy()
    word_labels = [idx_to_label[l] for l in labels_np]
    cat_labels = np.array([get_category(w) for w in word_labels])
    speakers = np.array([extract_speaker(p) for p in val_paths])
    unique_speakers = sorted(set(speakers))
    speaker_cmap = plt.get_cmap("tab20", len(unique_speakers))

    try:
        import umap
        method = "UMAP"
    except ImportError:
        method = "t-SNE"

    def get_emb(mdl):
        embeddings = []
        def hook_fn(module, input, output):
            embeddings.append(output.flatten(1).cpu())
        handle = mdl.gap.register_forward_hook(hook_fn)
        with torch.no_grad():
            for i in range(0, len(val_X), 64):
                batch = val_X[i:i+64].to(device)
                mdl(batch)
        handle.remove()
        return torch.cat(embeddings).numpy()

    emb_u = get_emb(model_u)
    emb_t = get_emb(model_t)

    def reduce(emb):
        if method == "UMAP":
            import umap
            return umap.UMAP(n_components=2, random_state=RANDOM_SEED,
                             n_neighbors=15, min_dist=0.1).fit_transform(emb)
        else:
            from sklearn.manifold import TSNE
            return TSNE(n_components=2, perplexity=30,
                        random_state=RANDOM_SEED, max_iter=1000).fit_transform(emb)

    emb_u_2d = reduce(emb_u)
    emb_t_2d = reduce(emb_t)

    # ── By category (2x1) ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, emb_2d, title in zip(axes, [emb_u_2d, emb_t_2d],
                                  ["Untrained", "Trained"]):
        for cat in CAT_ORDER:
            mask = cat_labels == cat
            if not mask.any():
                continue
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=CAT_COLORS[cat], s=12, alpha=0.6, label=cat,
                       edgecolors="none")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                   markerscale=2, framealpha=0.9)
    fig.suptitle(f"{method} Embeddings by Category: Untrained vs Trained",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, "cmp_umap_by_category")

    # ── By speaker (2x1) ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, emb_2d, title in zip(axes, [emb_u_2d, emb_t_2d],
                                  ["Untrained", "Trained"]):
        for i, spk in enumerate(unique_speakers):
            mask = speakers == spk
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[speaker_cmap(i)], s=10, alpha=0.5,
                       label=spk.replace("speaker-", "S"),
                       edgecolors="none")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6,
                   markerscale=2, ncol=2, framealpha=0.9)
    fig.suptitle(f"{method} Embeddings by Speaker: Untrained vs Trained",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, "cmp_umap_by_speaker")


# ── Comparison: Filters ───────────────────────────────────────────────────
def fig_cmp_filters(model_u, model_t):
    print("[cmp 3] Filter comparison...")
    fig, all_axes = plt.subplots(8, 16, figsize=(20, 6))

    for col_offset, mdl, title in [(0, model_u, "Untrained"),
                                    (8, model_t, "Trained")]:
        filters = mdl.block1.conv1.weight.data.cpu().numpy()
        vmax = np.abs(filters).max()
        for i in range(64):
            r, c = i // 8, (i % 8) + col_offset
            ax = all_axes[r, c]
            ax.imshow(filters[i, 0], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

    # Column group labels
    fig.text(0.28, 0.98, "Untrained (random init)", ha="center", fontsize=11,
             fontweight="bold")
    fig.text(0.72, 0.98, "Trained", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle("Conv1 Filters: Before vs After Training", fontsize=13,
                 fontweight="bold", y=1.03)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    savefig(fig, "cmp_conv1_filters")


# ── Comparison: Grad-CAM ─────────────────────────────────────────────────
def fig_cmp_gradcam(model_u, model_t, val_X, val_y, idx_to_label,
                    label_to_idx, device):
    print("[cmp 4] Grad-CAM comparison...")
    target_words = ["FIVE", "MONDAY", "CLICK", "BROTHER", "PLAY", "THIRTEEN"]
    labels_np = val_y.numpy()
    examples = []
    for word in target_words:
        if word not in label_to_idx:
            continue
        target_idx = label_to_idx[word]
        matches = np.where(labels_np == target_idx)[0]
        if len(matches) > 0:
            examples.append((word, matches[0]))

    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3 * n))
    col_titles = ["Spectrogram", "Untrained Grad-CAM", "Trained Grad-CAM"]

    for row, (word, sample_idx) in enumerate(examples):
        spec = val_X[sample_idx, 0].numpy()

        # Column 0: spectrogram
        axes[row, 0].imshow(spec, aspect="auto", origin="lower", cmap="viridis")
        axes[row, 0].set_ylabel(word, fontsize=11, fontweight="bold")

        for col, mdl, label in [(1, model_u, "Untrained"),
                                 (2, model_t, "Trained")]:
            x = val_X[sample_idx:sample_idx+1].to(device).requires_grad_(True)
            activations = {}
            gradients = {}

            def fwd_hook(module, inp, out):
                activations["b2"] = out
            def bwd_hook(module, grad_in, grad_out):
                gradients["b2"] = grad_out[0]

            h1 = mdl.block2.register_forward_hook(fwd_hook)
            h2 = mdl.block2.register_full_backward_hook(bwd_hook)
            logits = mdl(x)
            pred_class = logits.argmax(1).item()
            conf = torch.softmax(logits, dim=1)[0, pred_class].item()
            mdl.zero_grad()
            logits[0, pred_class].backward()
            h1.remove()
            h2.remove()

            act = activations["b2"].detach().cpu().numpy()[0]
            grad = gradients["b2"].detach().cpu().numpy()[0]
            weights = grad.mean(axis=(1, 2))
            cam = np.maximum(np.sum(weights[:, None, None] * act, axis=0), 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            cam_resized = np.array(Image.fromarray(cam).resize(
                (spec.shape[1], spec.shape[0]), Image.BILINEAR))

            axes[row, col].imshow(spec, aspect="auto", origin="lower",
                                  cmap="viridis")
            axes[row, col].imshow(cam_resized, aspect="auto", origin="lower",
                                  cmap="jet", alpha=0.4)
            pred_word = idx_to_label[pred_class]
            axes[row, col].set_title(
                f"{label}: {pred_word} ({conf:.0%})" if row == 0
                else f"{pred_word} ({conf:.0%})", fontsize=9)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(f"{title}\n{axes[0, col].get_title()}",
                               fontsize=10)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (frames)")

    fig.suptitle("Grad-CAM: Untrained vs Trained Attention",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "cmp_gradcam")


# ── Comparison: RDMs ──────────────────────────────────────────────────────
def fig_cmp_rdm(model_u, model_t, val_X, val_specs, val_y, idx_to_label,
                device):
    print("[cmp 5] RDM comparison...")
    from scipy.spatial.distance import cosine

    labels_np = val_y.numpy()
    num_classes = len(idx_to_label)
    sorted_cats = get_sorted_class_order(idx_to_label)
    order = [c[0] for c in sorted_cats]
    boundaries = get_cat_boundaries(sorted_cats)

    acts_u = extract_layer_activations(model_u, val_X, device)
    acts_t = extract_layer_activations(model_t, val_X, device)

    layer_names = ["block1", "block2", "gap"]
    display_names = ["Block 1", "Block 2", "GAP"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    row_labels = ["Untrained", "Trained"]

    for row, (acts_dict, row_label) in enumerate(
            [(acts_u, "Untrained"), (acts_t, "Trained")]):
        for col, (lname, dname) in enumerate(zip(layer_names, display_names)):
            acts = acts_dict[lname]
            class_means = np.zeros((num_classes, acts.shape[1]))
            for cls in range(num_classes):
                mask = labels_np == cls
                if mask.sum() > 0:
                    class_means[cls] = acts[mask].mean(axis=0)
            class_means = class_means[order]

            rdm = np.zeros((num_classes, num_classes))
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    d = cosine(class_means[i], class_means[j])
                    rdm[i, j] = d
                    rdm[j, i] = d

            im = axes[row, col].imshow(rdm, cmap="viridis",
                                        interpolation="nearest")
            for b in boundaries:
                axes[row, col].axhline(b, color="white", linewidth=0.3,
                                        alpha=0.8)
                axes[row, col].axvline(b, color="white", linewidth=0.3,
                                        alpha=0.8)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(row_label, fontsize=12,
                                           fontweight="bold")
            if row == 0:
                axes[row, col].set_title(dname)

    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Dissimilarity (1 - cosine sim)")
    fig.suptitle("RDMs: Untrained vs Trained", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "cmp_rdm_layers")


# ── Comparison: Speaker Invariance ────────────────────────────────────────
def fig_cmp_speaker_invariance(model_u, model_t, val_X, val_specs, val_y,
                                val_paths, device):
    print("[cmp 6] Speaker invariance comparison...")
    labels_np = val_y.numpy()
    speakers = [extract_speaker(p) for p in val_paths]
    unique_speakers = sorted(set(speakers))
    unique_words = sorted(set(labels_np))
    speaker_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    speaker_ids = np.array([speaker_to_idx[s] for s in speakers])

    input_acts = val_specs.numpy().reshape(len(val_specs), -1)

    def compute_ratios(acts_dict):
        all_layers = {"Input": input_acts}
        all_layers.update(acts_dict)
        layer_names = ["Input", "block1", "block2", "gap"]
        wvs, svs = [], []
        for lname in layer_names:
            acts = all_layers[lname]
            total_var = np.var(acts, axis=0).mean()
            word_means = np.array([acts[labels_np == w].mean(axis=0)
                                   for w in unique_words])
            speaker_means = np.array([acts[speaker_ids == s].mean(axis=0)
                                      for s in range(len(unique_speakers))])
            wv = np.var(word_means, axis=0).mean() / max(total_var, 1e-10)
            sv = np.var(speaker_means, axis=0).mean() / max(total_var, 1e-10)
            wvs.append(wv)
            svs.append(sv)
        return wvs, svs

    acts_u = extract_layer_activations(model_u, val_X, device)
    acts_t = extract_layer_activations(model_t, val_X, device)
    wv_u, sv_u = compute_ratios(acts_u)
    wv_t, sv_t = compute_ratios(acts_t)

    display_names = ["Input", "Block 1", "Block 2", "GAP"]
    x = np.arange(len(display_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, wv_u, width, label="Untrained: word var",
           color="#aec7e8", edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5*width, sv_u, width, label="Untrained: speaker var",
           color="#ffbb78", edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5*width, wv_t, width, label="Trained: word var",
           color="#1f77b4", edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5*width, sv_t, width, label="Trained: speaker var",
           color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel("Fraction of total variance")
    ax.set_xlabel("Layer")
    ax.set_title("Variance Decomposition: Untrained vs Trained")
    ax.legend(fontsize=8, ncol=2)

    # Add ratio annotations
    for i in range(len(display_names)):
        ratio_u = wv_u[i] / max(sv_u[i], 1e-10)
        ratio_t = wv_t[i] / max(sv_t[i], 1e-10)
        y_max = max(wv_u[i], sv_u[i], wv_t[i], sv_t[i])
        ax.text(x[i], y_max + 0.01,
                f"U:{ratio_u:.1f}  T:{ratio_t:.1f}",
                ha="center", fontsize=7, fontstyle="italic", color="#555555")

    savefig(fig, "cmp_speaker_invariance")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    (model_t, val_X, val_specs, val_y, val_paths,
     idx_to_label, label_to_idx, device) = load_data_and_model()

    # Create untrained model (same architecture, random weights)
    num_classes = len(label_to_idx)
    torch.manual_seed(0)  # reproducible random init
    model_u = WordResNet(num_classes).to(device)
    model_u.eval()
    print("Created untrained model (random weights)")

    # ── Trained-only figures ──
    print("\n=== Trained-model figures ===")
    cm_raw, _, _ = fig1a_confusion_matrix(model_t, val_X, val_y,
                                          idx_to_label, device)
    fig1b_top_confusions(cm_raw, idx_to_label)
    fig2_umap(model_t, val_X, val_y, val_paths, idx_to_label, device)
    fig3_filters(model_t)
    fig4_gradcam(model_t, val_X, val_y, idx_to_label, label_to_idx, device)
    print("[--] Extracting trained layer activations...")
    layer_acts_t = extract_layer_activations(model_t, val_X, device)
    rdms = fig5a_rdm(layer_acts_t, val_specs, val_y, idx_to_label)
    fig5b_dendrograms(rdms, idx_to_label)
    fig6_speaker_invariance(layer_acts_t, val_specs, val_y, val_paths)

    # ── Before/After comparison figures ──
    print("\n=== Untrained vs Trained comparison figures ===")
    fig_cmp_confusion(model_u, model_t, val_X, val_y, idx_to_label, device)
    fig_cmp_umap(model_u, model_t, val_X, val_y, val_paths, idx_to_label,
                 device)
    fig_cmp_filters(model_u, model_t)
    fig_cmp_gradcam(model_u, model_t, val_X, val_y, idx_to_label,
                    label_to_idx, device)
    fig_cmp_rdm(model_u, model_t, val_X, val_specs, val_y, idx_to_label,
                device)
    fig_cmp_speaker_invariance(model_u, model_t, val_X, val_specs, val_y,
                                val_paths, device)

    print(f"\nAll analyses complete. Results saved to {OUT_DIR}/")
    print(f"Files: {len(os.listdir(OUT_DIR))} outputs (PNG + PDF)")


if __name__ == "__main__":
    main()
