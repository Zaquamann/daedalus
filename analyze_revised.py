#!/usr/bin/env python3
"""Trained vs untrained model comparisons: UMAP, confusions, probes, Grad-CAM, RDMs, saliency."""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

# Audio params (from preprocess.py)
SAMPLE_RATE = 44100
N_MELS = 80
MAX_DURATION_S = 1.0

# ── Category definitions for UMAP colorings ───────────────────────────────
SEMANTIC = {
    "Numbers": [
        "ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN",
        "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN",
        "FIFTEEN", "SIXTEEN", "SEVENTEEN", "EIGHTEEN", "NINETEEN", "TWENTY",
        "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY",
        "HUNDRED", "THOUSAND", "MILLION",
    ],
    "Months": [
        "JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
        "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER",
    ],
    "Days": [
        "MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY",
        "SUNDAY",
    ],
    "Commands": [
        "CLICK", "SCROLL", "COPY", "PASTE", "DELETE", "SAVE", "OPEN",
        "CLOSE", "CUT", "PRINT", "SEARCH", "SELECT", "CANCEL", "REFRESH",
        "MINIMIZE", "EDIT", "INSERT", "EXPORT", "IMPORT", "ADD", "SEND",
        "REPLY", "FORWARD", "ATTACH", "CHECK", "SET", "RUN", "STOP",
        "START", "BEGIN", "END", "MOVE", "TURN", "CHANGE", "VIEW", "TYPE",
        "RECORD", "MUTE", "DIAL", "CALL", "GO", "HELP", "OK",
    ],
    "Family": ["MUM", "DAD", "BROTHER", "SISTER", "SON", "DAUGHTER",
               "HUSBAND", "WIFE"],
    "Media/Tech": [
        "PLAY", "PAUSE", "BROWSER", "CAMERA", "CALCULATOR", "CALENDAR",
        "FACEBOOK", "YOUTUBE", "GOOGLE", "COMPUTER", "MOUSE", "SCREEN",
        "VOLUME", "MUSIC", "SOUND", "FLASH", "PLAYER",
    ],
}

ONSET_CLASS = {
    "Plosives": [
        "P.M.", "PAGE", "PANEL", "PARAGRAPH", "PAST", "PASTE", "PAUSE",
        "PICTURE", "PICTURES", "PLAY", "PLAYER", "PREVIOUS", "PRINT",
        "BACK", "BEGIN", "BOLD", "BROTHER", "BROWSER",
        "TAB", "TEN", "TEXT", "TIME", "TO", "TOMORROW", "TUESDAY",
        "TURN", "TWELVE", "TWENTY", "TWO", "TYPE",
        "DAD", "DAUGHTER", "DECEMBER", "DELETE", "DIAL", "DOCUMENT",
        "DOCUMENTS", "DOUBLE", "DOWN", "DOWNLOADS",
        "CALCULATOR", "CALENDAR", "CALL", "CAMERA", "CANCEL", "CLICK",
        "CLOSE", "COMPUTER", "CONTROL", "COPY", "CUT", "QUARTER",
        "GO", "GOOGLE",
    ],
    "Fricatives": [
        "FACEBOOK", "FAVORITES", "FEBRUARY", "FIFTEEN", "FIFTY", "FILE",
        "FIVE", "FLASH", "FONT", "FORTY", "FORWARD", "FOUR", "FOURTEEN",
        "FRIDAY", "FULL",
        "VIDEOS", "VIEW", "VOLUME",
        "THIRTEEN", "THIRTY", "THOUSAND", "THREE", "THURSDAY",
        "SATURDAY", "SAVE", "SCREEN", "SCROLL", "SEARCH", "SELECT",
        "SEND", "SENTENCE", "SEPTEMBER", "SET", "SEVEN", "SEVENTEEN",
        "SEVENTY", "SIX", "SIXTEEN", "SIXTY", "SLEEP", "SON", "SOUND",
        "START", "STOP", "SUBJECT", "SUNDAY",
        "ZERO", "SHUT",
        "HALF", "HELP", "HIBERNATE", "HOME", "HOMEPAGE", "HUNDRED",
        "HUSBAND",
    ],
    "Nasals": [
        "MARCH", "MAY", "MESSAGE", "MILLION", "MINIMIZE", "MONDAY",
        "MOUSE", "MOVE", "MUM", "MUSIC", "MUTE", "MY",
        "NEW", "NEXT", "NINE", "NINETEEN", "NINETY", "NOVEMBER",
    ],
    "Liquids & Glides": [
        "LAST", "LEFT", "LINE",
        "RECORD", "REFRESH", "REMINDER", "REPLY", "RESTART", "RIGHT", "RUN",
        "WEDNESDAY", "WIFE", "WINDOW", "WORD", "ONE",
        "YESTERDAY", "YOUTUBE",
    ],
    "Vowel-Initial": [
        "ADD", "AS", "ALARM", "APPOINTMENT", "APRIL", "ATTACH",
        "ATTACHMENT", "A.M.", "EIGHT", "EIGHTEEN", "EIGHTY",
        "EDIT", "E-MAIL", "END", "EXPORT",
        "ELEVEN", "IMPORT", "INBOX", "INSERT", "ITALICS",
        "ALL", "AUGUST", "OCTOBER", "OFF", "OK", "ON", "OPEN", "OUTBOX",
        "UP", "MMS", "SMS",
    ],
    "Affricates": [
        "CHANGE", "CHECK", "JANUARY", "JULY", "JUNE",
    ],
}

FIRST_SOUND = {
    "S-": [
        "SATURDAY", "SAVE", "SCREEN", "SCROLL", "SEARCH", "SELECT",
        "SEND", "SENTENCE", "SEPTEMBER", "SET", "SEVEN", "SEVENTEEN",
        "SEVENTY", "SIX", "SIXTEEN", "SIXTY", "SLEEP", "SMS", "SON",
        "SOUND", "START", "STOP", "SUBJECT", "SUNDAY",
    ],
    "K/C-": [
        "CALCULATOR", "CALENDAR", "CALL", "CAMERA", "CANCEL", "CLICK",
        "CLOSE", "COMPUTER", "CONTROL", "COPY", "CUT", "QUARTER",
    ],
    "T-": [
        "TAB", "TEN", "TEXT", "THIRTEEN", "THIRTY", "THOUSAND", "THREE",
        "THURSDAY", "TIME", "TO", "TOMORROW", "TUESDAY", "TURN",
        "TWELVE", "TWENTY", "TWO", "TYPE",
    ],
    "F-": [
        "FACEBOOK", "FAVORITES", "FEBRUARY", "FIFTEEN", "FIFTY", "FILE",
        "FIVE", "FLASH", "FONT", "FORTY", "FORWARD", "FOUR", "FOURTEEN",
        "FRIDAY", "FULL",
    ],
    "M-": [
        "MARCH", "MAY", "MESSAGE", "MILLION", "MINIMIZE", "MMS",
        "MONDAY", "MOUSE", "MOVE", "MUM", "MUSIC", "MUTE", "MY",
    ],
    "P/B-": [
        "P.M.", "PAGE", "PANEL", "PARAGRAPH", "PAST", "PASTE", "PAUSE",
        "PICTURE", "PICTURES", "PLAY", "PLAYER", "PREVIOUS", "PRINT",
        "BACK", "BEGIN", "BOLD", "BROTHER", "BROWSER",
    ],
    "D-": [
        "DAD", "DAUGHTER", "DECEMBER", "DELETE", "DIAL", "DOCUMENT",
        "DOCUMENTS", "DOUBLE", "DOWN", "DOWNLOADS",
    ],
    "R-": [
        "RECORD", "REFRESH", "REMINDER", "REPLY", "RESTART", "RIGHT",
        "RUN",
    ],
    "N-": ["NEW", "NEXT", "NINE", "NINETEEN", "NINETY", "NOVEMBER"],
    "W/Y-": [
        "WEDNESDAY", "WIFE", "WINDOW", "WORD", "ONE",
        "YESTERDAY", "YOUTUBE",
    ],
    "H/SH-": [
        "HALF", "HELP", "HIBERNATE", "HOME", "HOMEPAGE", "HUNDRED",
        "HUSBAND", "SHUT",
    ],
    "Vowel-": [
        "ADD", "A.M.", "ALARM", "ALL", "APPOINTMENT", "APRIL", "AS",
        "ATTACH", "ATTACHMENT", "AUGUST",
        "EDIT", "E-MAIL", "EIGHT", "EIGHTEEN", "EIGHTY", "ELEVEN",
        "END", "EXPORT",
        "IMPORT", "INBOX", "INSERT", "ITALICS",
        "OCTOBER", "OFF", "OK", "ON", "OPEN", "OUTBOX", "UP",
    ],
    "Other": [
        "CHANGE", "CHECK", "JANUARY", "JULY", "JUNE",
        "GO", "GOOGLE", "LAST", "LEFT", "LINE",
        "VIDEOS", "VIEW", "VOLUME", "ZERO",
    ],
}


def _categorize(word, cat_dict):
    """Look up word in a category dict. Returns 'Other' if not found."""
    for cat, words in cat_dict.items():
        if word in words:
            return cat
    return "Other"


# ── Acoustic similarity clustering (raw spectrograms) ──────────────────────
N_CLUSTERS = 12


def compute_cluster_info(all_specs, all_labels, val_y, idx_to_label):
    """Cluster words by acoustic similarity (PCA + Ward's on class-mean spectrograms)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from scipy.cluster.hierarchy import linkage, fcluster

    all_labels_np = all_labels.numpy()
    val_labels_np = val_y.numpy()
    num_classes = len(idx_to_label)

    # Per-class mean spectrograms from ALL data (train + val)
    spec_flat_dim = all_specs.shape[1] * all_specs.shape[2]  # 80*99 = 7920
    class_means = np.zeros((num_classes, spec_flat_dim))
    for c in range(num_classes):
        mask = all_labels_np == c
        if mask.sum() > 0:
            class_means[c] = all_specs[mask].numpy().reshape(mask.sum(), -1).mean(axis=0)

    # Z-score normalize each feature across classes
    scaler = StandardScaler()
    centroids_norm = scaler.fit_transform(class_means)

    # PCA to 50 dims — reduces noise, helps Ward's
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    centroids_pca = pca.fit_transform(centroids_norm)

    # Ward's hierarchical clustering (Euclidean, produces balanced clusters)
    Z = linkage(centroids_pca, method="ward")
    cluster_labels = fcluster(Z, t=N_CLUSTERS, criterion="maxclust") - 1  # 0-indexed

    # Map each val sample to its word's cluster
    sample_clusters = cluster_labels[val_labels_np]

    # Build cluster members and names
    cluster_members = defaultdict(list)
    for c in range(num_classes):
        cname = f"C{cluster_labels[c]}"
        cluster_members[cname].append(idx_to_label[c])
    for cname in cluster_members:
        cluster_members[cname].sort()

    cat_order = [f"C{i}" for i in range(N_CLUSTERS)]
    cmap = plt.get_cmap("tab20", N_CLUSTERS)
    cat_colors = {f"C{i}": cmap(i) for i in range(N_CLUSTERS)}

    # Build word -> cluster lookup
    word_to_cluster = {}
    for c in range(num_classes):
        word_to_cluster[idx_to_label[c]] = f"C{cluster_labels[c]}"

    # Per-sample cluster name array (val set only)
    sample_cluster_names = np.array([f"C{sc}" for sc in sample_clusters])

    print(f"  Acoustic clustering: {N_CLUSTERS} clusters on {num_classes} "
          f"class mean spectrograms ({all_specs.shape[0]} total samples)")
    for cname in cat_order:
        words = cluster_members.get(cname, [])
        preview = ", ".join(words[:6])
        if len(words) > 6:
            preview += f", ... ({len(words)} total)"
        if words:
            print(f"    {cname}: {preview}")

    return {
        "cat_order": cat_order,
        "cat_colors": cat_colors,
        "cluster_members": dict(cluster_members),
        "word_to_cluster": word_to_cluster,
        "sample_cluster_names": sample_cluster_names,
        "cluster_labels": cluster_labels,  # per-class
    }


def savefig(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


def mel_bin_to_hz(mel_bin, n_mels=80, fmin=0.0, fmax=22050.0):
    """Convert mel bin index to Hz for axis labels."""
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_val = mel_min + (mel_max - mel_min) * mel_bin / (n_mels - 1)
    return 700.0 * (10.0 ** (mel_val / 2595.0) - 1.0)


def extract_speaker(path):
    parts = path.replace("\\", "/").split("/")
    for p in parts:
        if p.startswith("speaker-"):
            return p
    return "unknown"


# ── Data / model loading ───────────────────────────────────────────────────
def load_all():
    data = torch.load(DATA_PATH, weights_only=False)
    spectrograms = data["spectrograms"]
    labels = data["labels"]
    label_to_idx = data["label_to_idx"]
    idx_to_label = data["idx_to_label"]
    file_paths = data["file_paths"]
    num_classes = len(label_to_idx)

    _, val_idx = stratified_split(labels, TEST_SIZE, RANDOM_SEED)
    val_X = spectrograms[val_idx].unsqueeze(1)  # (N, 1, 80, 99)
    val_specs = spectrograms[val_idx]             # (N, 80, 99)
    val_y = labels[val_idx]
    val_paths = [file_paths[i] for i in val_idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Trained model
    model_t = WordResNet(num_classes)
    ckpt = torch.load(MODEL_PATH, weights_only=False, map_location=device)
    model_t.load_state_dict(ckpt["model_state_dict"])
    model_t.to(device).eval()

    # Untrained model (random init, reproducible)
    torch.manual_seed(0)
    model_u = WordResNet(num_classes).to(device)
    model_u.eval()

    print(f"Loaded {len(val_y)} val samples, {num_classes} classes")
    print(f"Trained model: epoch {ckpt['epoch']}, val acc {ckpt['best_val_acc']:.1%}")
    print(f"Device: {device}")

    return (model_t, model_u, val_X, val_specs, val_y, val_paths,
            idx_to_label, label_to_idx, device,
            spectrograms, labels)  # full dataset for acoustic clustering


def batch_forward(model, val_X, device, batch_size=64):
    all_preds, all_logits = [], []
    with torch.no_grad():
        for i in range(0, len(val_X), batch_size):
            batch = val_X[i:i+batch_size].to(device)
            logits = model(batch)
            all_logits.append(logits.cpu())
            all_preds.append(logits.argmax(1).cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_logits).numpy()


def extract_layer_activations(model, val_X, device, batch_size=64):
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
            model(val_X[i:i+batch_size].to(device))
    for h in handles:
        h.remove()
    return {k: torch.cat(v).flatten(1).numpy() for k, v in layer_acts.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1: UMAP — Trained vs Untrained
# ═══════════════════════════════════════════════════════════════════════════
def _plot_umap_coloring(emb_u_2d, emb_t_2d, sample_cats, cat_order,
                        cat_colors, suptitle, filename, method,
                        text_box=None):
    """Plot one UMAP coloring: untrained vs trained side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, emb_2d, title in zip(axes, [emb_u_2d, emb_t_2d],
                                  ["Untrained (random weights)", "Trained"]):
        for cat in cat_order:
            mask = sample_cats == cat
            if not mask.any():
                continue
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[cat_colors[cat]], s=12, alpha=0.6, label=cat,
                       edgecolors="none")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8,
                   markerscale=2, framealpha=0.9)
    if text_box:
        fig.text(0.02, -0.02, text_box, fontsize=5.5, fontfamily="monospace",
                 va="top", transform=fig.transFigure)
    fig.suptitle(f"{method} of GAP Embeddings: {suptitle}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{filename}.png"))
    plt.close(fig)
    print(f"  Saved {filename}.png")


def analysis1_umap(model_t, model_u, val_X, val_y, val_paths,
                   idx_to_label, device, cinfo):
    print("\n[1/6] UMAP: all 5 colorings + speaker...")
    from sklearn.cluster import KMeans

    labels_np = val_y.numpy()
    word_labels = [idx_to_label[l] for l in labels_np]
    num_classes = len(idx_to_label)

    # Extract GAP embeddings from both models
    def get_gap_emb(mdl):
        embs = []
        def hook(module, inp, out):
            embs.append(out.flatten(1).cpu())
        h = mdl.gap.register_forward_hook(hook)
        with torch.no_grad():
            for i in range(0, len(val_X), 64):
                mdl(val_X[i:i+64].to(device))
        h.remove()
        return torch.cat(embs).numpy()

    emb_u = get_gap_emb(model_u)
    emb_t = get_gap_emb(model_t)

    try:
        import umap
        method = "UMAP"
        def reduce(e):
            return umap.UMAP(n_components=2, random_state=RANDOM_SEED,
                             n_neighbors=15, min_dist=0.1).fit_transform(e)
    except ImportError:
        from sklearn.manifold import TSNE
        method = "t-SNE"
        def reduce(e):
            return TSNE(n_components=2, perplexity=30,
                        random_state=RANDOM_SEED, max_iter=1000).fit_transform(e)

    emb_u_2d = reduce(emb_u)
    emb_t_2d = reduce(emb_t)

    # ── Figure 1: Semantic categories ──
    sem_colors = {"Numbers": "#1f77b4", "Months": "#2ca02c", "Days": "#ff7f0e",
                  "Commands": "#d62728", "Family": "#9467bd",
                  "Media/Tech": "#17becf", "Other": "#999999"}
    sem_order = ["Numbers", "Months", "Days", "Commands", "Family",
                 "Media/Tech", "Other"]
    sem_cats = np.array([_categorize(w, SEMANTIC) for w in word_labels])
    _plot_umap_coloring(emb_u_2d, emb_t_2d, sem_cats, sem_order, sem_colors,
                        "Colored by Semantic Category", "umap_semantic", method)

    # ── Figure 2: Phonetic onset class (manner of articulation, 6 groups) ──
    oc_colors = {"Plosives": "#e41a1c", "Fricatives": "#377eb8",
                 "Nasals": "#4daf4a", "Liquids & Glides": "#ff7f00",
                 "Vowel-Initial": "#984ea3", "Affricates": "#a65628",
                 "Other": "#999999"}
    oc_order = ["Plosives", "Fricatives", "Nasals", "Liquids & Glides",
                "Vowel-Initial", "Affricates", "Other"]
    oc_cats = np.array([_categorize(w, ONSET_CLASS) for w in word_labels])
    _plot_umap_coloring(emb_u_2d, emb_t_2d, oc_cats, oc_order, oc_colors,
                        "Colored by Onset Phoneme Class", "umap_onset_class",
                        method)

    # ── Figure 3: First sound (13 groups) ──
    fs_cmap = plt.get_cmap("tab20", 13)
    fs_order = list(FIRST_SOUND.keys())
    fs_colors = {cat: fs_cmap(i) for i, cat in enumerate(fs_order)}
    fs_cats = np.array([_categorize(w, FIRST_SOUND) for w in word_labels])
    _plot_umap_coloring(emb_u_2d, emb_t_2d, fs_cats, fs_order, fs_colors,
                        "Colored by First Sound", "umap_first_sound", method)

    # ── Figure 4: K-means on trained model GAP centroids (k=10) ──
    class_centroids = np.zeros((num_classes, emb_t.shape[1]))
    for c in range(num_classes):
        mask = labels_np == c
        if mask.sum() > 0:
            class_centroids[c] = emb_t[mask].mean(axis=0)
    kmeans = KMeans(n_clusters=10, random_state=RANDOM_SEED, n_init=10)
    km_class_labels = kmeans.fit_predict(class_centroids)
    km_sample_cats = np.array([f"C{km_class_labels[l]}" for l in labels_np])
    km_order = [f"C{i}" for i in range(10)]
    km_cmap = plt.get_cmap("tab10", 10)
    km_colors = {f"C{i}": km_cmap(i) for i in range(10)}
    # Build text box with cluster members
    km_members = defaultdict(list)
    for c in range(num_classes):
        km_members[f"C{km_class_labels[c]}"].append(idx_to_label[c])
    km_text = "K-means clusters on trained GAP centroids (k=10):\n"
    for cat in km_order:
        words = sorted(km_members.get(cat, []))
        if len(words) > 6:
            km_text += f"  {cat}: {', '.join(words[:6])}, ... ({len(words)} total)\n"
        elif words:
            km_text += f"  {cat}: {', '.join(words)}\n"
    _plot_umap_coloring(emb_u_2d, emb_t_2d, km_sample_cats, km_order,
                        km_colors, "Colored by K-Means on Model Embeddings",
                        "umap_kmeans", method, text_box=km_text)

    # ── Figure 5: Acoustic similarity from raw spectrograms ──
    ac_cats = cinfo["sample_cluster_names"]
    ac_order = cinfo["cat_order"]
    ac_colors = cinfo["cat_colors"]
    ac_members = cinfo["cluster_members"]
    ac_text = "Acoustic clusters (Ward's on z-scored PCA'd mean spectrograms, k=12):\n"
    for cat in ac_order:
        words = ac_members.get(cat, [])
        if len(words) > 6:
            ac_text += f"  {cat}: {', '.join(words[:6])}, ... ({len(words)} total)\n"
        elif words:
            ac_text += f"  {cat}: {', '.join(words)}\n"
    _plot_umap_coloring(emb_u_2d, emb_t_2d, ac_cats, ac_order, ac_colors,
                        "Colored by Acoustic Similarity (Raw Spectrograms)",
                        "umap_acoustic", method, text_box=ac_text)

    # ── Speaker figure (unchanged) ──
    speakers = np.array([extract_speaker(p) for p in val_paths])
    unique_speakers = sorted(set(speakers))
    speaker_cmap = plt.get_cmap("tab20", len(unique_speakers))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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
    fig.suptitle(f"{method} of GAP Embeddings by Speaker: Untrained vs Trained",
                 fontweight="bold")
    plt.tight_layout()
    savefig(fig, "rev_umap_by_speaker")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2: Top Confused Pairs + Category-Level Matrix
# ═══════════════════════════════════════════════════════════════════════════
def analysis2_confusions(model_t, val_X, val_y, idx_to_label, device, cinfo):
    print("[2/6] Confusion analysis...")
    from sklearn.metrics import confusion_matrix

    preds, _ = batch_forward(model_t, val_X, device)
    true = val_y.numpy()
    num_classes = len(idx_to_label)
    cm = confusion_matrix(true, preds, labels=np.arange(num_classes))
    word_to_cluster = cinfo["word_to_cluster"]
    cat_order = cinfo["cat_order"]

    # ── Top-20 confused pairs ──
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    row_sums = cm.sum(axis=1)

    confused = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm_off[i, j] > 0:
                rate = cm_off[i, j] / max(row_sums[i], 1)
                confused.append((rate, cm_off[i, j], idx_to_label[i],
                                 idx_to_label[j]))
    confused.sort(reverse=True)
    top20 = confused[:20]

    ANNOTATIONS = {
        ("TO", "TWO"): "homophones /tuː/",
        ("TWO", "TO"): "homophones /tuː/",
        ("SEVENTEEN", "SEVENTY"): "shared prefix /sɛv-/",
        ("SEVENTY", "SEVENTEEN"): "shared prefix /sɛv-/",
        ("LINE", "NINE"): "rhyme /-aɪn/",
        ("NINE", "LINE"): "rhyme /-aɪn/",
        ("FORTY", "FOURTEEN"): "shared prefix /fɔːr-/",
        ("FOURTEEN", "FORTY"): "shared prefix /fɔːr-/",
        ("THIRTY", "THIRTEEN"): "shared prefix /θɜːr-/",
        ("THIRTEEN", "THIRTY"): "shared prefix /θɜːr-/",
        ("FIFTEEN", "FIFTY"): "shared prefix /fɪf-/",
        ("FIFTY", "FIFTEEN"): "shared prefix /fɪf-/",
        ("NINETEEN", "NINETY"): "shared prefix /naɪn-/",
        ("NINETY", "NINETEEN"): "shared prefix /naɪn-/",
        ("FILE", "FIVE"): "shared onset /faɪ-/",
        ("FIVE", "FILE"): "shared onset /faɪ-/",
        ("NEW", "VIEW"): "rhyme /-juː/",
        ("VIEW", "NEW"): "rhyme /-juː/",
        ("TEN", "TURN"): "shared onset /t-/",
        ("TURN", "TEN"): "shared onset /t-/",
        ("HALF", "PAST"): "shared vowel /ɑː/",
        ("FOUR", "ALL"): "shared vowel /ɔː/",
        ("EIGHT", "PAGE"): "shared vowel /eɪ/",
        ("WINDOW", "MONDAY"): "shared /-ndə/ pattern",
        ("SHUT", "SON"): "shared vowel /ʌ/",
        ("SON", "SHUT"): "shared vowel /ʌ/",
        ("RUN", "ONE"): "shared vowel /ʌ/, near-rhyme",
        ("ONE", "RUN"): "shared vowel /ʌ/, near-rhyme",
        ("PLAY", "MAY"): "rhyme /-eɪ/",
        ("MAY", "PLAY"): "rhyme /-eɪ/",
        ("ON", "ALL"): "similar vowels /ɒ/ vs /ɔː/",
        ("ALL", "ON"): "similar vowels /ɒ/ vs /ɔː/",
    }

    labels_bar = []
    rates_bar = []
    colors_bar = []
    annots_bar = []
    for rate, count, tl, pl in top20:
        labels_bar.append(f"{tl} -> {pl} ({count}, {rate:.0%})")
        rates_bar.append(rate * 100)
        annot = ANNOTATIONS.get((tl, pl), "")
        annots_bar.append(annot)
        colors_bar.append("#2ca02c" if annot else "#d62728")

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels_bar))[::-1]
    bars = ax.barh(y_pos, rates_bar, color=colors_bar, edgecolor="black",
                   linewidth=0.5, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_bar, fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Confusion rate (%)")
    ax.set_title("Top-20 Most Confused Word Pairs (Trained Model)")

    for i, (bar, annot) in enumerate(zip(bars, annots_bar)):
        if annot:
            ax.text(bar.get_width() + 0.2, y_pos[i], annot,
                    va="center", fontsize=6.5, fontstyle="italic",
                    color="#555555")

    legend_elements = [
        Line2D([0], [0], color="#2ca02c", lw=6, label="Phonetically expected"),
        Line2D([0], [0], color="#d62728", lw=6, label="No obvious phonetic link"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    savefig(fig, "rev_top_confusions")

    # ── Cluster-level confusion matrix ──
    word_cats_true = [word_to_cluster.get(idx_to_label[t], cat_order[0])
                      for t in true]
    word_cats_pred = [word_to_cluster.get(idx_to_label[p], cat_order[0])
                      for p in preds]

    cat_to_idx = {c: i for i, c in enumerate(cat_order)}
    n_cats = len(cat_order)
    cat_cm = np.zeros((n_cats, n_cats), dtype=int)
    for tc, pc in zip(word_cats_true, word_cats_pred):
        cat_cm[cat_to_idx[tc], cat_to_idx[pc]] += 1

    # Row-normalize
    cat_row_sums = cat_cm.sum(axis=1, keepdims=True)
    cat_cm_norm = cat_cm / np.maximum(cat_row_sums, 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cat_cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_cats))
    ax.set_xticklabels(cat_order, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(cat_order, fontsize=9)
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True cluster")
    ax.set_title("Cluster-Level Confusion Matrix (row-normalized)\n"
                 "Clusters from acoustic similarity of raw spectrograms")

    # Annotate each cell
    for i in range(n_cats):
        for j in range(n_cats):
            val = cat_cm_norm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold" if i == j else "normal")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Proportion")
    savefig(fig, "rev_category_confusion")

    return cm


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 3: Linear Probes
# ═══════════════════════════════════════════════════════════════════════════
def analysis3_linear_probes(model_t, model_u, val_X, val_specs, val_y,
                            val_paths, device):
    print("[3/6] Linear probes...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    labels_np = val_y.numpy()
    speakers = [extract_speaker(p) for p in val_paths]
    unique_speakers = sorted(set(speakers))
    spk_to_idx = {s: i for i, s in enumerate(unique_speakers)}
    speaker_ids = np.array([spk_to_idx[s] for s in speakers])

    input_acts = val_specs.numpy().reshape(len(val_specs), -1)

    acts_t = extract_layer_activations(model_t, val_X, device)
    acts_u = extract_layer_activations(model_u, val_X, device)

    layer_names = ["Input", "block1", "block2", "gap"]
    display_names = ["Input", "Block 1", "Block 2", "GAP"]

    results = {}  # {(model_label, layer, target): (mean_acc, std_acc)}

    for model_label, acts_dict in [("Untrained", acts_u), ("Trained", acts_t)]:
        all_layers = {"Input": input_acts}
        all_layers.update(acts_dict)

        for lname, dname in zip(layer_names, display_names):
            X = all_layers[lname]
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Reduce dimensionality for speed
            if X_scaled.shape[1] > 500:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(200, X_scaled.shape[1]),
                          random_state=RANDOM_SEED)
                X_scaled = pca.fit_transform(X_scaled)

            for target_name, y_target in [("Word", labels_np),
                                           ("Speaker", speaker_ids)]:
                clf = LogisticRegression(max_iter=200, solver="saga",
                                        random_state=RANDOM_SEED,
                                        tol=1e-2)
                scores = cross_val_score(clf, X_scaled, y_target, cv=3,
                                         scoring="accuracy")
                results[(model_label, dname, target_name)] = (
                    scores.mean(), scores.std())
                print(f"  {model_label} {dname} {target_name}: "
                      f"{scores.mean():.3f} +/- {scores.std():.3f}")

    # Plot
    x = np.arange(len(display_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_configs = [
        (-1.5*width, "Untrained", "Word", "#aec7e8", "//"),
        (-0.5*width, "Untrained", "Speaker", "#ffbb78", "//"),
        (0.5*width, "Trained", "Word", "#1f77b4", None),
        (1.5*width, "Trained", "Speaker", "#ff7f0e", None),
    ]

    for offset, mlabel, target, color, hatch in bar_configs:
        means = [results[(mlabel, dn, target)][0] for dn in display_names]
        stds = [results[(mlabel, dn, target)][1] for dn in display_names]
        label = f"{mlabel}: {target}"
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, hatch=hatch, edgecolor="black", linewidth=0.5,
               capsize=3, error_kw={"linewidth": 0.8})

    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylabel("3-fold CV accuracy")
    ax.set_xlabel("Layer")
    ax.set_title("Linear Probe Accuracy: Word vs Speaker Classification")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.set_ylim(0, 1.05)
    ax.axhline(1/181, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.text(0.02, 1/181 + 0.01, "chance (word: 0.55%)", fontsize=7,
            color="gray", transform=ax.get_yaxis_transform())
    ax.axhline(1/25, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.text(0.02, 1/25 + 0.01, "chance (speaker: 4.0%)", fontsize=7,
            color="gray", transform=ax.get_yaxis_transform())

    savefig(fig, "rev_linear_probes")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 4: Grad-CAM on Confused Word Pairs
# ═══════════════════════════════════════════════════════════════════════════
def analysis4_gradcam_confusions(model_t, val_X, val_y, idx_to_label,
                                  label_to_idx, device):
    print("[4/6] Grad-CAM on confused pairs...")

    pairs = [
        ("SEVENTEEN", "SEVENTY"),
        ("FIVE", "NINE"),
        ("TO", "TWO"),
    ]

    labels_np = val_y.numpy()
    n_time = val_X.shape[3]  # 99 frames
    time_axis = np.linspace(0, MAX_DURATION_S, n_time)

    # Mel bin center frequencies
    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + (SAMPLE_RATE / 2.0) / 700.0)
    mel_centers = np.linspace(mel_min, mel_max, N_MELS)
    freq_axis = 700.0 * (10.0 ** (mel_centers / 2595.0) - 1.0)
    freq_ticks_hz = [100, 500, 1000, 2000, 5000, 10000, 20000]
    freq_ticks_bins = [np.argmin(np.abs(freq_axis - f)) for f in freq_ticks_hz
                       if f <= freq_axis[-1]]
    freq_tick_labels = [f"{freq_ticks_hz[i]/1000:.0f}k"
                        if freq_ticks_hz[i] >= 1000
                        else str(freq_ticks_hz[i])
                        for i in range(len(freq_ticks_bins))]

    fig, axes = plt.subplots(len(pairs), 4, figsize=(12, 3 * len(pairs)))

    for row, (word_a, word_b) in enumerate(pairs):
        for col_offset, word in enumerate([word_a, word_b]):
            if word not in label_to_idx:
                continue
            target_idx = label_to_idx[word]
            partner_idx = label_to_idx.get(word_b if col_offset == 0 else word_a)
            matches = np.where(labels_np == target_idx)[0]
            if len(matches) == 0:
                continue
            # Prefer correctly classified, or misclassified as partner
            sample_idx = matches[0]
            with torch.no_grad():
                for m in matches:
                    pred = model_t(val_X[m:m+1].to(device)).argmax(1).item()
                    if pred == target_idx:
                        sample_idx = m
                        break
                    if partner_idx is not None and pred == partner_idx:
                        sample_idx = m

            x = val_X[sample_idx:sample_idx+1].to(device).requires_grad_(True)
            activations, gradients = {}, {}

            def fwd_hook(module, inp, out):
                activations["b2"] = out
            def bwd_hook(module, grad_in, grad_out):
                gradients["b2"] = grad_out[0]

            h1 = model_t.block2.register_forward_hook(fwd_hook)
            h2 = model_t.block2.register_full_backward_hook(bwd_hook)
            logits = model_t(x)
            pred_class = logits.argmax(1).item()
            conf = torch.softmax(logits, dim=1)[0, pred_class].item()
            model_t.zero_grad()
            logits[0, pred_class].backward()
            h1.remove()
            h2.remove()

            act = activations["b2"].detach().cpu().numpy()[0]
            grad = gradients["b2"].detach().cpu().numpy()[0]
            weights = grad.mean(axis=(1, 2))
            cam = np.maximum(np.sum(weights[:, None, None] * act, axis=0), 0)
            if cam.max() > 0:
                cam = cam / cam.max()

            spec = val_X[sample_idx, 0].numpy()
            cam_resized = np.array(Image.fromarray(cam).resize(
                (spec.shape[1], spec.shape[0]), Image.BILINEAR))

            # Spectrogram column
            ax_spec = axes[row, col_offset * 2]
            ax_spec.imshow(spec, aspect="auto", origin="lower", cmap="viridis",
                           extent=[0, MAX_DURATION_S, 0, N_MELS])
            ax_spec.set_title(f"{word}", fontsize=10)
            ax_spec.set_xlabel("Time (s)")
            ax_spec.set_yticks(freq_ticks_bins)
            ax_spec.set_yticklabels(freq_tick_labels, fontsize=7)
            if col_offset == 0:
                ax_spec.set_ylabel("Frequency (Hz)")

            # Grad-CAM overlay column
            ax_cam = axes[row, col_offset * 2 + 1]
            ax_cam.imshow(spec, aspect="auto", origin="lower", cmap="viridis",
                          extent=[0, MAX_DURATION_S, 0, N_MELS])
            ax_cam.imshow(cam_resized, aspect="auto", origin="lower",
                          cmap="jet", alpha=0.4,
                          extent=[0, MAX_DURATION_S, 0, N_MELS])
            pred_word = idx_to_label[pred_class]
            ax_cam.set_title(f"Grad-CAM ({pred_word} {conf:.0%})", fontsize=9)
            ax_cam.set_xlabel("Time (s)")
            ax_cam.set_yticks(freq_ticks_bins)
            ax_cam.set_yticklabels(freq_tick_labels, fontsize=7)

    fig.suptitle("Grad-CAM on Commonly Confused Word Pairs\n"
                 "(correctly classified examples -- showing discriminative features)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "rev_gradcam_confusions")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 5: RDMs with Category Separation Index
# ═══════════════════════════════════════════════════════════════════════════
def analysis5_rdm(model_t, model_u, val_X, val_y, idx_to_label, device, cinfo):
    print("[5/6] RDMs with CSI...")
    from scipy.spatial.distance import cosine

    labels_np = val_y.numpy()
    num_classes = len(idx_to_label)
    word_to_cluster = cinfo["word_to_cluster"]
    cat_order = cinfo["cat_order"]

    # Sort by cluster
    class_cats = [(idx, idx_to_label[idx],
                   word_to_cluster.get(idx_to_label[idx], cat_order[0]))
                  for idx in sorted(idx_to_label.keys())]
    class_cats.sort(key=lambda x: (cat_order.index(x[2])
                                   if x[2] in cat_order else len(cat_order),
                                   x[1]))
    order = [c[0] for c in class_cats]
    cat_per_class = [c[2] for c in class_cats]

    # Category boundaries
    boundaries = []
    prev = None
    for i, cat in enumerate(cat_per_class):
        if cat != prev and prev is not None:
            boundaries.append(i - 0.5)
        prev = cat

    acts_u = extract_layer_activations(model_u, val_X, device)
    acts_t = extract_layer_activations(model_t, val_X, device)

    layer_names = ["block1", "block2", "gap"]
    display_names = ["Block 1", "Block 2", "GAP"]

    def compute_rdm(acts, labels, order):
        nc = len(order)
        class_means = np.zeros((nc, acts.shape[1]))
        for i, cls in enumerate(order):
            mask = labels == cls
            if mask.sum() > 0:
                class_means[i] = acts[mask].mean(axis=0)
        rdm = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(i + 1, nc):
                d = cosine(class_means[i], class_means[j])
                rdm[i, j] = d
                rdm[j, i] = d
        return rdm

    def compute_csi(rdm, cat_per_class):
        """Category Separation Index = mean between-cat / mean within-cat."""
        n = len(cat_per_class)
        within, between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if cat_per_class[i] == cat_per_class[j]:
                    within.append(rdm[i, j])
                else:
                    between.append(rdm[i, j])
        if not within or not between:
            return float("nan")
        return np.mean(between) / max(np.mean(within), 1e-10)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    all_rdms = []

    for row, (acts_dict, row_label) in enumerate(
            [(acts_u, "Untrained"), (acts_t, "Trained")]):
        for col, (lname, dname) in enumerate(zip(layer_names, display_names)):
            rdm = compute_rdm(acts_dict[lname], labels_np, order)
            csi = compute_csi(rdm, cat_per_class)
            all_rdms.append(rdm)

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
            title = dname if row == 0 else ""
            axes[row, col].set_title(title)
            # CSI annotation below
            axes[row, col].text(0.5, -0.08, f"CSI = {csi:.2f}",
                                 transform=axes[row, col].transAxes,
                                 ha="center", fontsize=9, fontstyle="italic")

    # Shared colorscale
    vmin = min(r.min() for r in all_rdms)
    vmax = max(r.max() for r in all_rdms)
    for row in range(2):
        for col in range(3):
            for im_obj in axes[row, col].images:
                im_obj.set_clim(vmin, vmax)

    cbar = fig.colorbar(axes[1, 2].images[0], ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label("Dissimilarity (1 - cosine sim)")

    fig.suptitle("Representational Dissimilarity Matrices: Untrained vs Trained",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "rev_rdm_comparison")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 6: Effective Receptive Field (Input Saliency)
# ═══════════════════════════════════════════════════════════════════════════
def analysis6_saliency(model_t, val_X, val_y, idx_to_label, label_to_idx,
                       device, cinfo):
    print("[6/6] Effective receptive field (input saliency)...")

    # Pick one word from each of the first 6 clusters
    cluster_members = cinfo["cluster_members"]
    cat_order = cinfo["cat_order"]
    target_words = {}
    for cat in cat_order[:6]:
        words = cluster_members.get(cat, [])
        # Pick the first word that exists in label_to_idx
        for w in words:
            if w in label_to_idx:
                target_words[cat] = w
                break

    labels_np = val_y.numpy()
    n_time = val_X.shape[3]
    time_axis = np.linspace(0, MAX_DURATION_S, n_time)

    mel_min = 2595.0 * np.log10(1.0 + 0.0 / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + (SAMPLE_RATE / 2.0) / 700.0)
    mel_centers = np.linspace(mel_min, mel_max, N_MELS)
    freq_axis = 700.0 * (10.0 ** (mel_centers / 2595.0) - 1.0)
    freq_ticks_hz = [100, 500, 1000, 2000, 5000, 10000, 20000]
    freq_ticks_bins = [np.argmin(np.abs(freq_axis - f)) for f in freq_ticks_hz
                       if f <= freq_axis[-1]]
    freq_tick_labels = [f"{freq_ticks_hz[i]/1000:.0f}k"
                        if freq_ticks_hz[i] >= 1000
                        else str(freq_ticks_hz[i])
                        for i in range(len(freq_ticks_bins))]

    examples = []
    for cat, word in target_words.items():
        if word not in label_to_idx:
            continue
        target_idx = label_to_idx[word]
        # Find a correctly classified example
        matches = np.where(labels_np == target_idx)[0]
        for m in matches:
            x = val_X[m:m+1].to(device)
            with torch.no_grad():
                pred = model_t(x).argmax(1).item()
            if pred == target_idx:
                examples.append((cat, word, m))
                break
        else:
            if len(matches) > 0:
                examples.append((cat, word, matches[0]))

    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    ax_flat = axes.flatten()

    for i, (cat, word, sample_idx) in enumerate(examples):
        if i >= nrows * ncols:
            break
        x = val_X[sample_idx:sample_idx+1].to(device).requires_grad_(True)
        logits = model_t(x)
        pred_class = logits.argmax(1).item()
        conf = torch.softmax(logits, dim=1)[0, pred_class].item()

        model_t.zero_grad()
        logits[0, pred_class].backward()

        grad = x.grad.detach().cpu().numpy()[0, 0]  # (80, 99)
        spec = val_X[sample_idx, 0].numpy()

        # |grad * input| saliency
        saliency = np.abs(grad * spec)
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        ax = ax_flat[i]
        ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis",
                  extent=[0, MAX_DURATION_S, 0, N_MELS])
        ax.imshow(saliency, aspect="auto", origin="lower", cmap="hot",
                  alpha=0.5, extent=[0, MAX_DURATION_S, 0, N_MELS])
        pred_word = idx_to_label[pred_class]
        ax.set_title(f"{word} [{cat}] ({conf:.0%})", fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_yticks(freq_ticks_bins)
        ax.set_yticklabels(freq_tick_labels, fontsize=7)
        if i % ncols == 0:
            ax.set_ylabel("Frequency (Hz)")

    for j in range(len(examples), len(ax_flat)):
        ax_flat[j].set_visible(False)

    fig.suptitle("Effective Receptive Field: Input Saliency (|grad x input|)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "rev_effective_receptive_field")




# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    (model_t, model_u, val_X, val_specs, val_y, val_paths,
     idx_to_label, label_to_idx, device,
     all_specs, all_labels) = load_all()

    # Compute acoustic similarity clusters from raw spectrograms
    cinfo = compute_cluster_info(all_specs, all_labels, val_y, idx_to_label)

    analysis1_umap(model_t, model_u, val_X, val_y, val_paths,
                   idx_to_label, device, cinfo)
    analysis2_confusions(model_t, val_X, val_y, idx_to_label, device, cinfo)
    analysis3_linear_probes(model_t, model_u, val_X, val_specs, val_y,
                            val_paths, device)
    analysis4_gradcam_confusions(model_t, val_X, val_y, idx_to_label,
                                 label_to_idx, device)
    analysis5_rdm(model_t, model_u, val_X, val_y, idx_to_label, device, cinfo)
    analysis6_saliency(model_t, val_X, val_y, idx_to_label, label_to_idx,
                       device, cinfo)

    all_figs = [f for f in os.listdir(OUT_DIR)
                if f.startswith("rev_") or f.startswith("umap_")]
    print(f"\nAll analyses complete. {len(all_figs)} files saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
