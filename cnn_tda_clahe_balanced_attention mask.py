import os
import random
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm
from gudhi import CubicalComplex
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# Optional: use skimage for CLAHE and attention (entropy, sobel)
try:
    from skimage import exposure
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    import warnings
    warnings.warn("scikit-image not found; falling back to simple histogram equalization.")

# ------------------ PARAMETERS ------------------
DATA_DIR = "/Users/satyaswaroopnune/Downloads/aptos/train_images/train_images"
CSV_PATH = "/Users/satyaswaroopnune/Downloads/aptos/train.csv"
IMG_SIZE = (256, 256)   # try (512,512) if VRAM/CPU allows
N_THRESH = 64           # Betti curve resolution
BALANCE_PER_CLASS = None  # None => balance to max class count; or set an int cap per class (e.g., 800)
AUGMENT = True
RANDOM_SEED = 42

# ------------------ ATTENTION MASK PARAMS ------------------
USE_ATTENTION_MASK = True      # toggle on/off
ATT_METHOD = "entropy_sobel"   # "entropy_sobel" (if scikit-image) or "grad"
ATT_GAMMA = 0.7                # <1.0 softens peaks; >1.0 sharpens
ATT_BIAS = 0.35                # base weight so background isn't zeroed (0..1)
# Effective weight = ATT_BIAS + (1-ATT_BIAS) * (mask ** ATT_GAMMA)
# ----------------------------------------------------------

# Device for CNN feature extraction
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# ------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# ---------- TDA HELPERS ----------
def betti_thresholds_from_diagram(diag, n_thresh=N_THRESH):
    """Adaptive thresholds spanning [min_birth, max_death] of the diagram.
    If diagram empty, default to [0,1].
    """
    if len(diag) == 0:
        return np.linspace(0.0, 1.0, n_thresh)
    births = diag[:, 0]
    deaths = np.where(np.isinf(diag[:, 1]), np.nan, diag[:, 1])
    lo = np.nanmin(births) if np.isfinite(births).any() else 0.0
    hi_candidates = np.concatenate([births[~np.isnan(births)], deaths[~np.isnan(deaths)]])
    hi = np.nanmax(hi_candidates) if hi_candidates.size else lo + 1.0
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_thresh)


def diagram_to_betti_vector(diag, n_thresh=N_THRESH, thresholds=None):
    """Compute Betti curve (count of features alive) over thresholds.
    diag: array of shape (n_pairs, 2) with (birth, death), possibly +/-inf deaths.
    thresholds: optional precomputed thresholds; if None, inferred from diag.
    """
    if len(diag) == 0:
        return np.zeros(n_thresh, dtype=np.float32)
    if thresholds is None:
        thresholds = betti_thresholds_from_diagram(diag, n_thresh)
    betti = np.zeros_like(thresholds, dtype=np.float32)
    # Replace inf death with max threshold to count as alive up to end
    death_vals = np.where(np.isinf(diag[:, 1]), thresholds[-1], diag[:, 1])
    birth_vals = diag[:, 0]
    for b, d in zip(birth_vals, death_vals):
        # alive where t in [b, d)
        alive = (thresholds >= b) & (thresholds < d)
        betti[alive] += 1.0
    return betti


def compute_persistence_diagram_from_array(a2d):
    """Build cubical complex on a2d using sublevel filtration via negated intensities."""
    a2d = np.asarray(a2d, dtype=np.float32)
    top_cells = (-a2d).ravel()
    cc = CubicalComplex(dimensions=a2d.shape, top_dimensional_cells=top_cells)
    cc.persistence()
    pd0 = np.array(cc.persistence_intervals_in_dimension(0))
    pd1 = np.array(cc.persistence_intervals_in_dimension(1))
    return pd0, pd1


def compute_tda_features_from_green(green_clahe_norm, n_thresh=N_THRESH):
    pd0, pd1 = compute_persistence_diagram_from_array(green_clahe_norm)
    # Betti curves in dims 0 and 1
    # Use a shared threshold grid to keep features aligned
    th0 = betti_thresholds_from_diagram(pd0, n_thresh)
    th1 = betti_thresholds_from_diagram(pd1, n_thresh)
    b0 = diagram_to_betti_vector(pd0, n_thresh=n_thresh, thresholds=th0)
    b1 = diagram_to_betti_vector(pd1, n_thresh=n_thresh, thresholds=th1)
    return np.concatenate([b0, b1]).astype(np.float32)


# ---------- PREPROCESS: GREEN + CLAHE + CIRCULAR MASK + Z-SCORE ----------
def preprocess_green_clahe(img_rgb, resize_shape=IMG_SIZE, clip_limit=0.03):
    """Return a z-scored green channel (inside circular mask) after CLAHE.
    Output is float32 2D array.
    """
    img = img_rgb.resize(resize_shape)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    green = arr[..., 1] if arr.ndim == 3 else arr

    if _HAS_SKIMAGE:
        # skimage expects [0,1] float; returns [0,1]
        from skimage import exposure as _exp
        green_eq = _exp.equalize_adapthist(green, clip_limit=clip_limit, nbins=256).astype(np.float32)
    else:
        # Fallback: simple histogram equalization with PIL
        g8 = (np.clip(green, 0, 1) * 255).astype(np.uint8)
        green_eq = np.array(ImageOps.equalize(Image.fromarray(g8)), dtype=np.float32) / 255.0

    h, w = green_eq.shape
    cy, cx = h // 2, w // 2
    r = int(0.9 * min(cy, cx))
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

    vals = green_eq[mask]
    if vals.size == 0:
        mu, sigma = 0.0, 1.0
    else:
        mu, sigma = float(vals.mean()), float(vals.std())
        if sigma < 1e-6:
            sigma = 1.0

    norm = np.zeros_like(green_eq, dtype=np.float32)
    norm[mask] = (green_eq[mask] - mu) / sigma
    return norm


# ---------- ATTENTION MASK (unsupervised) ----------
def compute_attention_mask_from_green_like(green_like: np.ndarray,
                                           method: str = ATT_METHOD) -> np.ndarray:
    """
    Build a 2D attention mask in [0,1] the same size as green_like.
    - If scikit-image available and method='entropy_sobel': combine local entropy + edges
    - Else: use gradient magnitude
    Mask is also restricted to the same circular retina region used in preprocessing.
    """
    g = np.asarray(green_like, dtype=np.float32)
    # normalize to [0,1] robustly
    g_min, g_max = float(np.nanmin(g)), float(np.nanmax(g))
    if not np.isfinite(g_min) or not np.isfinite(g_max) or g_max <= g_min:
        g_norm = np.zeros_like(g, dtype=np.float32)
    else:
        g_norm = (g - g_min) / (g_max - g_min + 1e-8)

    h, w = g_norm.shape
    cy, cx = h // 2, w // 2
    r = int(0.9 * min(cy, cx))
    Y, X = np.ogrid[:h, :w]
    circ = ((X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2).astype(np.float32)

    if _HAS_SKIMAGE and method == "entropy_sobel":
        from skimage.filters import sobel
        from skimage.filters.rank import entropy
        from skimage.morphology import disk

        # entropy wants uint8
        e = entropy((g_norm * 255).astype(np.uint8), disk(7)).astype(np.float32)
        if e.max() > 0:
            e /= e.max()
        s = sobel(g_norm).astype(np.float32)
        if s.max() > 0:
            s /= s.max()
        m = 0.6 * e + 0.4 * s
    else:
        # gradient magnitude fallback
        gy, gx = np.gradient(g_norm)
        m = np.hypot(gx, gy).astype(np.float32)
        if m.max() > 0:
            m /= m.max()

    m *= circ
    # normalize again to [0,1]
    m_min, m_max = float(m.min()), float(m.max())
    if m_max > m_min:
        m = (m - m_min) / (m_max - m_min + 1e-8)
    else:
        m = np.zeros_like(m, dtype=np.float32)
    return m


# ---------- CNN FEATURE EXTRACTOR ----------
def get_cnn_model():
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    return model.to(DEVICE)


_CNN_MODEL = get_cnn_model()
_CNN_TRANSFORM = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_cnn_features(img_rgb: Image.Image):
    x = _CNN_TRANSFORM(img_rgb).unsqueeze(0).to(DEVICE)
    feats = _CNN_MODEL(x)
    return feats.detach().cpu().numpy().flatten().astype(np.float32)


# ---------- AUGMENTATIONS ----------
def apply_aug_pil(img: Image.Image, aug_type: str | None):
    if not aug_type:
        return img
    if aug_type == 'rotation':
        angle = random.uniform(-15, 15)
        return img.rotate(angle, resample=Image.BILINEAR)
    if aug_type == 'flip_horizontal':
        return ImageOps.mirror(img)
    if aug_type == 'flip_vertical':
        return ImageOps.flip(img)
    if aug_type == 'gaussian_noise':
        arr = np.asarray(img).astype(np.float32) / 255.0
        noise = np.random.normal(0.0, 0.02, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if aug_type == 'brightness':
        factor = random.uniform(0.85, 1.15)
        return ImageEnhance.Brightness(img).enhance(factor)
    if aug_type == 'contrast':
        factor = random.uniform(0.85, 1.15)
        return ImageEnhance.Contrast(img).enhance(factor)
    return img


# ---------- DATASET BUILD (BALANCED + AUG) ----------
def resolve_image_path(data_dir, id_code):
    p_png = os.path.join(data_dir, f"{id_code}.png")
    p_jpg = os.path.join(data_dir, f"{id_code}.jpg")
    if os.path.exists(p_png):
        return p_png
    if os.path.exists(p_jpg):
        return p_jpg
    return None


def build_balanced_index(df: pd.DataFrame, data_dir: str, target_per_class: int | None = BALANCE_PER_CLASS):
    by_cls: dict[int, list[tuple[str, str]]] = {}
    classes = sorted(df['diagnosis'].unique())
    for c in classes:
        by_cls[int(c)] = []

    for _, r in df.iterrows():
        p = resolve_image_path(data_dir, r['id_code'])
        if p is not None:
            by_cls[int(r['diagnosis'])].append((r['id_code'], p))

    counts = {c: len(v) for c, v in by_cls.items()}
    max_count = max(counts.values()) if counts else 0
    tgt = target_per_class or max_count

    aug_types = ['rotation', 'flip_horizontal', 'flip_vertical', 'gaussian_noise', 'brightness', 'contrast']
    items = []  # (id_code_like, path, aug_type_or_None, label)

    for c, lst in by_cls.items():
        # Original images
        items.extend([(idc, p, None, c) for (idc, p) in lst])
        need = max(0, tgt - len(lst))
        for i in range(need):
            if not lst:
                continue
            idc, p = random.choice(lst)
            aug = random.choice(aug_types) if AUGMENT else None
            items.append((f"{idc}_aug_{i}", p, aug, c))

    return items, classes


def build_dataset_balanced():
    df = pd.read_csv(CSV_PATH)
    items, classes = build_balanced_index(df, DATA_DIR, target_per_class=BALANCE_PER_CLASS)

    X, y = [], []
    for idc, path, aug_type, label in tqdm(items, total=len(items), desc="Building features"):
        if not os.path.exists(path):
            continue
        img_rgb = Image.open(path).convert("RGB").resize(IMG_SIZE)
        img_rgb = apply_aug_pil(img_rgb, aug_type)

        # Build green (z-scored, circular) for attention + TDA
        green_norm = preprocess_green_clahe(img_rgb, resize_shape=IMG_SIZE, clip_limit=0.03)

        # ----- Attention mask -----
        if USE_ATTENTION_MASK:
            att = compute_attention_mask_from_green_like(green_norm, method=ATT_METHOD)
            att = np.clip(att, 0.0, 1.0) ** ATT_GAMMA
            weight = ATT_BIAS + (1.0 - ATT_BIAS) * att  # [0,1] -> [ATT_BIAS,1]

            # apply to RGB for CNN
            rgb_arr = (np.asarray(img_rgb).astype(np.float32) / 255.0)
            rgb_weighted = np.clip(rgb_arr * weight[..., None], 0.0, 1.0)
            img_rgb_for_cnn = Image.fromarray((rgb_weighted * 255).astype(np.uint8))

            # apply to green for TDA
            green_for_tda = (green_norm * weight).astype(np.float32)
        else:
            img_rgb_for_cnn = img_rgb
            green_for_tda = green_norm

        # CNN (RGB) with attention
        cnn_feat = extract_cnn_features(img_rgb_for_cnn)

        # TDA (Green + attention)
        tda_feat = compute_tda_features_from_green(green_for_tda, n_thresh=N_THRESH)

        feat = np.concatenate([cnn_feat, tda_feat])
        X.append(feat)
        y.append(int(label))

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=int)
    classes = np.array(sorted(set(y)))
    return X, y, classes


# ---------- MAIN ----------
if __name__ == "__main__":
    print("Building CNN+CLAHE(TDA)+Balanced feature dataset with Attention Mask...")
    X, y, classes = build_dataset_balanced()
    print("Feature matrix shape:", X.shape)
    print("Classes:", classes.tolist())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, class_weight='balanced', random_state=RANDOM_SEED)
    pipeline = make_pipeline(StandardScaler(), clf)

    accs, precs, recs, f1s = [], [], [], []
    y_true_all, y_pred_all = [], []
    # For ROC without leakage, store test-set probabilities across folds
    y_score_all = []  # shape accumulates (n_samples, n_classes)

    fold = 1
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        print(f"\nFold {fold} Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        fold += 1

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_score_all.append(y_prob)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_score_all = np.concatenate(y_score_all, axis=0)

    # ---------- SUMMARY ----------
    print("\n--- Cross-validation Summary ---")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1-score: {np.mean(f1s):.4f}")

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true_all, y_pred_all, labels=classes)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix (All folds)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

    # ---------- ACCURACY PLOT ----------
    plt.figure()
    plt.plot(range(1, len(accs)+1), accs, marker='o')
    plt.title("Cross-validation Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- Multi-class ROC (no leakage) ----------
    y_bin = label_binarize(y_true_all, classes=classes)
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score_all[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {cls} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves per Class (CV, held-out)")
    plt.legend()
    plt.tight_layout()
    plt.show()