import os
import random
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
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

# Optional: use skimage for CLAHE
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

# Use 300x300 so EfficientNet-B3 and TDA both see decent resolution
IMG_SIZE = (300, 300)

# Slightly higher Betti curve resolution for richer TDA features
N_THRESH = 128

BALANCE_PER_CLASS = None  # None => balance to max class count; or set an int cap per class (e.g., 800)
AUGMENT = True
RANDOM_SEED = 42

# Test-time augmentation flag for CNN feature extraction
USE_TTA = True

# Device for CNN feature extraction
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
# ------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
#                  TDA HELPERS
# ============================================================
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
    # Use separate threshold grids per dimension (could be unified, but this is fine)
    th0 = betti_thresholds_from_diagram(pd0, n_thresh)
    th1 = betti_thresholds_from_diagram(pd1, n_thresh)
    b0 = diagram_to_betti_vector(pd0, n_thresh=n_thresh, thresholds=th0)
    b1 = diagram_to_betti_vector(pd1, n_thresh=n_thresh, thresholds=th1)
    return np.concatenate([b0, b1]).astype(np.float32)


# ============================================================
#     PREPROCESS: GREEN + CLAHE + CIRCULAR MASK + Z-SCORE
# ============================================================
def preprocess_green_clahe(img_rgb, resize_shape=IMG_SIZE, clip_limit=0.03):
    """Return a z-scored green channel (inside circular mask) after CLAHE.
    Output is float32 2D array.
    """
    img = img_rgb.resize(resize_shape)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    green = arr[..., 1] if arr.ndim == 3 else arr

    if _HAS_SKIMAGE:
        # skimage expects [0,1] float; returns [0,1]
        green_eq = exposure.equalize_adapthist(green, clip_limit=clip_limit, nbins=256).astype(np.float32)
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


# ============================================================
#       DYNAMIC ATTENTION MASK (GREEN-DRIVEN)
# ============================================================
def apply_dynamic_attention(img_rgb: Image.Image,
                            resize_shape=IMG_SIZE,
                            clip_limit=0.03,
                            alpha: float = 0.7):
    """
    Build a dynamic spatial attention map from the green channel after CLAHE+z-score.
    Higher |z-score| => higher attention (more salient vessels/lesions).

    alpha controls how much we mix attention-weighted image vs. original.
    """
    # Use the same preprocessing we already trust for TDA
    green_norm = preprocess_green_clahe(img_rgb, resize_shape=resize_shape,
                                        clip_limit=clip_limit)

    # Attention magnitude from green_norm
    att = np.abs(green_norm)
    att = att - att.min()
    if att.max() > 0:
        att = att / att.max()

    # Smooth attention
    att_img = Image.fromarray((att * 255).astype(np.uint8))
    att_img = att_img.filter(ImageFilter.GaussianBlur(radius=3))
    att = np.asarray(att_img).astype(np.float32) / 255.0

    # Broadcast to 3 channels
    att3 = np.stack([att, att, att], axis=-1)

    arr = np.asarray(img_rgb.resize(resize_shape)).astype(np.float32) / 255.0

    # Apply attention with a residual connection
    arr_att = arr * att3
    arr_combined = alpha * arr_att + (1.0 - alpha) * arr
    arr_combined = np.clip(arr_combined, 0.0, 1.0)

    return Image.fromarray((arr_combined * 255).astype(np.uint8))


# ============================================================
#              CNN FEATURE EXTRACTOR (EffNet-B3)
# ============================================================
def get_cnn_model_and_transform():
    """
    Use EfficientNet-B3 as a frozen feature extractor.
    We remove the classifier and use the penultimate feature vector.
    """
    try:
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
        transform = weights.transforms()  # Resize(300), CenterCrop, ToTensor, Normalize
    except Exception:
        # Fallback if weights enum not available
        model = models.efficientnet_b3(pretrained=True)
        transform = T.Compose([
            T.Resize(300),
            T.CenterCrop(300),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    # Remove classifier: EfficientNet-B3 has classifier = [Dropout, Linear]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Identity()

    model.eval()
    model.to(DEVICE)
    return model, transform


_CNN_MODEL, _CNN_TRANSFORM = get_cnn_model_and_transform()


@torch.no_grad()
def extract_cnn_features(img_rgb: Image.Image,
                         use_tta: bool = True):
    """
    Extract EfficientNet-B3 features with:
      - dynamic attention mask applied on the image
      - optional test-time augmentation (TTA) and feature averaging
    """
    base_img = img_rgb

    # Build TTA variants at the image level
    imgs = [base_img]
    if USE_TTA and use_tta:
        imgs.append(ImageOps.mirror(base_img))  # horizontal flip
        imgs.append(base_img.rotate(10, resample=Image.BILINEAR))
        imgs.append(base_img.rotate(-10, resample=Image.BILINEAR))

    feat_list = []
    for im in imgs:
        # Apply dynamic attention before CNN transform
        im_att = apply_dynamic_attention(im, resize_shape=IMG_SIZE, clip_limit=0.03)
        x = _CNN_TRANSFORM(im_att).unsqueeze(0).to(DEVICE)
        feats = _CNN_MODEL(x)
        feat_list.append(feats.detach().cpu().numpy().reshape(1, -1))

    feats_mean = np.mean(np.vstack(feat_list), axis=0)
    return feats_mean.astype(np.float32).flatten()


# ============================================================
#                     AUGMENTATIONS
# ============================================================
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
    if aug_type == 'color':
        factor = random.uniform(0.9, 1.1)
        return ImageEnhance.Color(img).enhance(factor)
    return img


# ============================================================
#        DATASET BUILD (BALANCED + AUGMENTED)
# ============================================================
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

    aug_types = ['rotation', 'flip_horizontal', 'flip_vertical',
                 'gaussian_noise', 'brightness', 'contrast', 'color']
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

        # CNN (RGB with dynamic attention + TTA)
        cnn_feat = extract_cnn_features(img_rgb)

        # TDA (Green + CLAHE + mask + z-score)
        green_norm = preprocess_green_clahe(img_rgb, resize_shape=IMG_SIZE, clip_limit=0.03)
        tda_feat = compute_tda_features_from_green(green_norm, n_thresh=N_THRESH)

        feat = np.concatenate([cnn_feat, tda_feat])
        X.append(feat)
        y.append(int(label))

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=int)
    classes = np.array(sorted(set(y)))
    return X, y, classes


# ============================================================
#                         MAIN
# ============================================================
if __name__ == "__main__":
    print("Building EfficientNetB3+Attention+CLAHE(TDA)+Balanced feature dataset...")
    X, y, classes = build_dataset_balanced()
    print("Feature matrix shape:", X.shape)
    print("Classes:", classes.tolist())

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    clf = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_SEED
    )
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
    print(f"Precision (macro): {np.mean(precs):.4f}")
    print(f"Recall (macro): {np.mean(recs):.4f}")
    print(f"F1-score (macro): {np.mean(f1s):.4f}")

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