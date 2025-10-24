import os
import numpy as np
import pandas as pd
from PIL import Image
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
import torchvision.transforms as transforms

# ------------------ PARAMETERS ------------------
DATA_DIR = "/Users/satyaswaroopnune/Downloads/aptos/train_images/train_images"
CSV_PATH = "/Users/satyaswaroopnune/Downloads/aptos/train.csv"
IMG_SIZE = (128, 128)
N_THRESH = 50
SAMPLE_LIMIT = 1000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# ------------------------------------------------

# ---------- TDA FUNCTIONS ----------
def compute_persistence_diagram(channel_img):
    norm_img = channel_img / 255.0
    cc = CubicalComplex(top_dimensional_cells=norm_img)
    cc.persistence()
    pd0 = np.array(cc.persistence_intervals_in_dimension(0))
    pd1 = np.array(cc.persistence_intervals_in_dimension(1))
    return pd0, pd1

def diagram_to_betti_vector(diag, n_thresh=N_THRESH):
    if len(diag) == 0:
        return np.zeros(n_thresh)
    thresholds = np.linspace(0, 1, n_thresh)
    betti = np.zeros(n_thresh)
    for birth, death in diag:
        death = 1.0 if np.isinf(death) else death
        betti += (thresholds >= birth) & (thresholds < death)
    return betti

def compute_tda_features(img):
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    pd0, pd1 = compute_persistence_diagram(img_gray)
    b0 = diagram_to_betti_vector(pd0)
    b1 = diagram_to_betti_vector(pd1)
    return np.concatenate([b0, b1])

# ---------- CNN FEATURE EXTRACTOR ----------
def get_cnn_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.eval()
    return model.to(DEVICE)

cnn_model = get_cnn_model()

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_cnn_features(img):
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    feats = cnn_model(img_tensor)
    return feats.cpu().numpy().flatten()

# ---------- DATASET BUILD ----------
def build_dataset():
    df = pd.read_csv(CSV_PATH)
    ids = df['id_code'].values[:SAMPLE_LIMIT]
    labels = df['diagnosis'].values[:SAMPLE_LIMIT]

    X, y = [], []
    for id_code, label in tqdm(zip(ids, labels), total=len(ids)):
        path = os.path.join(DATA_DIR, f"{id_code}.png")
        if not os.path.exists(path):
            path = path.replace(".png", ".jpg")
            if not os.path.exists(path):
                continue

        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        cnn_feat = extract_cnn_features(img)
        tda_feat = compute_tda_features(img)
        feat = np.concatenate([cnn_feat, tda_feat])
        X.append(feat)
        y.append(int(label))
    return np.stack(X), np.array(y)

# ---------- MAIN ----------
if __name__ == "__main__":
    print("Building CNN+TDA feature dataset...")
    X, y = build_dataset()
    print("Feature matrix shape:", X.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    pipeline = make_pipeline(StandardScaler(), clf)

    accs, precs, recs, f1s = [], [], [], []
    fold = 1
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, average='macro', zero_division=0))
        recs.append(recall_score(y_test, y_pred, average='macro', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average='macro', zero_division=0))

        print(f"\nFold {fold} Report:")
        print(classification_report(y_test, y_pred))
        fold += 1

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # ---------- SUMMARY ----------
    print("\n--- Cross-validation Summary ---")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1-score: {np.mean(f1s):.4f}")

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # ---------- ACCURACY PLOT ----------
    plt.figure()
    plt.plot(range(1, 6), accs, marker='o')
    plt.title("Cross-validation Accuracy per Fold")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.show()

    # Multi class ROC Curve
    y_bin = label_binarize(y_true_all, classes=np.unique(y))
    y_prob = pipeline.predict_proba(X)

    plt.figure(figsize=(8,6))
    for i in range(y_bin.shape[1]):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves per Class")
    plt.legend()
    plt.show()