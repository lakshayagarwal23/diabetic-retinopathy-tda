import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from gudhi import CubicalComplex
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# ------------------ PARAMETERS ------------------
DATA_DIR = "/Users/satyaswaroopnune/Desktop/aptos/train_images/train_images"
CSV_PATH = "/Users/satyaswaroopnune/Desktop/aptos/train.csv"
IMG_SIZE = (128, 128)   # smaller size for faster computation
CHANNELS = ["gray","R","G","B"]
N_THRESH = 100           # number of thresholds for filtration
# ------------------------------------------------

def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img)
    return arr

def channel_image(img_arr, channel):
    if channel == "gray":
        return img_arr.mean(axis=2).astype(np.float32)
    elif channel in ("R","G","B"):
        idx = {"R":0,"G":1,"B":2}[channel]
        return img_arr[:,:,idx].astype(np.float32)
    else:
        raise ValueError("Unknown channel")

def compute_persistence_diagram(channel_img):
    """
    Computes persistent homology for a 2D image using cubical complex.
    Returns persistence intervals for H0 (components) and H1 (holes)
    """
    # Normalize image to [0,1]
    norm_img = channel_img / 255.0
    cc = CubicalComplex(top_dimensional_cells=norm_img)
    cc.persistence()
    pd0 = np.array(cc.persistence_intervals_in_dimension(0))
    pd1 = np.array(cc.persistence_intervals_in_dimension(1))
    return pd0, pd1

def diagram_to_betti_vector(diag, n_thresh=N_THRESH):
    """
    Converts a persistence diagram to a Betti vector of length n_thresh.
    Betti vector: number of "alive" features at each threshold
    """
    if len(diag) == 0:
        return np.zeros(n_thresh)
    # Thresholds evenly spaced in [0,1]
    thresholds = np.linspace(0,1,n_thresh)
    betti = np.zeros(n_thresh)
    for birth, death in diag:
        # death = np.inf → consider as 1 (max threshold)
        death = 1.0 if np.isinf(death) else death
        # mark Betti number active for thresholds in [birth, death)
        betti += (thresholds >= birth) & (thresholds < death)
    return betti

def compute_feature_vector(img_arr, channels=CHANNELS, n_thresh=N_THRESH):
    features = []
    for ch in channels:
        ch_img = channel_image(img_arr, ch)
        pd0, pd1 = compute_persistence_diagram(ch_img)
        b0 = diagram_to_betti_vector(pd0, n_thresh)
        b1 = diagram_to_betti_vector(pd1, n_thresh)
        features.append(b0)
        features.append(b1)
    return np.concatenate(features)

def build_dataset(csv_path=CSV_PATH, data_dir=DATA_DIR, sample_limit=None):
    df = pd.read_csv(csv_path)
    ids = df['id_code'].values
    labels = df['diagnosis'].values
    if sample_limit is not None:
        ids = ids[:sample_limit]
        labels = labels[:sample_limit]

    X, y = [], []
    for id_code, label in tqdm(zip(ids, labels), total=len(ids)):
        img_path = os.path.join(data_dir, f"{id_code}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(data_dir, f"{id_code}.jpg")
            if not os.path.exists(img_path):
                continue
        img_arr = load_image(img_path)
        feat = compute_feature_vector(img_arr)
        X.append(feat)
        y.append(int(label))
    return np.stack(X), np.array(y)

# ------------------- MAIN ----------------------
if __name__ == "__main__":
    print("Building dataset features...")
    X, y = build_dataset(sample_limit=300)  # adjust sample_limit for testing
    print("Feature matrix shape:", X.shape)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = make_pipeline(StandardScaler(), clf)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("Running cross-validation...")
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print("Mean accuracy: {:.4f}, Std: {:.4f}".format(scores.mean(), scores.std()))