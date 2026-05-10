# ═══════════════════════════════════════════════════════════════════════════
#  WESAD THREE-CLASS MOD — Low (Baseline) vs Moderate (Meditation) vs High (Stress)
#
#  LABEL MAP:
#    WESAD 1 (Baseline)    → 0  Low
#    WESAD 4 (Meditation)  → 1  Moderate
#    WESAD 2 (Stress/TSST) → 2  High
#    WESAD 3 (Amusement)   → DROPPED
#
#  METHODOLOGY (inherits all 5 fixes from binary fixed pipeline):
#    FIX 1 — No SMOTE; noise augmentation only
#    FIX 2 — Leakage-free normalisation (train stats → test per fold)
#    FIX 3 — 15s stride for more test windows per subject
#    FIX 4 — CNN and RF trained on same real-data distribution
#    FIX 5 — Per-class explicit weight boost for High (Stress) class
#
#  OUTPUTS → C:\Users\sharv\Downloads\UMT\.umtbatch\Scripts\model_outputs_binary\new_mod\
# ═══════════════════════════════════════════════════════════════════════════

import os
os.environ['TF_NUM_INTRAOP_THREADS'] = '12'
os.environ['TF_NUM_INTEROP_THREADS'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'
os.environ['OMP_NUM_THREADS']        = '12'
os.environ['MKL_NUM_THREADS']        = '12'

import pickle, warnings, json, gc, time
import numpy as np
import psutil
from collections import Counter
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.stats import skew

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    precision_recall_fscore_support,
    balanced_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')
tf.random.set_seed(42)
np.random.seed(42)

print("╔══════════════════════════════════════════════════════════════╗")
print("║  WESAD THREE-CLASS MOD — Low / Moderate / High Stress       ║")
print("╠══════════════════════════════════════════════════════════════╣")
print(f"║  TensorFlow  : {tf.__version__:<45}║")
print(f"║  GPU         : {str(len(tf.config.list_physical_devices('GPU'))>0):<45}║")
ram = psutil.virtual_memory()
print(f"║  RAM Total   : {f'{ram.total/1e9:.1f} GB':<45}║")
print(f"║  RAM Free    : {f'{ram.available/1e9:.1f} GB':<45}║")
print("╠══════════════════════════════════════════════════════════════╣")
print("║  LABEL MAP:                                                  ║")
print("║    Baseline(1) → 0 Low  |  Meditation(4) → 1 Moderate       ║")
print("║    Stress(2)   → 2 High |  Amusement(3) DROPPED             ║")
print("╚══════════════════════════════════════════════════════════════╝")

# ── CONFIG ─────────────────────────────────────────────────────────────────
DATA_PATH   = 'C:\\Users\\sharv\\Downloads\\WESAD'
SUBJECTS    = ['S2','S3','S4','S5','S6','S7','S8','S9',
               'S10','S11','S13','S14','S15','S16','S17']

LABEL_MAP   = {1: 0, 4: 1, 2: 2}    # Baseline=Low, Meditation=Moderate, Stress=High
SHORT_NAMES = ['Low', 'Moderate', 'High']
N_CLASSES   = 3

BVP_FS    = 64
EDA_FS    = 4
TEMP_FS   = 4
LABEL_FS  = 700
TARGET_FS = 64

WINDOW_SEC  = 60
SHIFT_SEC   = 15
WINDOW_SIZE = int(TARGET_FS * WINDOW_SEC)
STRIDE      = int(TARGET_FS * SHIFT_SEC)
N_CHANNELS  = 3

EPOCHS        = 200
BATCH_SIZE    = 32
RANDOM_STATE  = 42
RF_ESTIMATORS = 300

OUTPUT_DIR = r'C:\Users\sharv\Downloads\UMT\.umtbatch\Scripts\model_outputs_binary\new_mod'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'fold_checkpoints'), exist_ok=True)

print(f"\n  Window   : {WINDOW_SIZE} samples = {WINDOW_SEC}s @ {TARGET_FS}Hz")
print(f"  Stride   : {STRIDE} samples = {SHIFT_SEC}s (75% overlap)")
print(f"  Labels   : Low(Baseline) | Moderate(Meditation) | High(Stress)")
print(f"  Output   : {OUTPUT_DIR}")

# ── PREPROCESSING ───────────────────────────────────────────────────────────

def bandpass(sig, lo=0.5, hi=5.0, fs=64, order=4):
    try:
        nyq  = fs * 0.5
        b, a = butter(order, [lo/nyq, min(hi/nyq, 0.99)], btype='band')
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def lowpass(sig, cut=1.0, fs=64, order=4):
    try:
        nyq  = fs * 0.5
        b, a = butter(order, min(cut/nyq, 0.99), btype='low')
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def upsample(sig, src, tgt):
    try:
        if src == tgt: return sig
        xo = np.linspace(0, 1, len(sig))
        xn = np.linspace(0, 1, int(len(sig) * tgt / src))
        return interp1d(xo, sig, kind='linear')(xn)
    except Exception:
        return sig

def downsample_labels(raw, src, tgt):
    ratio = src / tgt
    n     = int(len(raw) / ratio)
    out   = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s = int(i * ratio)
        e = min(int((i+1)*ratio), len(raw))
        if s >= len(raw): break
        out[i] = int(Counter(raw[s:e]).most_common(1)[0][0])
    return out

def augment_window(window, noise_std=0.015, rng=None):
    if rng is None: rng = np.random.default_rng()
    return window + rng.normal(0, noise_std, window.shape).astype(np.float32)

# ── FEATURE EXTRACTION ──────────────────────────────────────────────────────

def bvp_features(bvp_win, fs=64):
    try:
        mean = float(np.mean(bvp_win)); std = float(np.std(bvp_win))
        mn   = float(np.min(bvp_win));  mx  = float(np.max(bvp_win))
        freqs     = np.fft.rfftfreq(len(bvp_win), 1.0/fs)
        power     = np.abs(np.fft.rfft(bvp_win))**2
        peak_freq = float(freqs[np.argmax(power)])
        thr  = np.mean(bvp_win) + 0.3*np.std(bvp_win)
        peaks, _ = find_peaks(bvp_win, height=thr, distance=int(fs*0.4))
        if len(peaks) < 4:
            hrv = np.zeros(10, dtype=np.float32)
        else:
            rr      = np.diff(peaks)/fs*1000.0
            mean_rr = float(np.mean(rr))
            sdnn    = float(np.std(rr))
            rmssd   = float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else 0.0
            pnn50   = float(np.sum(np.abs(np.diff(rr))>50)/len(rr)*100) if len(rr)>1 else 0.0
            cv      = float(sdnn/mean_rr) if mean_rr>0 else 0.0
            lf=hf=lfhf=sd1=sd2 = 0.0,0.0,0.0,0.0,0.0
            if len(rr) >= 4:
                t  = np.cumsum(rr)/1000.0
                ti = np.arange(float(t[0]), float(t[-1]), 0.25)
                if len(ti) >= 8:
                    try:
                        ri   = interp1d(t, rr, kind='linear')(ti)
                        f, p = welch(ri, fs=4.0, nperseg=min(len(ri), 64))
                        lf_m = (f>=0.04)&(f<0.15); hf_m = (f>=0.15)&(f<0.40)
                        lf   = float(trapezoid(p[lf_m],f[lf_m])) if lf_m.any() else 0.0
                        hf   = float(trapezoid(p[hf_m],f[hf_m])) if hf_m.any() else 0.0
                        lfhf = float(lf/hf) if hf>1e-10 else 0.0
                    except Exception: pass
            if len(rr) >= 3:
                try:
                    d=np.diff(rr); sd1=float(np.std(d)/np.sqrt(2))
                    sd2=float(np.sqrt(max(2.0*float(np.std(rr))**2-sd1**2, 0.0)))
                except Exception: pass
            hrv = np.array([mean_rr,sdnn,rmssd,pnn50,cv,lf,hf,lfhf,sd1,sd2], dtype=np.float32)
        return np.concatenate([np.array([mean,std,mn,mx,peak_freq], dtype=np.float32), hrv])
    except Exception:
        return np.zeros(15, dtype=np.float32)

def eda_features(eda_win, fs=4):
    try:
        nyq  = fs * 0.5
        b, a = butter(4, min(1.0/nyq, 0.99), btype='low')
        scl  = filtfilt(b, a, eda_win) if len(eda_win)>20 else eda_win.copy()
        scr  = eda_win - scl if len(eda_win)>20 else np.zeros_like(eda_win)
        peaks, props = find_peaks(scr, height=max(0.01*np.std(scr),1e-10), distance=int(fs*1.0))
        n_scr    = int(len(peaks))
        amp_mean = float(np.mean(props['peak_heights'])) if n_scr>0 else 0.0
        amp_std  = float(np.std(props['peak_heights']))  if n_scr>1 else 0.0
        rise_time = 0.0
        if n_scr > 0:
            onsets = []
            for pk in peaks:
                onset = pk
                while onset > 0 and scr[onset] > scr[onset-1]: onset -= 1
                onsets.append(pk - onset)
            rise_time = float(np.mean(onsets)) / fs
        return np.array([float(np.mean(scl)), float(np.std(scl)), float(n_scr),
                         amp_mean, amp_std, rise_time,
                         float(np.mean(eda_win)), float(np.std(eda_win)),
                         float(np.min(eda_win)), float(np.max(eda_win)),
                         float(skew(eda_win))], dtype=np.float32)
    except Exception:
        return np.zeros(11, dtype=np.float32)

def temp_features(temp_win):
    try:
        t     = np.arange(len(temp_win), dtype=np.float32)
        slope = float(np.polyfit(t, temp_win, 1)[0])
        return np.array([float(np.mean(temp_win)), float(np.std(temp_win)),
                         float(np.min(temp_win)),  float(np.max(temp_win)),
                         slope, float(np.max(temp_win)-np.min(temp_win))], dtype=np.float32)
    except Exception:
        return np.zeros(6, dtype=np.float32)

def extract_features(X, fs=64):
    out = []
    for w in X:
        try:
            step = max(1, int(fs/EDA_FS))
            full = np.concatenate([bvp_features(w[:,0], fs),
                                   eda_features(w[::step,1], fs=EDA_FS),
                                   temp_features(w[::step,2])])
            if np.isnan(full).any() or np.isinf(full).any():
                full = np.zeros(32, dtype=np.float32)
        except Exception:
            full = np.zeros(32, dtype=np.float32)
        out.append(full)
    return np.array(out, dtype=np.float32)

# ── SUBJECT LOADER ──────────────────────────────────────────────────────────

def load_subject_raw(subj):
    path = os.path.join(DATA_PATH, subj, f'{subj}.pkl')
    with open(path, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    wrist   = d['signal']['wrist']
    bvp     = np.array(wrist['BVP']).flatten()
    eda     = np.array(wrist['EDA']).flatten()
    temp    = np.array(wrist['TEMP']).flatten()
    lbl_raw = np.array(d['label']).flatten().astype(int)

    eda  = upsample(eda,  EDA_FS,  TARGET_FS)
    temp = upsample(temp, TEMP_FS, TARGET_FS)
    lbl  = downsample_labels(lbl_raw, LABEL_FS, TARGET_FS)

    n = min(len(bvp), len(eda), len(temp), len(lbl))
    bvp=bvp[:n]; eda=eda[:n]; temp=temp[:n]; lbl=lbl[:n]

    mask = np.array([l in LABEL_MAP for l in lbl])
    if mask.sum() == 0: raise ValueError(f"{subj}: no valid labels")
    bvp=bvp[mask]; eda=eda[mask]; temp=temp[mask]; lbl=lbl[mask]

    bvp  = bandpass(bvp, lo=0.5, hi=5.0, fs=BVP_FS)
    eda  = lowpass(eda,  cut=1.0, fs=TARGET_FS)
    temp = lowpass(temp, cut=0.5, fs=TARGET_FS)

    def rscale(x): return RobustScaler().fit_transform(x.reshape(-1,1)).flatten()
    bvp, eda, temp = rscale(bvp), rscale(eda), rscale(temp)

    lbl3 = np.array([LABEL_MAP[l] for l in lbl], dtype=np.int32)
    return bvp, eda, temp, lbl3

def make_windows(bvp, eda, temp, lbl, win=WINDOW_SIZE, stride=STRIDE):
    Xw, yw = [], []
    for s in range(0, len(bvp)-win+1, stride):
        e   = s + win
        top = Counter(lbl[s:e].tolist()).most_common(1)[0][0]
        Xw.append(np.stack([bvp[s:e], eda[s:e], temp[s:e]], axis=-1))
        yw.append(top)
    if not Xw:
        return np.empty((0,win,N_CHANNELS),dtype=np.float32), np.empty(0,dtype=np.int32)
    return np.array(Xw,dtype=np.float32), np.array(yw,dtype=np.int32)

def normalise_with_train_stats(X_train, X_test):
    X_tr = X_train.copy(); X_te = X_test.copy()
    for ch in range(N_CHANNELS):
        mu  = X_tr[:,:,ch].mean()
        sig = X_tr[:,:,ch].std() + 1e-8
        X_tr[:,:,ch] = (X_tr[:,:,ch] - mu) / sig
        X_te[:,:,ch] = (X_te[:,:,ch] - mu) / sig
    return X_tr.astype(np.float32), X_te.astype(np.float32)

def balance_with_augmentation(X, y, target_ratio=1.0, noise_std=0.015, seed=42):
    rng   = np.random.default_rng(seed)
    c     = Counter(y.tolist())
    n_max = max(c.values())
    X_out = [X]; y_out = [y]
    for cls, cnt in c.items():
        n_needed = int(n_max * target_ratio) - cnt
        if n_needed <= 0: continue
        idx   = np.where(y == cls)[0]
        aug_i = rng.choice(idx, size=n_needed, replace=True)
        X_aug = np.array([augment_window(X[i], noise_std, rng) for i in aug_i])
        X_out.append(X_aug); y_out.append(np.full(n_needed, cls, dtype=np.int32))
    X_bal = np.concatenate(X_out); y_bal = np.concatenate(y_out)
    idx   = rng.permutation(len(y_bal))
    return X_bal[idx], y_bal[idx]

# ── SE BLOCK + CNN ───────────────────────────────────────────────────────────

def se_block(x, ratio=4):
    f  = x.shape[-1]
    se = layers.GlobalAveragePooling1D()(x)
    se = layers.Dense(max(1,f//ratio), activation='relu')(se)
    se = layers.Dense(f, activation='sigmoid')(se)
    se = layers.Reshape((1,f))(se)
    return layers.Multiply()([x, se])

def build_cnn(win=WINDOW_SIZE, n_classes=N_CLASSES):
    inp    = keras.Input(shape=(win, N_CHANNELS), name='input')
    bvp_in = layers.Lambda(lambda x: x[:,:,0:1], name='bvp_in')(inp)
    eda_in = layers.Lambda(lambda x: x[:,:,1:2], name='eda_in')(inp)
    tmp_in = layers.Lambda(lambda x: x[:,:,2:3], name='tmp_in')(inp)

    # BVP branch
    bx = layers.Conv1D(16, 7, padding='same', use_bias=False, name='bvp_c1')(bvp_in)
    bx = layers.BatchNormalization()(bx); bx = layers.Activation('relu')(bx)
    bx = se_block(bx); bx = layers.MaxPooling1D(4)(bx); bx = layers.Dropout(0.25)(bx)
    bx = layers.Conv1D(32, 5, padding='same', use_bias=False, name='bvp_c2')(bx)
    bx = layers.BatchNormalization()(bx); bx = layers.Activation('relu')(bx)
    bx = se_block(bx); bx = layers.MaxPooling1D(4)(bx)
    bvp_feat = layers.GlobalAveragePooling1D(name='bvp_gap')(bx)

    # EDA branch
    ex = layers.Conv1D(16, 11, padding='same', use_bias=False, name='eda_c1')(eda_in)
    ex = layers.BatchNormalization()(ex); ex = layers.Activation('relu')(ex)
    ex = se_block(ex); ex = layers.MaxPooling1D(4)(ex); ex = layers.Dropout(0.25)(ex)
    ex = layers.Conv1D(32, 7, padding='same', use_bias=False, name='eda_c2')(ex)
    ex = layers.BatchNormalization()(ex); ex = layers.Activation('relu')(ex)
    ex = se_block(ex); ex = layers.MaxPooling1D(4)(ex)
    eda_feat = layers.GlobalAveragePooling1D(name='eda_gap')(ex)

    # TEMP branch
    tx = layers.Conv1D(8, 15, padding='same', use_bias=False, name='tmp_c1')(tmp_in)
    tx = layers.BatchNormalization()(tx); tx = layers.Activation('relu')(tx)
    tx = se_block(tx); tx = layers.MaxPooling1D(4)(tx)
    tmp_feat = layers.GlobalAveragePooling1D(name='tmp_gap')(tx)

    # Fusion: 32+32+8 = 72-D
    merged = layers.Concatenate(name='merged')([bvp_feat, eda_feat, tmp_feat])
    x  = layers.Dense(64, activation='relu', name='dense_64')(merged)
    x  = layers.Dropout(0.4, name='drop_1')(x)
    x  = layers.Dense(32, activation='relu', name='dense_32')(x)
    x  = layers.Dropout(0.3, name='drop_2')(x)
    out= layers.Dense(n_classes, activation='softmax', name='output')(x)

    full  = Model(inp, out,    name='CNN_ModThree')
    feats = Model(inp, merged, name='Feat_ModThree')
    return full, feats

def majority_vote(preds, n=3):
    if len(preds) < n: return preds
    out = preds.copy()
    for i in range(n-1, len(preds)):
        out[i] = Counter(preds[i-n+1:i+1].tolist()).most_common(1)[0][0]
    return out

# Print model info
_f, _e = build_cnn(); _p = _f.count_params()
print(f"\n  CNN params     : {_p:,}")
print(f"  Float32 size   : ~{_p*4/1024:.0f} KB")
print(f"  INT8 size      : ~{_p/1024:.0f} KB")
print(f"  Feature vector : 72-D CNN + 32-D HC = 104-D → RF ({N_CLASSES}-class)")
del _f, _e; keras.backend.clear_session(); gc.collect()

# ── LOAD DATA ───────────────────────────────────────────────────────────────

print(f"\n{'═'*65}")
print(f"  LOADING DATA  (Low=Baseline, Moderate=Meditation, High=Stress, Amusement DROPPED)")
print(f"{'═'*65}\n")

subject_data = {}; available = []; t_load = time.time()
for subj in SUBJECTS:
    if not os.path.exists(os.path.join(DATA_PATH, subj, f'{subj}.pkl')):
        print(f"  ⚠  {subj} — not found"); continue
    try:
        bvp, eda, temp, lbl = load_subject_raw(subj)
        Xs, ys = make_windows(bvp, eda, temp, lbl)
        if len(Xs) == 0: continue
        subject_data[subj] = (Xs, ys); available.append(subj)
        c = Counter(ys.tolist())
        print(f"  ✅ {subj}: {len(Xs):3d} windows  "
              f"Low:{c.get(0,0):3d}  Moderate:{c.get(1,0):3d}  High:{c.get(2,0):3d}")
    except Exception as e:
        print(f"  ⚠  {subj} — {e}")

total_w = sum(len(v[1]) for v in subject_data.values())
print(f"\n  {len(available)} subjects | {total_w} total windows | {time.time()-t_load:.1f}s")
print(f"  RAM: {psutil.virtual_memory().percent:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
#  LOSO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*65}")
print(f"  LOSO — THREE-CLASS MOD  (Low / Moderate / High)")
print(f"{'═'*65}")

results_raw=[];results_voted=[];results_f1=[];results_bal=[]
all_true=[];all_pred=[];fold_log=[]
total_start = time.time()

for fold_idx, test_subj in enumerate(available):
    fold_start  = time.time()
    train_subjs = [s for s in available if s != test_subj]
    fold_seed   = RANDOM_STATE + fold_idx
    np.random.seed(fold_seed); tf.random.set_seed(fold_seed)

    print(f"\n{'─'*65}")
    print(f"  FOLD {fold_idx+1:2d}/{len(available)}  |  Test:{test_subj}  |  Train:{len(train_subjs)} subjects")
    print(f"{'─'*65}")

    X_tr_raw = np.concatenate([subject_data[s][0] for s in train_subjs])
    y_tr_all = np.concatenate([subject_data[s][1] for s in train_subjs])
    X_te_raw = subject_data[test_subj][0].copy()
    y_te     = subject_data[test_subj][1].copy()

    X_tr_norm, X_te_norm = normalise_with_train_stats(X_tr_raw, X_te_raw)

    c_tr = Counter(y_tr_all.tolist()); c_te = Counter(y_te.tolist())
    print(f"  Train: {len(y_tr_all)} windows  {dict(c_tr)}")
    print(f"  Test : {len(y_te)} windows  {dict(c_te)}")

    if len(np.unique(y_tr_all)) < N_CLASSES or len(y_te) == 0:
        print("  ⚠  Skipping — missing class"); continue

    X_bal, y_bal = balance_with_augmentation(
        X_tr_norm, y_tr_all, target_ratio=1.0, noise_std=0.015, seed=fold_seed)
    c_bal = Counter(y_bal.tolist())
    print(f"  After augmentation: {len(y_bal)} windows  {dict(c_bal)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.12, random_state=fold_seed)
    tr_i, vl_i = next(sss.split(X_bal, y_bal))
    X_train, y_train = X_bal[tr_i], y_bal[tr_i]
    X_val,   y_val   = X_bal[vl_i], y_bal[vl_i]

    # Class weights — boost High (Stress) class
    keras.backend.clear_session(); gc.collect()
    present = np.unique(y_train)
    cw_arr  = compute_class_weight('balanced', classes=present, y=y_train)
    cw      = {int(c):float(w) for c,w in zip(present, cw_arr)}
    cw[2]   = cw.get(2, 1.0) * 1.5    # boost High/Stress class
    for c in range(N_CLASSES): cw.setdefault(c, 1.0)
    print(f"  Class weights: Low={cw[0]:.3f}  Moderate={cw[1]:.3f}  High={cw[2]:.3f}")

    cnn_full, cnn_feat = build_cnn()
    cnn_full.compile(optimizer=keras.optimizers.Adam(2e-4, clipnorm=1.0),
                     loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"  Training CNN ({len(y_train)} train, {len(y_val)} val)...")
    hist = cnn_full.fit(
        X_train, to_categorical(y_train, N_CLASSES),
        validation_data=(X_val, to_categorical(y_val, N_CLASSES)),
        epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=cw,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=10, min_lr=1e-7, verbose=0)
        ], verbose=0)
    fold_val_acc = max(hist.history.get('val_accuracy', [0]))
    stopped_ep   = len(hist.history['val_loss'])
    print(f"  CNN val acc : {fold_val_acc*100:.2f}%  (stopped ep {stopped_ep})")

    print("  Extracting features from real windows...")
    Fc_tr = cnn_feat.predict(X_tr_norm, batch_size=64, verbose=0)
    Fc_te = cnn_feat.predict(X_te_norm, batch_size=64, verbose=0)
    Fs_tr = extract_features(X_tr_norm, fs=TARGET_FS)
    Fs_te = extract_features(X_te_norm, fs=TARGET_FS)
    F_tr  = np.concatenate([Fc_tr, Fs_tr], axis=1)
    F_te  = np.concatenate([Fc_te, Fs_te], axis=1)

    feat_scaler = StandardScaler()
    F_tr_sc     = feat_scaler.fit_transform(F_tr)
    F_te_sc     = feat_scaler.transform(F_te)
    print(f"  Fused: {F_tr.shape[1]}-D (72 CNN + 32 HC) — leakage-free")

    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS, max_features='sqrt',
        class_weight={0:1.0, 1:1.0, 2:2.0},
        min_samples_leaf=2, n_jobs=-1, random_state=fold_seed)
    rf.fit(F_tr_sc, y_tr_all)

    y_pred_raw   = rf.predict(F_te_sc)
    y_pred_voted = majority_vote(y_pred_raw, n=3)

    acc_raw   = accuracy_score(y_te, y_pred_raw)
    acc_voted = accuracy_score(y_te, y_pred_voted)
    f1_macro  = f1_score(y_te, y_pred_voted, average='macro',    zero_division=0)
    bal_acc   = balanced_accuracy_score(y_te, y_pred_voted)

    # Per-class F1
    f1_per = f1_score(y_te, y_pred_voted, average=None, zero_division=0)
    f1_low = float(f1_per[0]) if len(f1_per)>0 else 0.0
    f1_mod = float(f1_per[1]) if len(f1_per)>1 else 0.0
    f1_hi  = float(f1_per[2]) if len(f1_per)>2 else 0.0

    results_raw.append(acc_raw); results_voted.append(acc_voted)
    results_f1.append(f1_macro); results_bal.append(bal_acc)
    all_true.extend(y_te.tolist()); all_pred.extend(y_pred_voted.tolist())

    fold_time = (time.time()-fold_start)/60
    elapsed   = (time.time()-total_start)/60
    remaining = fold_time*(len(available)-fold_idx-1)

    fold_result = {
        'subject': test_subj, 'acc_raw': float(acc_raw),
        'acc_voted': float(acc_voted), 'f1_macro': float(f1_macro),
        'f1_low': f1_low, 'f1_moderate': f1_mod, 'f1_high': f1_hi,
        'bal_acc': float(bal_acc), 'cnn_val': float(fold_val_acc),
        'time_min': float(fold_time), 'n_test': int(len(y_te))
    }
    fold_log.append(fold_result)
    with open(os.path.join(OUTPUT_DIR,'fold_checkpoints',f'fold_{fold_idx+1}.json'),'w') as f:
        json.dump(fold_result, f, indent=2)

    print(f"\n  ✅ FOLD {fold_idx+1} | {test_subj}")
    print(f"     Accuracy (voted) : {acc_voted*100:.2f}%")
    print(f"     Macro F1         : {f1_macro*100:.2f}%")
    print(f"     Low F1           : {f1_low*100:.2f}%  "
          f"Moderate F1: {f1_mod*100:.2f}%  High F1: {f1_hi*100:.2f}%")
    print(f"     Balanced Acc     : {bal_acc*100:.2f}%")
    print(f"     Time: {fold_time:.1f} min | Elapsed: {elapsed:.1f} min | ~{remaining:.0f} min left")
    print(f"     RAM: {psutil.virtual_memory().percent:.1f}%")

    del (cnn_full, cnn_feat, rf,
         X_tr_raw, X_tr_norm, X_te_raw, X_te_norm,
         X_bal, y_bal, X_train, y_train, X_val, y_val,
         Fc_tr, Fs_tr, F_tr, F_tr_sc, Fc_te, Fs_te, F_te, F_te_sc)
    gc.collect(); keras.backend.clear_session()

# ── FINAL RESULTS ────────────────────────────────────────────────────────────

total_min    = (time.time()-total_start)/60
all_true_arr = np.array(all_true)
all_pred_arr = np.array(all_pred)
overall_acc  = accuracy_score(all_true_arr, all_pred_arr)
overall_f1   = f1_score(all_true_arr, all_pred_arr, average='macro',    zero_division=0)
overall_wf1  = f1_score(all_true_arr, all_pred_arr, average='weighted', zero_division=0)
overall_bal  = balanced_accuracy_score(all_true_arr, all_pred_arr)
full_report  = classification_report(all_true_arr, all_pred_arr,
                                      target_names=SHORT_NAMES, digits=4)
prec, rec, f1s, _ = precision_recall_fscore_support(
    all_true_arr, all_pred_arr, labels=[0,1,2], zero_division=0)
sub_accs = [d['acc_voted'] for d in fold_log]

print(f"\n{'═'*65}")
print(f"  ALL {len(available)} FOLDS COMPLETE  |  {total_min:.1f} min")
print(f"{'═'*65}")
print(f"  Accuracy (voted) : {np.mean(sub_accs)*100:.2f}% ± {np.std(sub_accs)*100:.2f}%")
print(f"  Macro F1         : {np.mean(results_f1)*100:.2f}% ± {np.std(results_f1)*100:.2f}%")
print(f"  Balanced Acc     : {np.mean(results_bal)*100:.2f}% ± {np.std(results_bal)*100:.2f}%")
print(f"\n  CLASSIFICATION REPORT (pooled LOSO):")
print(full_report)
print(f"  Best  : {max(fold_log,key=lambda x:x['acc_voted'])['subject']} ({max(sub_accs)*100:.2f}%)")
print(f"  Worst : {min(fold_log,key=lambda x:x['acc_voted'])['subject']} ({min(sub_accs)*100:.2f}%)")

# Save report
report_text = f"""WESAD THREE-CLASS MOD — Low (Baseline) / Moderate (Meditation) / High (Stress)
{"="*65}
Date       : {time.strftime("%Y-%m-%d %H:%M:%S")}
Label Map  : Baseline(1)→Low(0)  Meditation(4)→Moderate(1)  Stress(2)→High(2)  Amusement(3) DROPPED
Subjects   : {len(available)} LOSO  |  Window: {WINDOW_SEC}s  |  Stride: {SHIFT_SEC}s

AGGREGATE RESULTS:
  Accuracy (voted) : {np.mean(sub_accs)*100:.2f}% ± {np.std(sub_accs)*100:.2f}%
  Macro F1         : {np.mean(results_f1)*100:.2f}% ± {np.std(results_f1)*100:.2f}%
  Balanced Acc     : {np.mean(results_bal)*100:.2f}% ± {np.std(results_bal)*100:.2f}%

CLASSIFICATION REPORT (pooled LOSO):
{"─"*55}
{full_report}

PER-SUBJECT RESULTS:
{"─"*65}
{"Subject":<10} {"Voted%":>8} {"MacroF1":>9} {"LowF1":>8} {"ModF1":>8} {"HighF1":>8} {"BalAcc":>8}
{"─"*65}
"""
for r in fold_log:
    report_text += (f"{r['subject']:<10} {r['acc_voted']*100:>7.2f}% "
                    f"{r['f1_macro']*100:>8.2f}% "
                    f"{r['f1_low']*100:>7.2f}% "
                    f"{r['f1_moderate']*100:>7.2f}% "
                    f"{r['f1_high']*100:>7.2f}% "
                    f"{r['bal_acc']*100:>7.2f}%\n")
report_text += f"\n{'─'*65}\n"
report_text += (f"{'Mean':<10} {np.mean(sub_accs)*100:>7.2f}% "
                f"{np.mean(results_f1)*100:>8.2f}%\n")

with open(os.path.join(OUTPUT_DIR,'classification_report_mod.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)
print(f"\n  Report saved → {OUTPUT_DIR}")

# ── PLOTS ────────────────────────────────────────────────────────────────────
cm   = confusion_matrix(all_true_arr, all_pred_arr)
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig = plt.figure(figsize=(18,12))
import matplotlib.gridspec as gridspec
gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)
fig.suptitle(
    f'Three-Class Mod CNN+RF  |  WESAD (Low / Moderate / High)  |  Full LOSO\n'
    f'Acc: {overall_acc*100:.2f}%  |  Macro F1: {overall_f1*100:.2f}%  |  Bal Acc: {overall_bal*100:.2f}%',
    fontsize=12, fontweight='bold')

ax1 = fig.add_subplot(gs[0,0])
sns.heatmap(cm_n, annot=False, cmap='Blues', ax=ax1,
            xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
            linewidths=2, linecolor='white', vmin=0, vmax=1, cbar_kws={'shrink':0.8})
for i in range(N_CLASSES):
    for j in range(N_CLASSES):
        tc_col = 'white' if cm_n[i,j]>0.55 else 'black'
        ax1.text(j+0.5, i+0.38, f'{cm_n[i,j]*100:.1f}%', ha='center', va='center',
                 fontsize=13, fontweight='bold', color=tc_col)
        ax1.text(j+0.5, i+0.65, f'n={cm[i,j]}', ha='center', va='center',
                 fontsize=9, color=tc_col, alpha=0.9)
ax1.set_title('Confusion Matrix', fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12); ax1.set_xlabel('Predicted Label', fontsize=12)

ax2 = fig.add_subplot(gs[0,1])
sub_names = [d['subject'] for d in fold_log]
accs = [d['acc_voted'] for d in fold_log]
colors = ['#27ae60' if v>=0.75 else '#e67e22' if v>=0.60 else '#e74c3c' for v in accs]
x = np.arange(len(sub_names))
ax2.bar(x, [a*100 for a in accs], 0.5, color=colors, alpha=0.85, edgecolor='white')
ax2.axhline(np.mean(accs)*100, color='#2980b9', ls='--', lw=2,
            label=f'Mean {np.mean(accs)*100:.2f}%')
ax2.set_xticks(x); ax2.set_xticklabels(sub_names, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Accuracy (%)'); ax2.set_ylim(0, 115)
ax2.set_title('Per-Subject LOSO Accuracy', fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(axis='y', alpha=0.3)
ax2.spines[['top','right']].set_visible(False)

ax3 = fig.add_subplot(gs[1,0])
x3 = np.arange(N_CLASSES); w3 = 0.25
for idx,(d,lb,col) in enumerate(zip(
        [prec*100, rec*100, f1s*100],
        ['Precision','Recall','F1-Score'],
        ['#3498db','#2ecc71','#e74c3c'])):
    bars = ax3.bar(x3+(idx-1)*w3, d, w3, label=lb, color=col, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, d):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax3.axhline(overall_acc*100, color='black', ls='--', lw=2, alpha=0.6)
ax3.set_xticks(x3); ax3.set_xticklabels(SHORT_NAMES, fontsize=11)
ax3.set_ylabel('Score (%)'); ax3.set_ylim(0, 115)
ax3.set_title('Per-Class Precision / Recall / F1', fontweight='bold')
ax3.legend(fontsize=9); ax3.grid(axis='y', alpha=0.3)
ax3.spines[['top','right']].set_visible(False)

ax4 = fig.add_subplot(gs[1,1])
labs4 = ['Schmidt\n2018\n(3-class\nLOSO)','Old 3-class\n(SMOTE\nbuggy)','Mod Three-Class\n(this work)']
vals4 = [80.34, 81.37, overall_acc*100]
cols4 = ['#bdc3c7','#e67e22','#27ae60']
bars4 = ax4.bar(labs4, vals4, color=cols4, alpha=0.9, width=0.5, edgecolor='white')
for b, v in zip(bars4, vals4):
    ax4.text(b.get_x()+b.get_width()/2, b.get_height()+0.4,
             f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')
ax4.set_ylabel('LOSO Accuracy (%)'); ax4.set_ylim(0, 115)
ax4.set_title('Comparison vs Baselines', fontweight='bold')
ax4.grid(axis='y', alpha=0.3); ax4.spines[['top','right']].set_visible(False)

plt.savefig(os.path.join(OUTPUT_DIR, 'loso_results_mod.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print(f"  Plot saved → {OUTPUT_DIR}/loso_results_mod.png")

# ── TRAIN FINAL MODEL ────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print(f"  TRAINING FINAL DEPLOYMENT MODEL (all {len(available)} subjects)")
print(f"{'═'*65}")
t0 = time.time()

X_all_raw = np.concatenate([subject_data[s][0] for s in available])
y_all     = np.concatenate([subject_data[s][1] for s in available])

X_all_norm = X_all_raw.copy()
norm_stats = {}
for ch in range(N_CHANNELS):
    mu  = X_all_raw[:,:,ch].mean(); sig = X_all_raw[:,:,ch].std() + 1e-8
    X_all_norm[:,:,ch] = (X_all_raw[:,:,ch] - mu) / sig
    norm_stats[ch] = {'mean': float(mu), 'std': float(sig)}
X_all_norm = X_all_norm.astype(np.float32)

X_bal_f, y_bal_f = balance_with_augmentation(
    X_all_norm, y_all, target_ratio=1.0, noise_std=0.015, seed=RANDOM_STATE)
c_f = Counter(y_bal_f.tolist())
print(f"  After augmentation: {len(y_bal_f)} windows  {dict(c_f)}")

cw_arr = compute_class_weight('balanced', classes=np.unique(y_bal_f), y=y_bal_f)
cw = {int(c):float(w) for c,w in zip(np.unique(y_bal_f), cw_arr)}
cw[2] = cw.get(2, 1.0) * 1.5

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
tr_i, vl_i = next(sss.split(X_bal_f, y_bal_f))

keras.backend.clear_session(); gc.collect()
cnn_full_f, cnn_feat_f = build_cnn()
cnn_full_f.compile(optimizer=keras.optimizers.Adam(2e-4, clipnorm=1.0),
                   loss='categorical_crossentropy', metrics=['accuracy'])
print("  Training final CNN...")
cnn_full_f.fit(
    X_bal_f[tr_i], to_categorical(y_bal_f[tr_i], N_CLASSES),
    validation_data=(X_bal_f[vl_i], to_categorical(y_bal_f[vl_i], N_CLASSES)),
    epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=cw,
    callbacks=[EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)],
    verbose=1)

Fc_all   = cnn_feat_f.predict(X_all_norm, batch_size=64, verbose=0)
Fs_all   = extract_features(X_all_norm, fs=TARGET_FS)
F_all    = np.concatenate([Fc_all, Fs_all], axis=1)
scaler   = StandardScaler()
F_all_sc = scaler.fit_transform(F_all)

rf_final = RandomForestClassifier(
    n_estimators=RF_ESTIMATORS, max_features='sqrt',
    class_weight={0:1.0, 1:1.0, 2:2.0},
    min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE)
rf_final.fit(F_all_sc, y_all)

h5_path  = os.path.join(OUTPUT_DIR, 'cnn_feat_mod.h5')
rf_path  = os.path.join(OUTPUT_DIR, 'rf_mod.pkl')
sc_path  = os.path.join(OUTPUT_DIR, 'scaler_mod.pkl')
cfg_path = os.path.join(OUTPUT_DIR, 'config_mod.pkl')

cnn_feat_f.save(h5_path)
with open(rf_path,  'wb') as f: pickle.dump(rf_final, f)
with open(sc_path,  'wb') as f: pickle.dump(scaler,   f)
with open(cfg_path, 'wb') as f:
    pickle.dump({'CLASS_NAMES': SHORT_NAMES, 'LABEL_MAP': LABEL_MAP,
                 'WINDOW_SIZE': WINDOW_SIZE, 'STRIDE_SEC': SHIFT_SEC,
                 'TARGET_FS': TARGET_FS, 'N_CHANNELS': N_CHANNELS,
                 'N_CLASSES': N_CLASSES, 'FEATURE_DIM': 72,
                 'HC_FEATURE_DIM': 32, 'TOTAL_FEATURES': 104,
                 'NORM_STATS': norm_stats}, f)

print(f"\n{'═'*65}")
print(f"  FILES SAVED → {OUTPUT_DIR}")
print(f"{'═'*65}")
print(f"  classification_report_mod.txt")
print(f"  loso_results_mod.png")
print(f"  cnn_feat_mod.h5")
print(f"  rf_mod.pkl  /  scaler_mod.pkl  /  config_mod.pkl")
print(f"\n  Accuracy (voted) : {np.mean(sub_accs)*100:.2f}% ± {np.std(sub_accs)*100:.2f}%")
print(f"  Macro F1         : {np.mean(results_f1)*100:.2f}% ± {np.std(results_f1)*100:.2f}%")
print(f"  Balanced Acc     : {np.mean(results_bal)*100:.2f}% ± {np.std(results_bal)*100:.2f}%")
print(f"  Total time       : {(time.time()-total_start)/60:.1f} min")
print("✅ Three-class mod pipeline complete")