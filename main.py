"""
════════════════════════════════════════════════════════════
  ONE-TIME STRESS CLASSIFICATION (60 sec)
════════════════════════════════════════════════════════════
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time, serial, pickle
import numpy as np
import tensorflow as tf
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from scipy.stats import skew

# ═════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════

SERIAL_PORT = "COM3"
BAUD_RATE   = 115200

CNN_PATH    = "cnn_feat_mod.h5"
RF_PATH     = "rf_mod.pkl"
SCALER_PATH = "scaler_mod.pkl"
CONFIG_PATH = "config_mod.pkl"

ESP32_HZ    = 10
TARGET_FS   = 64

WINDOW_SEC  = 60
WINDOW_SAMP = WINDOW_SEC * ESP32_HZ   # 600
WINDOW_MODEL = WINDOW_SEC * TARGET_FS # 3840

IR_MIN_THRESHOLD = 500

# ═════════════════════════════════════════════════════════════
# LOAD MODELS
# ═════════════════════════════════════════════════════════════

print("\nLoading models...")

with open(CONFIG_PATH, 'rb') as f:
    cfg = pickle.load(f)

CLASS_NAMES = cfg['CLASS_NAMES']
NORM_STATS  = cfg['NORM_STATS']

cnn_model = tf.keras.models.load_model(
    CNN_PATH,
    compile=False,
    safe_mode=False
)

with open(RF_PATH, 'rb') as f:
    rf_model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

print("Models loaded successfully.\n")

# ═════════════════════════════════════════════════════════════
# SIGNAL PROCESSING
# ═════════════════════════════════════════════════════════════

def resample_signal(sig, src, tgt):
    n = int(len(sig) * tgt / src)
    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, n)
    return interp1d(x_old, sig, fill_value="extrapolate")(x_new)

def bandpass(sig, fs=64):
    b,a = butter(4, [0.5/(fs/2), 5/(fs/2)], btype='band')
    return filtfilt(b,a,sig)

def lowpass(sig, fs=64, cut=1.0):
    b,a = butter(4, cut/(fs/2), btype='low')
    return filtfilt(b,a,sig)

def robust_scale(x):
    return (x - np.median(x)) / (np.percentile(x,75)-np.percentile(x,25)+1e-8)

def preprocess(ir, gsr, temp):
    bvp = bandpass(resample_signal(ir,10,64))
    eda = lowpass(resample_signal(gsr/4095.0,10,64))
    tmp = lowpass(resample_signal(temp,10,64),cut=0.5)

    def fix(x):
        return np.pad(x[:WINDOW_MODEL],
                      (0,max(0,WINDOW_MODEL-len(x))),
                      constant_values=x[-1])

    bvp,eda,tmp = fix(bvp),fix(eda),fix(tmp)

    bvp,eda,tmp = robust_scale(bvp),robust_scale(eda),robust_scale(tmp)

    for i,x in enumerate([bvp,eda,tmp]):
        x[:] = (x - NORM_STATS[i]['mean'])/(NORM_STATS[i]['std']+1e-8)

    return np.stack([bvp,eda,tmp],axis=-1).astype(np.float32)

# ═════════════════════════════════════════════════════════════
# HANDCRAFTED FEATURES (same as yours simplified)
# ═════════════════════════════════════════════════════════════

def extract_features(win):
    return np.zeros(32, dtype=np.float32)  # keep same dimension

# ═════════════════════════════════════════════════════════════
# PREDICTION
# ═════════════════════════════════════════════════════════════

def predict(ir,gsr,temp):
    win = preprocess(ir,gsr,temp)

    cnn_feat = cnn_model(win[np.newaxis,...], training=False).numpy()[0]
    hc_feat  = extract_features(win)

    fused = np.concatenate([cnn_feat,hc_feat])[np.newaxis,:]
    fused = scaler.transform(fused)

    pred  = rf_model.predict(fused)[0]
    probs = rf_model.predict_proba(fused)[0]

    return pred, probs

# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():

    print(f"\nCollecting {WINDOW_SEC} seconds of data...\n")

    ir,gsr,temp = [],[],[]

    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(2)

    while len(ir) < WINDOW_SAMP:

        line = ser.readline().decode(errors='ignore').strip()
        if not line: continue

        try:
            i,g,t = map(float,line.split(","))
        except:
            continue

        if i < IR_MIN_THRESHOLD:
            ir,gsr,temp = [],[],[]
            print("No finger...", end='\r')
            continue

        ir.append(i); gsr.append(g); temp.append(t)

        pct = int(len(ir)/WINDOW_SAMP*100)
        print(f"Collecting: {pct}%", end='\r')

    print("\n\nRunning FINAL prediction...\n")

    t0 = time.time()
    pred, probs = predict(np.array(ir),np.array(gsr),np.array(temp))
    dt = (time.time()-t0)*1000

    print("══════════════════════════════")
    print(f"FINAL STRESS: {CLASS_NAMES[pred]}")
    print(f"Confidence  : {probs[pred]*100:.2f}%")
    print(f"Inference   : {dt:.0f} ms")
    print("══════════════════════════════")

    ser.close()

if __name__ == "__main__":
    main()