import os
import sys
# ensure the local BigVGAN package directory is on sys.path so imports work
# whether the script is run from repo root or from inside the subfolder.
script_dir = os.path.dirname(os.path.abspath(__file__))
local_pkg = os.path.join(script_dir, "BigVGAN")
if os.path.isdir(local_pkg) and local_pkg not in sys.path:
    sys.path.insert(0, local_pkg)

import hashlib
import numpy as np
import torch
import librosa
import soundfile as sf

import bigvgan
from meldataset import get_mel_spectrogram

# ---------- helpers ----------
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_wav(path, y, sr):
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr, subtype="PCM_16")

def rms(a): return float(np.sqrt(np.mean(a*a) + 1e-12))

def align_by_xcorr(a, b, max_shift=2000):
    # returns aligned copies (same length) and chosen shift
    best, best_s = -1e9, 0
    for s in range(-max_shift, max_shift+1):
        if s >= 0:
            aa, bb = a[s:], b[:len(a)-s]
        else:
            aa, bb = a[:len(a)+s], b[-s:]
        # ensure aa and bb are same length for the dot product
        L = min(len(aa), len(bb))
        if L < 100:
            continue
        aa_trunc = aa[:L]
        bb_trunc = bb[:L]
        c = np.dot(aa_trunc, bb_trunc) / (np.linalg.norm(aa_trunc)*np.linalg.norm(bb_trunc) + 1e-12)
        if c > best:
            best, best_s = c, s
    if best_s >= 0:
        aa, bb = a[best_s:], b[:len(a)-best_s]
    else:
        aa, bb = a[:len(a)+best_s], b[-best_s:]
    L = min(len(aa), len(bb))
    return aa[:L], bb[:L], best_s, best

def lsd(a, b, n_fft=1024, hop=256):
    Sa = np.abs(librosa.stft(a, n_fft=n_fft, hop_length=hop)) + 1e-8
    Sb = np.abs(librosa.stft(b, n_fft=n_fft, hop_length=hop)) + 1e-8
    La, Lb = 20*np.log10(Sa), 20*np.log10(Sb)
    return float(np.mean(np.sqrt(np.mean((La - Lb)**2, axis=0))))

def si_sdr(a, b):
    a = a - np.mean(a); b = b - np.mean(b)
    s = (np.dot(b, a) / (np.dot(a, a) + 1e-12)) * a
    e = b - s
    return float(10*np.log10((np.dot(s, s)+1e-12)/(np.dot(e, e)+1e-12)))

# ---------- device ----------
device = "cpu" if torch.backends.mps.is_available() else "cpu"

# ---------- load model ----------
model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_24khz_100band_256x",
    use_cuda_kernel=False  # no CUDA on Mac
).eval().to(device)
model.remove_weight_norm()
sr = model.h.sampling_rate

# ---------- paths ----------
in_path = "/Users/denaclink/Downloads/CR_01_001.01.wav"               # <-- change this
# create an output folder next to this script so files are written
# predictably regardless of the current working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(script_dir, "bigvgan_check")
os.makedirs(out_dir, exist_ok=True)
syn_path = os.path.join(out_dir, "synth.wav")
syn_mask_path = os.path.join(out_dir, "synth_masked.wav")

# ---------- original ----------
y, _ = librosa.load(in_path, sr=sr, mono=True)
y_t = torch.from_numpy(y).float().unsqueeze(0)   # [1, T]
mel = get_mel_spectrogram(y_t, model.h).to(device)

# ---------- synth from true mel ----------
with torch.inference_mode():
    y_hat = model(mel).squeeze().detach().cpu().numpy()
# gain-match & align
y_hat *= (rms(y) / max(rms(y_hat), 1e-12))
ya, yha, shift, xcorr = align_by_xcorr(y, y_hat)
write_wav(syn_path, y_hat, sr)
print("Wrote:", os.path.abspath(syn_path))

# ---------- quick comparisons ----------
same_samples = (len(y) == len(y_hat)) and np.allclose(y, y_hat, atol=0.0)  # exact equality
print("Exact sample-by-sample identical? ", same_samples)
print("SHA256 original: ", sha256(in_path))
print("SHA256 synth   : ", sha256(syn_path))

print("Aligned corr   : ", xcorr)
print("SI-SDR (dB)    : ", si_sdr(ya, yha))
print("LSD   (dB)     : ", lsd(ya, yha))

# ---------- prove it's not a passthrough: perturb mel and resynthesize ----------
mel_mask = mel.clone()
T = mel_mask.shape[-1]
t0 = int(0.30*T); t1 = int(0.40*T)
mel_mask[:,:,t0:t1] = 0.0  # zero a 10% time band

with torch.inference_mode():
    y_hat_mask = model(mel_mask).squeeze().detach().cpu().numpy()
y_hat_mask *= (rms(y) / max(rms(y_hat_mask), 1e-12))
write_wav(syn_mask_path, y_hat_mask, sr)
print("Wrote:", os.path.abspath(syn_mask_path))

# report diffs between synth and masked-synth
ya2, yha2, _, _ = align_by_xcorr(y_hat, y_hat_mask)
print("Synth vs Masked-Synth SI-SDR (dB): ", si_sdr(ya2, yha2))
print("Wrote:", syn_path, "and", syn_mask_path)
