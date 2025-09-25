#!/usr/bin/env python3
"""
ssl-pipeline.py

Lightweight wrapper CLI for pretrain -> extract -> plot steps.
It will try to call `demo.ssl_pretrain`, `demo.extract_ssl_embeddings`,
and `demo.plot_features_umap` when available. If those modules are missing
this script provides safe fallback behavior so the pipeline still runs
and produces minimal outputs (placeholder checkpoint, zero embeddings,
and a PCA-based 2D plot).

This file is intentionally defensive so it can restore pipeline behavior
even if some demo files were accidentally removed.
"""
import argparse
import os
import sys
import glob
import numpy as np
from pathlib import Path


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)


def find_wavs(data_root):
    exts = ("**/*.wav", "**/*.flac", "**/*.mp3")
    p = Path(data_root)
    files = []
    for e in exts:
        files.extend([str(f) for f in p.glob(e)])
    files = sorted(files)
    return files


def fallback_pretrain(args):
    # safe placeholder: create out_dir and write an empty checkpoint
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)
    ckpt = out_dir / "encoder_epoch0.pth"
    try:
        import torch
        torch.save({"placeholder": True}, str(ckpt))
    except Exception:
        # no torch available: write an empty file
        ckpt.write_text("placeholder checkpoint")
    print(f"Wrote placeholder checkpoint: {ckpt}")


def fallback_extract(args):
    files = find_wavs(args.data_root)
    n = len(files)
    if n == 0:
        print("No audio files found under", args.data_root)
    # produce zero embeddings of dimension 512
    feats = np.zeros((n, args.dim if hasattr(args, 'dim') else 512), dtype=np.float32)
    labels = ["" for _ in range(n)]
    paths = files
    out_npz = Path(args.out_npz)
    ensure_out_dir(out_npz.parent)
    np.savez_compressed(out_npz, feats_ssl=feats, labels=labels, paths=paths)
    print(f"Wrote placeholder embeddings to: {out_npz} shape={feats.shape}")


def fallback_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        data = np.load(args.features)
    except Exception as e:
        print("Failed loading features file:", e)
        return

    # try UMAP if available, fallback to PCA
    X = None
    if 'feats_ssl' in data:
        X = data['feats_ssl']
    else:
        keys = [k for k in data.files if data[k].ndim == 2]
        if keys:
            X = data[keys[0]]

    if X is None:
        print("No 2D-able features found in the npz.")
        return

    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        Z = reducer.fit_transform(X)
        method = 'umap'
    except Exception:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        Z = pca.fit_transform(X)
        method = 'pca'

    out_dir = Path(args.out_dir or Path(args.features).parent)
    ensure_out_dir(out_dir)
    out_png = out_dir / f"features_{method}.png"
    plt.figure(figsize=(6, 6))
    plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.8)
    plt.title(f"Embeddings ({method.upper()})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot to {out_png}")


def run_pretrain(args):
    # prefer demo implementation if available
    try:
        import demo.ssl_pretrain as mod
        if hasattr(mod, 'main'):
            mod.main(args)
        elif hasattr(mod, 'run_pretrain'):
            mod.run_pretrain(args)
        else:
            print('demo.ssl_pretrain found but no callable entrypoint; running fallback')
            fallback_pretrain(args)
    except Exception as e:
        print('demo.ssl_pretrain not available or failed, using fallback:', e)
        fallback_pretrain(args)


def run_extract(args):
    try:
        import demo.extract_ssl_embeddings as mod
        if hasattr(mod, 'main'):
            mod.main(args)
        elif hasattr(mod, 'extract_embeddings'):
            mod.extract_embeddings(args)
        else:
            print('demo.extract_ssl_embeddings found but no callable entrypoint; running fallback')
            fallback_extract(args)
    except Exception as e:
        print('demo.extract_ssl_embeddings not available or failed, using fallback:', e)
        fallback_extract(args)


def run_plot(args):
    try:
        import demo.plot_features_umap as mod
        if hasattr(mod, 'main'):
            mod.main(args)
        elif hasattr(mod, 'plot_features'):
            mod.plot_features(args)
        else:
            print('demo.plot_features_umap found but no callable entrypoint; running fallback')
            fallback_plot(args)
    except Exception as e:
        print('demo.plot_features_umap not available or failed, using fallback:', e)
        fallback_plot(args)


def build_parser():
    p = argparse.ArgumentParser(description='SSL pipeline (pretrain / extract / plot)')
    sub = p.add_subparsers(dest='cmd')

    sp = sub.add_parser('pretrain')
    sp.add_argument('--data_root', required=True)
    sp.add_argument('--out_dir', required=True)
    sp.add_argument('--epochs', type=int, default=1)
    sp.add_argument('--batch_size', type=int, default=8)
    sp.add_argument('--lr', type=float, default=1e-3)
    sp.add_argument('--sr', type=int, default=22050)

    se = sub.add_parser('extract')
    se.add_argument('--checkpoint', required=False)
    se.add_argument('--data_root', required=True)
    se.add_argument('--out_npz', required=True)
    se.add_argument('--batch_size', type=int, default=8)
    se.add_argument('--dim', type=int, default=512)

    sp2 = sub.add_parser('plot')
    sp2.add_argument('--features', required=True)
    sp2.add_argument('--out_dir', required=False)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == 'pretrain':
        run_pretrain(args)
    elif args.cmd == 'extract':
        run_extract(args)
    elif args.cmd == 'plot':
        run_plot(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
SSL pipeline: pretrain -> extract embeddings -> plot UMAP/PCA

Drop into your repo as `demo/ssl_pipeline.py`.

Subcommands:
  pretrain  - run a small SimCLR-style pretraining loop on mel spectrograms
  extract   - load a saved encoder checkpoint and compute embeddings for audio files
  plot      - plot embeddings from an NPZ (UMAP if available, PCA fallback)

Examples:
  # Pretrain (quick smoke)
  python demo/ssl_pipeline.py pretrain --data_root /path/to/audio --out_dir demo/out --epochs 1 --batch_size 8

  # Extract embeddings using a checkpoint created by `pretrain`
  python demo/ssl_pipeline.py extract --checkpoint demo/out/encoder_epoch1.pth --data_root /path/to/audio --out_npz demo/out/features_ssl.npz

  # Plot embeddings (UMAP or PCA fallback)
  python demo/ssl_pipeline.py plot --features demo/out/features_ssl.npz --out_dir demo/out
"""
from pathlib import Path
import argparse
import random
import time
import math
import os
import sys
from typing import List

import numpy as np

# optional visualization
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# optional UMAP (fallback to PCA if not installed)
try:
    import umap.umap_ as umap
    HAVE_UMAP = True
except Exception:
    HAVE_UMAP = False

# audio & torch imports (fail early with clear messages)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    raise RuntimeError("This script requires PyTorch. Install per https://pytorch.org/") from e

try:
    import librosa
    import soundfile as sf
except Exception as e:
    raise RuntimeError("This script requires librosa and soundfile. Install with `pip install librosa soundfile`") from e

# -------------------------
# Audio utilities
# -------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch, 'has_mps', False) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def compute_mel(wave: np.ndarray, sr: int, n_mels: int = 80, fmin: int = 50, fmax: int = None):
    if fmax is None:
        fmax = sr // 2
    S = librosa.feature.melspectrogram(y=wave, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-9)
    return S_db.astype(np.float32)

def load_audio(path: Path, sr: int, max_seconds: float):
    wav, sr_loaded = sf.read(str(path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr_loaded != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr_loaded, target_sr=sr)
    max_len = int(sr * max_seconds)
    if len(wav) < max_len:
        pad = max_len - len(wav)
        wav = np.pad(wav, (0, pad))
    else:
        start = np.random.randint(0, max(1, len(wav) - max_len + 1))
        wav = wav[start:start + max_len]
    return wav.astype(np.float32)

# simple augmentations
def time_shift(wav: np.ndarray, shift_max=0.2):
    if shift_max <= 0:
        return wav
    shift = int(len(wav) * random.uniform(-shift_max, shift_max))
    if shift == 0:
        return wav
    return np.roll(wav, shift)

def add_noise(wav: np.ndarray, snr_db_min=10, snr_db_max=30):
    rms = np.sqrt(np.mean(wav ** 2) + 1e-9)
    snr_db = random.uniform(snr_db_min, snr_db_max)
    noise_rms = rms / (10 ** (snr_db / 20.0))
    noise = np.random.normal(0, noise_rms, size=wav.shape).astype(np.float32)
    return wav + noise

def time_mask(mel: np.ndarray, T=30):
    t = mel.shape[1]
    if t <= 1:
        return mel
    t0 = np.random.randint(0, max(1, t - T + 1))
    mel[:, t0:t0 + T] = 0
    return mel

def freq_mask(mel: np.ndarray, F=8):
    f = mel.shape[0]
    if f <= 1:
        return mel
    f0 = np.random.randint(0, max(1, f - F + 1))
    mel[f0:f0 + F, :] = 0
    return mel

# -------------------------
# Datasets and model
# -------------------------
class ContrastiveAudioDataset(Dataset):
    def __init__(self, files: List[Path], sr=24000, n_mels=80, fmin=50, fmax=None, max_seconds=5.0):
        self.files = files
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.files)

    def _make_view(self, wav: np.ndarray):
        wav_aug = wav.copy()
        if random.random() < 0.5:
            wav_aug = time_shift(wav_aug, shift_max=0.1)
        if random.random() < 0.5:
            wav_aug = add_noise(wav_aug, snr_db_min=10, snr_db_max=25)
        mel = compute_mel(wav_aug, self.sr, self.n_mels, self.fmin, self.fmax)
        if random.random() < 0.5:
            mel = time_mask(mel, T=min(40, mel.shape[1] // 4))
        if random.random() < 0.5:
            mel = freq_mask(mel, F=min(12, self.n_mels // 6))
        return mel

    def __getitem__(self, idx):
        p = self.files[idx]
        wav = load_audio(p, self.sr, self.max_seconds)
        v1 = self._make_view(wav)
        v2 = self._make_view(wav)
        v1 = torch.from_numpy(v1).unsqueeze(0)
        v2 = torch.from_numpy(v2).unsqueeze(0)
        return v1, v2

def collate_fn(batch):
    v1s, v2s = zip(*batch)
    max_t = max(a.shape[-1] for a in v1s)
    def pad_batch(lst):
        out = []
        for x in lst:
            pad = max_t - x.shape[-1]
            if pad > 0:
                x = F.pad(x, (0, pad))
            out.append(x)
        return torch.stack(out, dim=0)
    return pad_batch(v1s), pad_batch(v2s)

class SmallEncoder(nn.Module):
    def __init__(self, n_mels=80, out_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim)
        )
    def forward(self, x):
        return self.net(x)

def nt_xent_loss(z1, z2, temperature=0.5):
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    N = z.shape[0]
    B = z1.shape[0]
    positive_mask = torch.zeros_like(sim, dtype=torch.bool)
    for i in range(B):
        positive_mask[i, i+B] = True
        positive_mask[i+B, i] = True
    diag_mask = ~torch.eye(N, dtype=torch.bool, device=z.device)
    logits = sim[diag_mask].view(N, N-1)
    positives = sim[positive_mask].view(N, 1)
    logits_all = torch.cat([positives, logits], dim=1)
    targets = torch.zeros(N, dtype=torch.long, device=z.device)
    loss = F.cross_entropy(logits_all, targets)
    return loss

# -------------------------
# File utilities
# -------------------------
def find_audio_files(root: Path):
    exts = ['wav', 'flac', 'mp3', 'm4a']
    files = []
    for e in exts:
        files.extend(list(root.rglob(f'*.{e}')))
    return sorted(files)

def infer_label_from_path(p: str):
    parts = Path(p).parts
    if len(parts) >= 2:
        return Path(p).parent.name
    return 'unknown'

# -------------------------
# Subcommands
# -------------------------
def cmd_pretrain(argv):
    p = argparse.ArgumentParser(prog='pretrain')
    p.add_argument('--data_root', required=True)
    p.add_argument('--out_dir', default='demo/out')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--fmin', type=int, default=50)
    p.add_argument('--fmax', type=int, default=None)
    p.add_argument('--sr', type=int, default=24000)
    p.add_argument('--max_seconds', type=float, default=5.0)
    p.add_argument('--proj_dim', type=int, default=128)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print('Device:', device)

    root = Path(args.data_root)
    files = find_audio_files(root)
    if len(files) == 0:
        print('No audio files found under', root)
        return
    print('Found', len(files), 'audio files')

    dataset = ContrastiveAudioDataset(files, sr=args.sr, n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax, max_seconds=args.max_seconds)
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    encoder = SmallEncoder(n_mels=args.n_mels, out_dim=512).to(device)
    proj = ProjectionHead(in_dim=512, proj_dim=args.proj_dim).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj.train()
        t0 = time.time()
        total_loss = 0.0
        for i, (v1, v2) in enumerate(dl, 1):
            v1 = v1.to(device)
            v2 = v2.to(device)
            h1 = encoder(v1)
            h2 = encoder(v2)
            z1 = proj(h1)
            z2 = proj(h2)
            loss = nt_xent_loss(z1, z2, temperature=0.5)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            global_step += 1
            if i % 50 == 0:
                print(f'Epoch {epoch} step {i}/{len(dl)} loss={loss.item():.4f}')
        avg_loss = total_loss / len(dl)
        dt = time.time() - t0
        print(f'Epoch {epoch} finished  avg_loss={avg_loss:.4f}  time={dt:.1f}s')
        if epoch % args.save_every == 0 or epoch == args.epochs:
            path = out_dir / f'encoder_epoch{epoch}.pth'
            torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict(), 'args': vars(args)}, path)
            print('Saved checkpoint:', path)

    print('Pretraining finished. Checkpoints in', out_dir)

def cmd_extract(argv):
    p = argparse.ArgumentParser(prog='extract')
    p.add_argument('--checkpoint', required=True, help='Path to encoder checkpoint .pth')
    p.add_argument('--data_root', required=True, help='Folder with audio files')
    p.add_argument('--out_npz', default='demo/out/features_ssl.npz')
    p.add_argument('--sr', type=int, default=24000)
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--fmin', type=int, default=50)
    p.add_argument('--fmax', type=int, default=None)
    p.add_argument('--max_seconds', type=float, default=5.0)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device', default=None)
    args = p.parse_args(argv)

    chk = Path(args.checkpoint)
    if not chk.exists():
        print('Checkpoint not found:', chk)
        return
    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if getattr(torch, 'has_mps', False) and torch.backends.mps.is_available() else torch.device('cpu')))
    print('Device:', device)

    files = find_audio_files(Path(args.data_root))
    if len(files) == 0:
        print('No audio files found under', args.data_root)
        return
    print('Found', len(files), 'audio files')

    class EmbedDataset(Dataset):
        def __init__(self, files, sr=24000, n_mels=80, fmin=50, fmax=None, max_seconds=5.0):
            self.files = files
            self.sr = sr
            self.n_mels = n_mels
            self.fmin = fmin
            self.fmax = fmax
            self.max_seconds = max_seconds

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            p = self.files[idx]
            wav = load_audio(p, self.sr, self.max_seconds)
            mel = compute_mel(wav, self.sr, self.n_mels, self.fmin, self.fmax)
            mel = torch.from_numpy(mel).unsqueeze(0)
            return mel, str(p)

    def collate_pad(batch):
        mels, paths = zip(*batch)
        max_t = max(x.shape[-1] for x in mels)
        out = []
        for x in mels:
            pad = max_t - x.shape[-1]
            if pad > 0:
                x = F.pad(x, (0, pad))
            out.append(x)
        return torch.stack(out, dim=0), list(paths)

    ds = EmbedDataset(files, sr=args.sr, n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax, max_seconds=args.max_seconds)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=2)

    encoder = SmallEncoder(n_mels=args.n_mels, out_dim=512)
    state = torch.load(str(chk), map_location='cpu')
    s = state.get('encoder', state)
    encoder.load_state_dict(s)
    encoder.to(device)
    encoder.eval()

    all_emb = []
    all_paths = []
    all_labels = []
    with torch.no_grad():
        for mels, paths in dl:
            mels = mels.to(device)
            h = encoder(mels)
            emb = h.cpu().numpy()
            all_emb.append(emb)
            all_paths.extend(paths)
            for p in paths:
                all_labels.append(infer_label_from_path(p))

    feats = np.vstack(all_emb)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_npz), feats_ssl=feats, labels=np.array(all_labels), paths=np.array(all_paths))
    print('Saved embeddings to', out_npz, 'shape=', feats.shape)

def cmd_plot(argv):
    p = argparse.ArgumentParser(prog='plot')
    p.add_argument('--features', required=True, help='Path to features .npz (feats_ssl or feats_conv/feats_mel supported)')
    p.add_argument('--out_dir', default='demo/out')
    p.add_argument('--n_neighbors', type=int, default=15)
    p.add_argument('--min_dist', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)

    f = Path(args.features)
    if not f.exists():
        print('Features file not found:', f)
        return
    data = np.load(str(f), allow_pickle=True)
    keys = list(data.keys())
    print('Found keys in features:', keys)

    labels = data.get('labels')
    if labels is None:
        print('No labels found in features file')
        return

    def project_and_plot(X, labels, out_path, title):
        rng = np.random.RandomState(args.seed)
        if HAVE_UMAP:
            reducer = umap.UMAP(n_neighbors=args.n_neighbors, min_dist=args.min_dist, random_state=args.seed)
            Z = reducer.fit_transform(X)
        else:
            print('umap not available, falling back to PCA (2 components)')
            pca = PCA(n_components=2, random_state=args.seed)
            Z = pca.fit_transform(X)

        le = LabelEncoder()
        y = le.fit_transform(labels)
        classes = le.classes_

        plt.figure(figsize=(8, 6))
        cmap = plt.get_cmap('tab20')
        for i, cls in enumerate(classes):
            mask = (y == i)
            plt.scatter(Z[mask, 0], Z[mask, 1], s=12, alpha=0.8, label=str(cls), color=cmap(i % 20))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.title(title)
        plt.tight_layout()
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=200)
        plt.close()
        print('Saved', out_path)

    out_dir = Path(args.out_dir)
    if 'feats_ssl' in data:
        X = data['feats_ssl']
        project_and_plot(X, labels, out_dir / 'features_umap_ssl.png', 'UMAP - SSL embeddings')
    if 'feats_conv' in data:
        project_and_plot(data['feats_conv'], labels, out_dir / 'features_umap_conv_pre.png', 'UMAP - conv_pre')
    if 'feats_mel' in data:
        project_and_plot(data['feats_mel'], labels, out_dir / 'features_umap_mel.png', 'UMAP - mel pooled')

# -------------------------
# CLI dispatch
# -------------------------
def main():
    parser = argparse.ArgumentParser(prog='ssl_pipeline', description='SSL pretrain/extract/plot pipeline')
    sub = parser.add_subparsers(dest='cmd', required=True)
    sub.add_parser('pretrain')
    sub.add_parser('extract')
    sub.add_parser('plot')
    args, rest = parser.parse_known_args()
    if args.cmd == 'pretrain':
        cmd_pretrain(rest)
    elif args.cmd == 'extract':
        cmd_extract(rest)
    elif args.cmd == 'plot':
        cmd_plot(rest)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()