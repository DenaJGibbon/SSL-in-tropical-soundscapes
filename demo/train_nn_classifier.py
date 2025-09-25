#!/usr/bin/env python3
"""
Train a small neural-network classifier on folder-structured audio.

Usage (example):
  python demo/train_nn_classifier.py --input_dir /path/to/data --out_dir demo/out --epochs 10 --batch_size 16 --fmin 500 --fmax 3000

The data layout expected is:
  input_dir/label1/*.wav
  input_dir/label2/*.wav

This script is intentionally small and robust for local runs (CPU/MPS/GPU).
"""
import argparse
import os
import sys
from pathlib import Path
import time
import json

import numpy as np
import librosa
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


# make local BigVGAN package importable when requested
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
local_pkg = os.path.join(repo_root, "BigVGAN")
if os.path.isdir(local_pkg) and local_pkg not in sys.path:
    sys.path.insert(0, local_pkg)



class AudioFolderDataset(Dataset):
    def __init__(self, files, labels, sr=22050, n_mels=80, fmin=0, fmax=None, duration=None):
        self.files = files
        self.labels = labels
        self.sr = sr
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        y, _ = librosa.load(p, sr=self.sr, mono=True, duration=self.duration)
        # compute log-mel and global mean pooling to fixed vector
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax)
        logS = librosa.power_to_db(S, ref=np.max)
        feat = np.mean(logS, axis=1).astype(np.float32)  # (n_mels,)
        label = self.labels[idx]
        return feat, label, str(p)


class SimpleMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x):
        return self.net(x)


def collate_batch(batch):
    feats = np.stack([b[0] for b in batch], axis=0)
    labels = np.array([b[1] for b in batch], dtype=np.int64)
    paths = [b[2] for b in batch]
    return torch.from_numpy(feats), torch.from_numpy(labels), paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--fmin', type=float, default=0.0)
    p.add_argument('--fmax', type=float, default=None)
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--n_mels', type=int, default=80)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--use_bigvgan_features', action='store_true', help='Use BigVGAN conv_pre pooled features instead of mel mean pooling')
    p.add_argument('--bigvgan_dim', type=int, default=512, help='Dimensionality to pool conv_pre features to (via mean over time)')
    p.add_argument('--compare_conv_mel', action='store_true', help='Train both conv_pre and mel classifiers and output combined CSV')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def gather_files(input_dir):
    p = Path(input_dir)
    classes = [d for d in sorted(p.iterdir()) if d.is_dir()]
    files = []
    labels = []
    class_map = {}
    for i, c in enumerate(classes):
        class_map[i] = c.name
        for f in c.glob('**/*'):
            try:
                if not f.is_file():
                    continue
                name = f.name
                # skip macOS resource-fork files like '._foo.wav'
                if name.startswith('._'):
                    continue
                if f.suffix.lower() not in ('.wav', '.flac', '.mp3'):
                    continue
                # skip tiny or likely-corrupt files
                try:
                    if f.stat().st_size < 100:
                        continue
                except Exception:
                    continue
                files.append(str(f))
                labels.append(i)
            except Exception:
                # ignore files we can't stat or access
                continue
    return files, labels, class_map


def train_loop(model, opt, loss_fn, loader, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb = xb.to(device).float()
        yb = yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    return avg_loss, acc


def eval_loop(model, loss_fn, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                xb, yb, _ = batch
            else:
                xb, yb = batch
            xb = xb.to(device).float()
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
            all_probs.append(probs)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    if all_probs:
        all_probs = np.vstack(all_probs)
    else:
        all_probs = np.zeros((len(all_labels), 1))
    return avg_loss, acc, np.array(all_labels), np.array(all_preds), all_probs


def compute_auc(y_true, y_probs):
    # y_true: (N,) int labels; y_probs: (N, C)
    try:
        classes = np.unique(y_true)
        if len(classes) == 2:
            # binary: use probability of class 1
            if y_probs.shape[1] == 1:
                return float('nan')
            score = roc_auc_score(y_true, y_probs[:, 1])
        else:
            # multiclass: one-vs-rest
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=classes)
            score = roc_auc_score(y_bin, y_probs, average='macro', multi_class='ovr')
        return float(score)
    except Exception:
        return float('nan')


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    files, labels, class_map = gather_files(args.input_dir)
    if len(files) == 0:
        print('No audio files found under', args.input_dir)
        return
    X_train, X_val, y_train, y_val = train_test_split(files, labels, test_size=args.val_frac, stratify=labels, random_state=args.seed)
    if args.use_bigvgan_features:
        # Precompute BigVGAN conv_pre pooled features
        try:
            import bigvgan
            from meldataset import get_mel_spectrogram
        except Exception as e:
            print('Failed to import BigVGAN modules, falling back to mel pooling:', e)
            args.use_bigvgan_features = False

    if args.use_bigvgan_features:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available() else 'cpu')
        device = torch.device(device)
        model = None
        try:
            model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False).to(device).eval()
            model.remove_weight_norm()
            print('Loaded BigVGAN model for conv_pre features; device=', device)
        except Exception as e:
            print('Could not load BigVGAN model, falling back to mel pooling:', e)
            args.use_bigvgan_features = False

    if args.use_bigvgan_features:
        def compute_conv_pre(path):
            y, _ = librosa.load(path, sr=args.sr, mono=True)
            y_t = torch.from_numpy(y).float().unsqueeze(0)
            mel = get_mel_spectrogram(y_t, model.h).to(device)
            with torch.inference_mode():
                x = model.conv_pre(mel)
            # x shape: [B, C, T]; pool over time to args.bigvgan_dim
            x = x.squeeze(0).detach().cpu().numpy()  # (C, T)
            # if C != bigvgan_dim, do a linear projection via mean pooling then pad/trim
            vec = x.mean(axis=1)
            if vec.shape[0] != args.bigvgan_dim:
                # simple resize: if larger, trim; if smaller, pad with zeros
                v = np.zeros((args.bigvgan_dim,), dtype=np.float32)
                n = min(len(v), vec.shape[0])
                v[:n] = vec[:n]
                vec = v
            return vec.astype(np.float32)

        # compute features for datasets
        def build_feature_dataset(file_list, label_list):
            feats = []
            labs = []
            for p, l in zip(file_list, label_list):
                try:
                    feats.append(compute_conv_pre(p))
                    labs.append(l)
                except Exception as e:
                    print('Failed to compute conv_pre for', p, e)
            if len(feats) == 0:
                raise RuntimeError('No features computed')
            X = np.stack(feats, axis=0)
            y = np.array(labs, dtype=np.int64)
            t = torch.from_numpy(X)
            ty = torch.from_numpy(y)
            ds = data_utils.TensorDataset(t, ty)
            return ds

        train_ds = build_feature_dataset(X_train, y_train)
        val_ds = build_feature_dataset(X_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    else:
        train_ds = AudioFolderDataset(X_train, y_train, sr=args.sr, n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
        val_ds = AudioFolderDataset(X_val, y_val, sr=args.sr, n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available() else 'cpu')
    device = torch.device(device)

    model = SimpleMLP(in_dim=args.n_mels, hidden=256, n_classes=len(class_map)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.compare_conv_mel:
        best_val = float('inf')
        best_epoch = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss, train_acc = train_loop(model, opt, loss_fn, train_loader, device)
            val_loss, val_acc, val_labels, val_preds, val_probs = eval_loop(model, loss_fn, val_loader, device)
            # compute validation AUC
            val_auc = compute_auc(np.array(val_labels), val_probs)
            t1 = time.time()
            history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'val_auc': float(val_auc), 'time_s': t1-t0})
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} val_auc={val_auc:.4f} time={t1-t0:.1f}s")
            # save best
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                torch.save({'model_state_dict': model.state_dict(), 'class_map': class_map}, str(out_dir / 'best_model.pth'))
                print('Saved best model to', out_dir / 'best_model.pth')
            # early stopping
            if epoch - best_epoch >= args.patience:
                print('Early stopping triggered')
                break
    else:
        # Compare conv_pre vs mel pooled features: train two small MLPs on same splits
        # Build datasets/loaders for both modalities. We already built train_loader and val_loader
        # in appropriate mode; for compare mode, build both explicitly
        # build mel loaders
        mel_train_ds = AudioFolderDataset(X_train, y_train, sr=args.sr, n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
        mel_val_ds = AudioFolderDataset(X_val, y_val, sr=args.sr, n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax)
        mel_train_loader = DataLoader(mel_train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch, num_workers=0)
        mel_val_loader = DataLoader(mel_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch, num_workers=0)

        # build conv_pre loaders (precompute)
        try:
            import bigvgan
            from meldataset import get_mel_spectrogram
            device_bg = 'cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', False) and torch.backends.mps.is_available() else 'cpu')
            device_bg = torch.device(device_bg)
            bg_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_24khz_100band_256x', use_cuda_kernel=False).to(device_bg).eval()
            bg_model.remove_weight_norm()
        except Exception as e:
            print('Failed to load BigVGAN for compare mode:', e)
            return

        def conv_features_for_list(file_list):
            feats = []
            labs = []
            for p, l in zip(file_list, y_train if file_list is X_train else y_val):
                try:
                    y, _ = librosa.load(p, sr=args.sr, mono=True)
                    y_t = torch.from_numpy(y).float().unsqueeze(0)
                    mel = get_mel_spectrogram(y_t, bg_model.h).to(device_bg)
                    with torch.inference_mode():
                        x = bg_model.conv_pre(mel)
                    x = x.squeeze(0).detach().cpu().numpy()
                    vec = x.mean(axis=1)
                    # pad/trim
                    if vec.shape[0] != args.bigvgan_dim:
                        v = np.zeros((args.bigvgan_dim,), dtype=np.float32)
                        n = min(len(v), vec.shape[0])
                        v[:n] = vec[:n]
                        vec = v
                    feats.append(vec.astype(np.float32))
                    labs.append(l)
                except Exception as e:
                    print('skip conv feature for', p, e)
            return np.stack(feats, axis=0), np.array(labs, dtype=np.int64)

        conv_train_X, conv_train_y = conv_features_for_list(X_train)
        conv_val_X, conv_val_y = conv_features_for_list(X_val)

        conv_train_ds = data_utils.TensorDataset(torch.from_numpy(conv_train_X), torch.from_numpy(conv_train_y))
        conv_val_ds = data_utils.TensorDataset(torch.from_numpy(conv_val_X), torch.from_numpy(conv_val_y))
        conv_train_loader = DataLoader(conv_train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        conv_val_loader = DataLoader(conv_val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # models
        conv_model = SimpleMLP(in_dim=args.bigvgan_dim, hidden=256, n_classes=len(class_map)).to(device)
        mel_model = SimpleMLP(in_dim=args.n_mels, hidden=256, n_classes=len(class_map)).to(device)
        conv_opt = torch.optim.Adam(conv_model.parameters(), lr=args.lr)
        mel_opt = torch.optim.Adam(mel_model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        rows = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            conv_train_loss, conv_train_acc = train_loop(conv_model, conv_opt, loss_fn, conv_train_loader, device)
            mel_train_loss, mel_train_acc = train_loop(mel_model, mel_opt, loss_fn, mel_train_loader, device)

            conv_val_loss, conv_val_acc, conv_val_labels, conv_val_preds, conv_val_probs = eval_loop(conv_model, loss_fn, conv_val_loader, device)
            mel_val_loss, mel_val_acc, mel_val_labels, mel_val_preds, mel_val_probs = eval_loop(mel_model, loss_fn, mel_val_loader, device)

            conv_val_auc = compute_auc(conv_val_labels, conv_val_probs)
            mel_val_auc = compute_auc(mel_val_labels, mel_val_probs)

            # include both existing '..._auc' and explicit '..._aucroc' fields
            rows.append({'epoch': epoch,
                         'conv_train_acc': float(conv_train_acc),
                         'conv_val_acc': float(conv_val_acc),
                         'conv_val_auc': float(conv_val_auc),
                         'conv_val_aucroc': float(conv_val_auc),
                         'mel_train_acc': float(mel_train_acc),
                         'mel_val_acc': float(mel_val_acc),
                         'mel_val_auc': float(mel_val_auc),
                         'mel_val_aucroc': float(mel_val_auc)})

            print(f"Epoch {epoch}: conv_train_acc={conv_train_acc:.3f} conv_val_acc={conv_val_acc:.3f} mel_train_acc={mel_train_acc:.3f} mel_val_acc={mel_val_acc:.3f} time={time.time()-t0:.1f}s")

        # write combined CSV
        import csv
        combined_csv = out_dir / 'combined_metrics.csv'
        keys = ['epoch','conv_train_acc','conv_val_acc','conv_val_auc','conv_val_aucroc','mel_train_acc','mel_val_acc','mel_val_auc','mel_val_aucroc']
        with open(combined_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

        # also save models
        conv_path = out_dir / 'best_model_conv.pth'
        mel_path = out_dir / 'best_model_mel.pth'
        torch.save({'model_state_dict': conv_model.state_dict(), 'class_map': class_map}, str(conv_path))
        torch.save({'model_state_dict': mel_model.state_dict(), 'class_map': class_map}, str(mel_path))
        print('Compare finished. Artifacts written to:')
        print('  combined_csv:', combined_csv)
        print('  conv_model:', conv_path)
        print('  mel_model :', mel_path)
    return
    # final metrics and artifacts
    import csv
    metrics_csv = out_dir / 'metrics.csv'
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # confusion matrix on val
    if 'val_labels' in locals():
        cm = confusion_matrix(val_labels, val_preds)
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(range(len(class_map)), [class_map[i] for i in range(len(class_map))], rotation=45)
        plt.yticks(range(len(class_map)), [class_map[i] for i in range(len(class_map))])
        plt.tight_layout()
        plt.savefig(out_dir / 'confusion_matrix.png', dpi=150)

    # save class_map
    with open(out_dir / 'class_map.json', 'w') as f:
        json.dump(class_map, f, indent=2)

    print('Training finished. Artifacts written to:')
    print('  metrics_csv:', metrics_csv)
    print('  best_model :', out_dir / 'best_model.pth')
    print('  confusion  :', out_dir / 'confusion_matrix.png')
    print('  class_map  :', out_dir / 'class_map.json')


if __name__ == '__main__':
    main()
