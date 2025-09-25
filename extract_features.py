#!/usr/bin/env python3
"""Extract features from BigVGAN and produce a 2D biplot colored by folder (class).

Usage:
    python demo/extract_features.py --input_dir /path/to/wav_root --out_dir ./demo_out

The script treats each immediate subfolder under --input_dir as a class and processes
all .wav files recursively. It pools model.conv_pre activations (mean over time)
to make a fixed-size feature vector per file, runs PCA -> 2D, and saves a scatter PNG
and a .npz with raw features and labels.
"""
import argparse
import os
import glob
from collections import defaultdict

import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import bigvgan
from meldataset import get_mel_spectrogram

# Ensure repo root is on sys.path so imports like `from meldataset import ...` work
import sys
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def find_wavs(input_dir):
    # return list of (filepath, class_label) where class_label is the immediate parent dir name
    files = []
    for path in glob.glob(os.path.join(input_dir, "**", "*.wav"), recursive=True):
        parent = os.path.basename(os.path.dirname(path))
        files.append((path, parent))
    return sorted(files)


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # follow BigVGAN dataset normalization
    y = librosa.util.normalize(y) * 0.95
    return y


def extract_feature_for_file(model, device, wav_path, h_override=None):
    # load audio, compute mel, run conv_pre and pool over time
    sr = model.h.sampling_rate
    y = load_audio(wav_path, sr)
    y_t = torch.from_numpy(y).float().unsqueeze(0).to(device)  # [1, T]
    with torch.inference_mode():
        h_for_mel = h_override if h_override is not None else model.h
        mel = get_mel_spectrogram(y_t, h_for_mel).to(device)  # [1, C, T]
        # conv_pre maps from num_mels -> channels
        x = model.conv_pre(mel)  # [1, ch, T]
        feat_mean = x.mean(dim=2).squeeze(0).cpu().numpy()
        feat_std = x.std(dim=2).squeeze(0).cpu().numpy()
        feat_conv = np.concatenate([feat_mean, feat_std], axis=0)
        # mel-based features: mean+std across time of the mel spectrogram
        mel_mean = mel.mean(dim=2).squeeze(0).cpu().numpy()
        mel_std = mel.std(dim=2).squeeze(0).cpu().numpy()
        feat_mel = np.concatenate([mel_mean, mel_std], axis=0)
    # return both conv_pre-derived features and mel-derived features
    return feat_conv, feat_mel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Root folder containing class subfolders with wavs")
    parser.add_argument("--out_dir", default="./demo_out")
    parser.add_argument("--fmin", type=int, default=None, help="Minimum frequency (Hz) for mel extraction")
    parser.add_argument("--fmax", type=int, default=None, help="Maximum frequency (Hz) for mel extraction")
    parser.add_argument("--model_id", default="nvidia/bigvgan_v2_24khz_100band_256x")
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP instead of PCA for 2D")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # device selection (prefer mps on mac, otherwise cpu)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...", args.model_id)
    model = bigvgan.BigVGAN.from_pretrained(args.model_id, use_cuda_kernel=False).eval().to(device)
    try:
        model.remove_weight_norm()
    except Exception:
        pass

    files = find_wavs(args.input_dir)
    if not files:
        print("No wav files found under", args.input_dir)
        return

    print(f"Found {len(files)} wav files. Extracting features on device: {device}")

    feats = []
    mel_feats = []
    labels = []
    paths = []

    for i, (p, label) in enumerate(files):
        try:
            # allow overriding mel fmin/fmax for feature extraction
            if args.fmin is not None or args.fmax is not None:
                # model.h is an AttrDict; ensure the override is also an AttrDict so attributes work
                from env import AttrDict

                h_local = AttrDict(dict(model.h))
                if args.fmin is not None:
                    h_local.fmin = args.fmin
                if args.fmax is not None:
                    h_local.fmax = args.fmax
                feat_conv, feat_mel = extract_feature_for_file(model, device, p, h_override=h_local)
            else:
                feat_conv, feat_mel = extract_feature_for_file(model, device, p)
            feats.append(feat_conv)
            mel_feats.append(feat_mel)
            labels.append(label)
            paths.append(p)
            if (i + 1) % 20 == 0:
                print(f"Processed {i+1}/{len(files)}")
        except Exception as e:
            print(f"Failed to process {p}: {e}")

    feats = np.stack(feats, axis=0)
    mel_feats = np.stack(mel_feats, axis=0)
    labels = np.array(labels)

    def reduce_and_plot(features, name_prefix):
        # reduce
        if args.use_umap:
            try:
                import umap

                reducer = umap.UMAP(n_components=2)
                proj_local = reducer.fit_transform(features)
            except Exception as e:
                print("UMAP not available or failed, falling back to PCA:", e)
                pca_local = PCA(n_components=2)
                proj_local = pca_local.fit_transform(features)
        else:
            pca_local = PCA(n_components=2)
            proj_local = pca_local.fit_transform(features)

        # plot scatter colored by label
        unique_labels = sorted(list(set(labels)))
        color_map = {lab: i for i, lab in enumerate(unique_labels)}
        colors = [color_map[l] for l in labels]

        plt.figure(figsize=(10, 8))
        plt.scatter(proj_local[:, 0], proj_local[:, 1], c=colors, cmap="tab10", alpha=0.8)
        # legend
        handles = []
        for lab in unique_labels:
            handles.append(plt.Line2D([], [], marker="o", linestyle="", color=plt.cm.tab10(color_map[lab] % 10), label=lab))
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"BigVGAN {name_prefix} features (pooled) - 2D")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        out_png_local = os.path.join(args.out_dir, f"features_biplot_{name_prefix}.png")
        plt.tight_layout()
        plt.savefig(out_png_local)
        print("Saved biplot:", out_png_local)
        return proj_local

    proj_conv = reduce_and_plot(feats, "conv_pre")
    proj_mel = reduce_and_plot(mel_feats, "mel")

    # save features + metadata
    out_npz = os.path.join(args.out_dir, "features.npz")
    np.savez(
        out_npz,
        feats_conv=feats,
        feats_mel=mel_feats,
        labels=labels,
        paths=np.array(paths),
        proj_conv=proj_conv,
        proj_mel=proj_mel,
    )
    print("Saved features:", out_npz)


if __name__ == "__main__":
    main()
