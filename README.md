# SSL in Tropical Soundscapes 

This repository contains a small, focused set of scripts and demos for
self-supervised learning (SSL) on tropical soundscape audio. The goal is to
provide a compact, easy-to-run pipeline for pretraining an audio encoder,
extracting embeddings, and visualizing them.

---

## What is included

- `demo/ssl_pretrain.py` — SimCLR-style SSL pretraining on mel spectrograms
- `demo/extract_ssl_embeddings.py` — extract encoder embeddings from a checkpoint
- `demo/plot_features_umap.py` — project and plot embeddings (UMAP or PCA fallback)
- `demo/ssl-pipeline.py` — combined CLI (pretrain / extract / plot) for convenience
- Small helper scripts used during experiments (in `demo/`)

## What we did

- Added a lightweight SSL pretraining script that creates a small encoder and
  projection head, uses simple waveform and mel augmentations, and saves
  checkpoints.
- Added an extractor that loads a checkpoint and computes fixed-length
  embeddings for a directory of audio files, saving a compressed NPZ with
  `feats_ssl`, `labels`, and `paths`.
- Added plotting utilities to visualize embeddings in 2D (UMAP preferred,
  PCA fallback if UMAP is not installed).
- Created a small, combined `ssl-pipeline.py` convenience CLI that runs
  pretrain → extract → plot as separate subcommands for quick demos.

These scripts are intentionally small and dependency-light so you can run
quick experiments on local machines (CPU, MPS, or GPU).

## Quick demo (copy/paste)

1) Install dependencies (example):

```bash
pip install torch librosa soundfile numpy matplotlib scikit-learn
# optional: umap-learn for better visualizations
pip install umap-learn
```

2) Run the pipeline from the repo root (smoke run with small batch):

```bash
python demo/ssl-pipeline.py pretrain --data_root /path/to/audio --out_dir demo/ssl --epochs 1 --batch_size 8
python demo/ssl-pipeline.py extract --checkpoint demo/ssl/encoder_epoch1.pth --data_root /path/to/audio --out_npz demo/ssl/features_ssl.npz --batch_size 8
python demo/ssl-pipeline.py plot --features demo/ssl/features_ssl.npz --out_dir demo/ssl
```

3) Outputs (by default in `demo/ssl/`):

- `encoder_epoch{E}.pth` — saved checkpoints from pretraining
- `features_ssl.npz` — embeddings (feats_ssl), labels, and file paths
- `features_umap_ssl.png` — 2D visualization of embeddings

## Notes