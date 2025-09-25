#!/usr/bin/env python3
import argparse
import os
import numpy as np
import soundfile as sf

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', required=True)
    p.add_argument('--sr', type=int, default=22050)
    p.add_argument('--duration', type=float, default=1.0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for label in ['classA', 'classB']:
        od = os.path.join(args.out_dir, label)
        os.makedirs(od, exist_ok=True)
        for i in range(4):
            t = np.linspace(0, args.duration, int(args.sr * args.duration), endpoint=False)
            freq = 440 + (i * 50) + (0 if label == 'classA' else 300)
            y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            sf.write(os.path.join(od, f"{label}_{i}.wav"), y, args.sr)
    print('Wrote test data to', args.out_dir)

if __name__ == '__main__':
    main()
