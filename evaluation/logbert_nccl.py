#!/usr/bin/env python3
"""
LogBERT entry point for NCCL log anomaly detection.

This script mirrors the role of HDFS/logbert.py but is adapted for the
NCCL log dataset.  It exposes three subcommands:

    vocab   – build WordVocab from the training sequence file
    train   – train LogBERT on normal (phase1) sequences
    predict – evaluate LogBERT on test_normal / test_abnormal and print metrics

The options dict defined here is also imported by run_evaluation.py so
that both scripts share exactly the same hyperparameters.

Typical usage (run from the evaluation/ directory):
    python data_process.py          # preprocess logs first
    python logbert_nccl.py vocab
    python logbert_nccl.py train
    python logbert_nccl.py predict
"""

import sys
import os
import argparse
from pathlib import Path

import torch

# Make bert_pytorch and logparser importable regardless of CWD.
EVAL_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(EVAL_DIR / "logbert"))

from bert_pytorch.dataset import WordVocab          # noqa: E402
from bert_pytorch import Predictor, Trainer         # noqa: E402
from bert_pytorch.dataset.utils import seed_everything  # noqa: E402

# ── Output paths ──────────────────────────────────────────────────────────────

OUTPUT_DIR = str((EVAL_DIR / "../output/nccl").resolve()) + "/"
MODEL_DIR = OUTPUT_DIR + "bert/"

# ── Hyperparameters ───────────────────────────────────────────────────────────

options: dict = {
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Paths
    "output_dir":  OUTPUT_DIR,
    "model_dir":   MODEL_DIR,
    "model_path":  MODEL_DIR + "best_bert.pth",
    "train_vocab": OUTPUT_DIR + "train",
    "vocab_path":  OUTPUT_DIR + "vocab.pkl",
    "scale_path":  MODEL_DIR + "scale.pkl",

    # Windowing / sequence length
    # adaptive_window=True means the whole log session becomes one window,
    # which is appropriate here (each log file is one training/test session).
    "window_size":       128,
    "adaptive_window":   True,
    "seq_len":           512,
    "max_len":           512,
    # Sequences shorter than min_len events are discarded.  NCCL log files
    # typically produce 100+ events, so 10 is a safe lower bound.
    "min_len":           0,
    "mask_ratio":        0.65,

    # Data split ratios
    "train_ratio":  1,
    # 0.2 ensures at least 1 validation sample from the 6 normal sequences
    # (floor(6 × 0.2) = 1), avoiding a ZeroDivisionError in the validation loop.
    "valid_ratio":  0.2,
    "test_ratio":   1,

    # Feature flags
    "is_logkey":   True,   # use masked log-key prediction (core LogBERT task)
    "is_time":     False,  # no timestamp delta features in NCCL logs
    # hypersphere_loss=False: with only 6 training sequences deep-SVDD does
    # not generalise well; pure masked-LM anomaly detection is sufficient.
    "hypersphere_loss":      False,
    "hypersphere_loss_test": False,
    "scale": None,

    # BERT architecture – smaller than the HDFS default because our dataset
    # is tiny; this avoids overfitting and keeps training fast on CPU.
    "hidden":     128,
    "layers":     2,
    "attn_heads": 4,

    # Training schedule
    "epochs":         200,
    "n_epochs_stop":  10,   # early-stop patience (epochs without improvement)
    # batch_size=1: with only 5–6 training sequences and drop_last=True we
    # need batches of 1 so no sample is silently discarded.
    "batch_size":  1,
    "corpus_lines": None,
    "on_memory":    True,
    "num_workers":  0,      # 0 avoids multiprocessing issues on small datasets
    "lr":               1e-3,
    "adam_beta1":       0.9,
    "adam_beta2":       0.999,
    "adam_weight_decay": 0.00,
    "with_cuda":     True,
    "cuda_devices":  None,
    "log_freq":      None,

    # Prediction
    # A sequence is anomalous when (undetected_tokens / masked_tokens) > threshold.
    # num_candidates: how many top predicted tokens count as "correct".
    "num_candidates": 6,
    "gaussian_mean":  0,
    "gaussian_std":   1,
}

seed_everything(seed=1234)

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LogBERT for NCCL log anomaly detection"
    )
    subparsers = parser.add_subparsers(dest="mode")
    subparsers.required = True

    # --- vocab ---
    vocab_parser = subparsers.add_parser(
        "vocab", help="Build WordVocab from the training sequence file"
    )
    vocab_parser.add_argument(
        "-s", "--vocab_size", type=int, default=None,
        help="Maximum vocabulary size (default: unlimited)"
    )
    vocab_parser.add_argument(
        "-e", "--encoding", type=str, default="utf-8"
    )
    vocab_parser.add_argument(
        "-f", "--min_freq", type=int, default=1,
        help="Minimum token frequency to include in vocab"
    )

    # --- train ---
    subparsers.add_parser("train", help="Train LogBERT on normal sequences")

    # --- predict ---
    predict_parser = subparsers.add_parser(
        "predict", help="Run LogBERT prediction and print aggregate metrics"
    )
    predict_parser.add_argument("-m", "--mean", type=float, default=0)
    predict_parser.add_argument("-s", "--std",  type=float, default=1)

    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    if args.mode == "vocab":
        with open(options["train_vocab"], "r", encoding=args.encoding) as f:
            texts = f.readlines()
        vocab = WordVocab(texts, max_size=args.vocab_size, min_freq=args.min_freq)
        print(f"Vocabulary size: {len(vocab)}")
        vocab.save_vocab(options["vocab_path"])

    elif args.mode == "train":
        Trainer(options).train()

    elif args.mode == "predict":
        options["gaussian_mean"] = args.mean
        options["gaussian_std"] = args.std
        Predictor(options).predict()
