"""Reproducible semi-supervised and cross-fitting splits."""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from config import LABELLED_RATIO, PPI_TRAIN_RATIO, CROSSPPI_K


def make_label_unlabel_split(n_total: int, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    n_labelled = max(2, int(round(LABELLED_RATIO * n_total)))
    return np.sort(perm[:n_labelled]), np.sort(perm[n_labelled:])


def make_ppi_train_inf_split(labelled_idx, y, binary: bool, seed: int):
    labelled_idx = np.asarray(labelled_idx)
    stratify = y[labelled_idx] if binary else None
    train_idx, inf_idx = train_test_split(
        labelled_idx,
        train_size=PPI_TRAIN_RATIO,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.sort(train_idx), np.sort(inf_idx)


def make_crossppi_folds(labelled_idx, y, binary: bool, seed: int, K: int = CROSSPPI_K):
    labelled_idx = np.asarray(labelled_idx)
    if binary:
        splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
        iterator = splitter.split(labelled_idx, y[labelled_idx])
    else:
        splitter = KFold(n_splits=K, shuffle=True, random_state=seed)
        iterator = splitter.split(labelled_idx)
    folds = []
    seen = []
    for train_pos, hold_pos in iterator:
        train_idx = labelled_idx[train_pos]
        hold_idx = labelled_idx[hold_pos]
        folds.append((np.asarray(train_idx), np.asarray(hold_idx)))
        seen.extend(hold_idx.tolist())
    if sorted(seen) != sorted(labelled_idx.tolist()):
        raise RuntimeError("Cross-PPI folds do not cover the labelled set exactly once.")
    return folds
