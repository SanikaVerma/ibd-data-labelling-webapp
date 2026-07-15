"""
Generate a small synthetic dataset in the EXACT on-disk numpy structure that
TransEHR2's preprocessing.py produces (save_dataset / extract_mimic), plus a
parallel set of XAI importance-score arrays in the proposed format.

Purpose: develop and test the importance-score -> timeline mapping against the
real array layout before real model outputs exist. Nothing here is real patient
data — only shapes and file names mirror the real pipeline.

Run with:  python generate_fake_arrays.py

Output layout (default: fake_transehr_output/):
    train/                       input arrays (mirror the model's inputs)
    train_xai/                   importance scores (proposed parallel format)
    train_ids.pkl                row index -> patient episode id
    summary_statistics_train.npz means/p5/p95 for reverse-standardization
    variable_properties.yaml     feature schema (type/size/category_map/description)
    fake_vocab.json              token id -> word (stand-in for the real tokenizer)

Array shapes (n_ep episodes, max_ts timesteps):
    val_numeric_indicators      (n_ep, max_ts, n_numeric)          float32  the "z" vector
    val_numeric_values_{f}      (n_ep, max_ts, feat_dim)           float32  standardized
    val_categorical_indicators  (n_ep, max_ts, n_cat)              float32
    val_categorical_values_{f}  (n_ep, max_ts, n_classes)          int64    one-hot
    val_ordinal_indicators      (n_ep, max_ts, n_ord)              float32
    val_ordinal_values_{f}      (n_ep, max_ts, n_levels)           int64    one-hot
    val_text_indicators         (n_ep, max_ts, n_text)             float32
    val_text_offsets_{f}        (n_ep+1,)                          int64    CSR row pointers
    val_text_values_{f}         (n_nonempty, token_len)            int64    token ids
    val_text_masks_{f}          (n_nonempty, token_len)            float32
    val_text_timesteps_{f}      (n_nonempty,)                      int32
    val_times / val_masks       (n_ep, max_ts)                     float32
    event_indicators            (n_ep, max_ts, n_event)            float32
    event_times / event_masks   (n_ep, max_ts)                     float32
    static_data                 (n_ep, static_total_dim)           float32
    mortality / length_of_stay  (n_ep,)                            float32
    phenotype                   (n_ep, phenotype_dim)              float32

XAI score arrays (train_xai/) mirror the shapes of the value arrays they explain,
with one score per input element; text is one score per token.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Feature schema — a small synthetic stand-in for variable_properties.yaml.
# Feature ordering here defines the column order in the arrays, exactly as
# reader.valued_feats does in the real pipeline.
# ---------------------------------------------------------------------------
VALUED_FEATS = [
    # name,          type,          size, category_map (for cat/ordinal), description
    ("CRP",          "numeric",      1,   None,                              "C-reactive protein"),
    ("Hemoglobin",   "numeric",      1,   None,                              "Hemoglobin"),
    ("Creatinine",   "numeric",      1,   None,                              "Creatinine"),
    ("Institution",  "categorical",  4,   {0: "University Hospital", 1: "Regional Centre",
                                            2: "Community Clinic", 3: "Emergency Dept"},
                                          "Institution"),
    ("AdmitCategory", "categorical", 3,   {0: "Elective", 1: "Urgent", 2: "Emergency"},
                                          "Admission category"),
    ("TriageCode",   "ordinal",      5,   {0: 5, 1: 4, 2: 3, 3: 2, 4: 1},
                                          "Triage priority (5 least, 1 most pressing)"),
    ("ClinicalNote", "text",         1,   None,                              "Clinical note"),
]

EVENT_FEATS = [
    ("Imaging",   "Imaging performed"),
    ("Procedure", "Procedure performed"),
]

STATIC_FEATS = [
    ("Sex",       "categorical", 2, {0: "F", 1: "M"}, "Sex"),
    ("BirthYear", "numeric",     1, None,             "Birth year"),
]

# Plausible raw value ranges for numeric features (before standardization).
NUMERIC_RAW_RANGES = {
    "CRP":        (1.0, 60.0),
    "Hemoglobin": (80.0, 170.0),
    "Creatinine": (40.0, 120.0),
    "BirthYear":  (1940.0, 2005.0),
}

# Tiny vocabulary standing in for the real Llama tokenizer. Token id 0 = PAD.
VOCAB = [
    "[PAD]", "patient", "presents", "with", "abdominal", "pain", "and",
    "diarrhea", "crohn", "flare", "stable", "no", "acute", "distress",
    "elevated", "inflammatory", "markers", "colitis", "improved", "on", "biologic",
]
TOKEN_LEN = 8
PAD_ID = 0


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.int64)
    if 0 <= idx < size:
        v[idx] = 1
    return v


def generate(out_dir: Path, n_ep: int = 3, max_ts: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)

    numeric = [f for f in VALUED_FEATS if f[1] == "numeric"]
    categorical = [f for f in VALUED_FEATS if f[1] == "categorical"]
    ordinal = [f for f in VALUED_FEATS if f[1] == "ordinal"]
    text = [f for f in VALUED_FEATS if f[1] == "text"]

    n_num, n_cat, n_ord, n_txt = len(numeric), len(categorical), len(ordinal), len(text)
    n_evt = len(EVENT_FEATS)

    # Per-episode valid length (variable, to test length handling).
    ep_lens = rng.integers(low=max(3, max_ts // 2), high=max_ts + 1, size=n_ep)

    # ---- Dense value-associated arrays ----
    val_times = np.zeros((n_ep, max_ts), dtype=np.float32)
    val_masks = np.zeros((n_ep, max_ts), dtype=np.float32)
    num_ind = np.zeros((n_ep, max_ts, n_num), dtype=np.float32)
    cat_ind = np.zeros((n_ep, max_ts, n_cat), dtype=np.float32)
    ord_ind = np.zeros((n_ep, max_ts, n_ord), dtype=np.float32)
    txt_ind = np.zeros((n_ep, max_ts, n_txt), dtype=np.float32)

    num_vals = [np.zeros((n_ep, max_ts, 1), dtype=np.float32) for _ in numeric]
    cat_vals = [np.zeros((n_ep, max_ts, f[2]), dtype=np.int64) for f in categorical]
    ord_vals = [np.zeros((n_ep, max_ts, f[2]), dtype=np.int64) for f in ordinal]

    # Store RAW numeric values first; standardize afterward so we can save the
    # exact stats needed to reverse it (round-trip test for the mapping).
    num_raw = [np.zeros((n_ep, max_ts, 1), dtype=np.float32) for _ in numeric]

    # ---- Event arrays ----
    evt_ind = np.zeros((n_ep, max_ts, n_evt), dtype=np.float32)
    evt_times = np.zeros((n_ep, max_ts), dtype=np.float32)
    evt_masks = np.zeros((n_ep, max_ts), dtype=np.float32)

    # ---- Sparse text (CSR-style) collected in episode order ----
    text_values_rows, text_masks_rows, text_timesteps, text_counts = [], [], [], []

    for i in range(n_ep):
        ep_len = int(ep_lens[i])
        # Hours since admission, increasing.
        hours = np.cumsum(rng.integers(1, 12, size=ep_len)).astype(np.float32)
        val_times[i, :ep_len] = hours
        val_masks[i, :ep_len] = 1.0
        evt_times[i, :ep_len] = hours
        evt_masks[i, :ep_len] = 1.0

        n_text_this_ep = 0
        for t in range(ep_len):
            # Numeric: each feature observed with ~50% chance.
            for f, (name, _, _, _, _) in enumerate(numeric):
                if rng.random() < 0.5:
                    num_ind[i, t, f] = 1.0
                    lo, hi = NUMERIC_RAW_RANGES[name]
                    num_raw[f][i, t, 0] = rng.uniform(lo, hi)

            # Categorical: observed with ~40% chance.
            for f, feat in enumerate(categorical):
                if rng.random() < 0.4:
                    cat_ind[i, t, f] = 1.0
                    idx = int(rng.integers(0, feat[2]))
                    cat_vals[f][i, t, :] = _one_hot(idx, feat[2])

            # Ordinal: observed with ~30% chance.
            for f, feat in enumerate(ordinal):
                if rng.random() < 0.3:
                    ord_ind[i, t, f] = 1.0
                    idx = int(rng.integers(0, feat[2]))
                    ord_vals[f][i, t, :] = _one_hot(idx, feat[2])

            # Text: observed with ~20% chance (sparse).
            for f in range(n_txt):
                if rng.random() < 0.2:
                    txt_ind[i, t, f] = 1.0
                    n_real = int(rng.integers(3, TOKEN_LEN + 1))
                    ids = rng.integers(1, len(VOCAB), size=n_real)  # skip PAD id 0
                    row = np.full(TOKEN_LEN, PAD_ID, dtype=np.int64)
                    row[:n_real] = ids
                    mask = np.zeros(TOKEN_LEN, dtype=np.float32)
                    mask[:n_real] = 1.0
                    text_values_rows.append(row)
                    text_masks_rows.append(mask)
                    text_timesteps.append(t)
                    n_text_this_ep += 1

            # Events: observed with ~25% chance.
            for f in range(n_evt):
                if rng.random() < 0.25:
                    evt_ind[i, t, f] = 1.0

        text_counts.append(n_text_this_ep)

    # ---- Static arrays ----
    # static_total_dim = sum of static feature sizes; categorical stores a single
    # index at its first slot (mirrors process_static_data's offset advance).
    static_dims = [f[2] for f in STATIC_FEATS]
    static_total = int(sum(static_dims))
    static_data = np.zeros((n_ep, static_total), dtype=np.float32)
    for i in range(n_ep):
        offset = 0
        for feat in STATIC_FEATS:
            name, ftype, size, cmap, _ = feat
            if ftype == "numeric":
                lo, hi = NUMERIC_RAW_RANGES.get(name, (0.0, 1.0))
                static_data[i, offset] = rng.uniform(lo, hi)
            elif ftype == "categorical":
                static_data[i, offset] = float(rng.integers(0, size))
            offset += size

    # ---- Standardize numeric features (mirror standardize_feats) ----
    means = np.zeros(n_num, dtype=np.float32)
    p5 = np.zeros(n_num, dtype=np.float32)
    p95 = np.zeros(n_num, dtype=np.float32)
    for f in range(n_num):
        mask = num_ind[:, :, f] == 1.0
        if mask.any():
            observed = num_raw[f][mask]                       # (n_observed, 1)
            means[f] = observed.mean()
            norms = np.linalg.norm(observed, ord=2, axis=-1)  # abs() for feat_dim=1
            p5[f] = np.percentile(norms, 5)
            p95[f] = np.percentile(norms, 95)
        std = num_raw[f].copy()
        if p95[f] != p5[f]:
            std = (std - means[f]) / (p95[f] - p5[f])
        else:
            std[:] = 0.0
        std[num_ind[:, :, f] == 0.0] = 0.0  # keep unobserved at zero
        num_vals[f][:] = std

    # ---- Targets ----
    mortality = rng.integers(0, 2, size=n_ep).astype(np.float32)
    length_of_stay = rng.uniform(24, 240, size=n_ep).astype(np.float32)
    phenotype_dim = 2
    phenotype = rng.integers(0, 2, size=(n_ep, phenotype_dim)).astype(np.float32)

    # ---- CSR offsets for text ----
    offsets = np.zeros(n_ep + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(text_counts)
    if text_values_rows:
        text_values = np.stack(text_values_rows, axis=0)
        text_masks = np.stack(text_masks_rows, axis=0)
        text_ts = np.array(text_timesteps, dtype=np.int32)
    else:
        text_values = np.zeros((0, TOKEN_LEN), dtype=np.int64)
        text_masks = np.zeros((0, TOKEN_LEN), dtype=np.float32)
        text_ts = np.zeros(0, dtype=np.int32)

    # -----------------------------------------------------------------
    # XAI score arrays — same shapes as the value arrays they explain.
    # Scores are placed only where the feature was actually recorded
    # (indicator == 1), matching gradient*input being ~0 for absent inputs.
    # -----------------------------------------------------------------
    # Per-feature score arrays (kept per-feature so each can match its own width).
    xai_numeric = []
    for f in range(n_num):
        a = np.zeros((n_ep, max_ts, 1), dtype=np.float32)
        m = num_ind[:, :, f] == 1.0
        a[m] = rng.uniform(-1, 1, size=(int(m.sum()), 1))
        xai_numeric.append(a)

    xai_categorical = []
    for f, feat in enumerate(categorical):
        a = np.zeros((n_ep, max_ts, feat[2]), dtype=np.float32)
        m = cat_ind[:, :, f] == 1.0
        # Score sits on the active one-hot class only (others contribute ~0).
        for i in range(n_ep):
            for t in range(max_ts):
                if m[i, t]:
                    active = int(np.argmax(cat_vals[f][i, t]))
                    a[i, t, active] = rng.uniform(-1, 1)
        xai_categorical.append(a)

    xai_ordinal = []
    for f, feat in enumerate(ordinal):
        a = np.zeros((n_ep, max_ts, feat[2]), dtype=np.float32)
        m = ord_ind[:, :, f] == 1.0
        for i in range(n_ep):
            for t in range(max_ts):
                if m[i, t]:
                    active = int(np.argmax(ord_vals[f][i, t]))
                    a[i, t, active] = rng.uniform(-1, 1)
        xai_ordinal.append(a)

    xai_event = np.zeros((n_ep, max_ts, n_evt), dtype=np.float32)
    m = evt_ind == 1.0
    xai_event[m] = rng.uniform(-1, 1, size=int(m.sum()))

    xai_static = rng.uniform(-1, 1, size=(n_ep, static_total)).astype(np.float32)

    # Text: one score per token, only for real (non-PAD) tokens.
    xai_text = np.zeros_like(text_values, dtype=np.float32)
    xai_text[text_masks == 1.0] = rng.uniform(-1, 1, size=int((text_masks == 1.0).sum()))

    # -----------------------------------------------------------------
    # Write everything to disk in the real layout.
    # -----------------------------------------------------------------
    train_dir = out_dir / "train"
    xai_dir = out_dir / "train_xai"
    train_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    def save(d, name, arr):
        np.save(d / f"{name}.npy", arr)

    # Input arrays (mirror save_dataset naming).
    save(train_dir, "val_numeric_indicators", num_ind)
    save(train_dir, "val_categorical_indicators", cat_ind)
    save(train_dir, "val_ordinal_indicators", ord_ind)
    save(train_dir, "val_text_indicators", txt_ind)
    save(train_dir, "val_times", val_times)
    save(train_dir, "val_masks", val_masks)
    save(train_dir, "event_indicators", evt_ind)
    save(train_dir, "event_times", evt_times)
    save(train_dir, "event_masks", evt_masks)
    save(train_dir, "static_data", static_data)
    save(train_dir, "mortality", mortality)
    save(train_dir, "length_of_stay", length_of_stay)
    save(train_dir, "phenotype", phenotype)
    for f in range(n_num):
        save(train_dir, f"val_numeric_values_{f}", num_vals[f])
    for f in range(n_cat):
        save(train_dir, f"val_categorical_values_{f}", cat_vals[f])
    for f in range(n_ord):
        save(train_dir, f"val_ordinal_values_{f}", ord_vals[f])
    for f in range(n_txt):
        save(train_dir, f"val_text_offsets_{f}", offsets)
        save(train_dir, f"val_text_values_{f}", text_values)
        save(train_dir, f"val_text_masks_{f}", text_masks)
        save(train_dir, f"val_text_timesteps_{f}", text_ts)

    metadata = {
        "max_ts_len": max_ts,
        "text_token_len": [TOKEN_LEN] * n_txt,
        "text_embed_dim": 0,
        "n_numeric_feats": n_num,
        "n_categorical_feats": n_cat,
        "n_ordinal_feats": n_ord,
        "n_text_feats": n_txt,
    }
    with open(train_dir / "metadata.pkl", "wb") as fh:
        pickle.dump(metadata, fh)

    # XAI score arrays (proposed parallel format).
    for f in range(n_num):
        save(xai_dir, f"xai_numeric_{f}", xai_numeric[f])
    for f in range(n_cat):
        save(xai_dir, f"xai_categorical_{f}", xai_categorical[f])
    for f in range(n_ord):
        save(xai_dir, f"xai_ordinal_{f}", xai_ordinal[f])
    save(xai_dir, "xai_event", xai_event)
    save(xai_dir, "xai_static", xai_static)
    for f in range(n_txt):
        save(xai_dir, f"xai_text_{f}", xai_text)

    # Episode IDs (row -> patient episode id). Real IDs may differ from webapp
    # patient IDs — see open question to supervisor.
    episode_ids = [9, 20, 2][:n_ep] + list(range(1000, 1000 + max(0, n_ep - 3)))
    with open(out_dir / "train_ids.pkl", "wb") as fh:
        pickle.dump(episode_ids, fh)

    # Standardization stats for reverse-mapping numeric values.
    np.savez(out_dir / "summary_statistics_train.npz", means=means, p5=p5, p95=p95)

    # Feature schema, in the real variable_properties.yaml shape.
    var_props = {}
    for name, ftype, size, cmap, desc in VALUED_FEATS + [
        (n, t, s, c, d) for (n, t, s, c, d) in STATIC_FEATS
    ]:
        entry = {"type": ftype, "size": size, "description": desc}
        if cmap is not None:
            entry["category_map"] = cmap
        var_props[name] = entry
    for name, desc in EVENT_FEATS:
        var_props[name] = {"type": "event", "size": 1, "description": desc}
    with open(out_dir / "variable_properties.yaml", "w") as fh:
        yaml.safe_dump(var_props, fh, sort_keys=False)

    # Stand-in vocabulary (token id -> word) for detokenizing text without HF.
    with open(out_dir / "fake_vocab.json", "w") as fh:
        json.dump({i: w for i, w in enumerate(VOCAB)}, fh, indent=2)

    # Feature layout — which features are value-associated vs static vs event,
    # in array-column order. In the real pipeline this comes from the reader
    # config (valued_feats / event_feats / static_feats), NOT from
    # variable_properties.yaml (types alone don't distinguish the roles).
    feature_layout = {
        "numeric_feats": [f[0] for f in numeric],
        "categorical_feats": [f[0] for f in categorical],
        "ordinal_feats": [f[0] for f in ordinal],
        "text_feats": [f[0] for f in text],
        "event_feats": [f[0] for f in EVENT_FEATS],
        "static_feats": [f[0] for f in STATIC_FEATS],
    }
    with open(out_dir / "feature_layout.json", "w") as fh:
        json.dump(feature_layout, fh, indent=2)

    print(f"Wrote fake dataset for {n_ep} episodes (max_ts={max_ts}) to {out_dir}/")
    print(f"  episodes:        {episode_ids}")
    print(f"  numeric feats:   {[f[0] for f in numeric]}")
    print(f"  categorical:     {[f[0] for f in categorical]}")
    print(f"  ordinal:         {[f[0] for f in ordinal]}")
    print(f"  text feats:      {[f[0] for f in text]}  (non-empty entries: {len(text_values)})")
    print(f"  event feats:     {[f[0] for f in EVENT_FEATS]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="fake_transehr_output", help="Output directory")
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--max_ts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(Path(args.out), n_ep=args.n_episodes, max_ts=args.max_ts, seed=args.seed)


if __name__ == "__main__":
    main()
