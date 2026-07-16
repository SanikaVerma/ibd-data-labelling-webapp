"""
Generate a small synthetic dataset in the EXACT on-disk numpy structure that
TransEHR2's preprocessing.py produces, plus a parallel set of XAI importance-
score arrays.

Purpose: develop and test the importance-score -> timeline mapping against the
real array layout before real model outputs exist. Nothing here is real patient
data — only the shapes, file names and feature ordering mirror the real pipeline.

Run with:  python generate_fake_arrays.py

Structure mirrors (see TransEHR2/data/preprocessing.py):
  - save_dataset()            for the input array file names
  - standardize_feats()       for the numeric standardization we invert later
  - DataProcessor.__init__()  for how features map to array columns
  - LLMTextProcessor          for the text tokenization settings

FEATURE ORDER (important — this is what makes scores traceable):
Array column order is NOT arbitrary and is NOT stored in a separate file. It is
dictated by the dataset config's ordered feature lists, split by the `type`
declared in variable_properties.yaml, preserving order of appearance:

    numeric_feats     = [f for f in VALUED_FEATS if type(f) == 'numeric']
    categorical_feats = [f for f in VALUED_FEATS if type(f) == 'categorical']
    ordinal_feats     = [f for f in VALUED_FEATS if type(f) == 'ordinal']
    text_feats        = TEXT_FEATS      (config order)
    event_feats       = EVENT_FEATS     (config order)
    static_feats      = STATIC_FEATS    (config order)

This is exactly what DataProcessor.__init__ does. We therefore emit a dataset
config in the same shape as TransEHR2/configs/datasets/*.yaml, and the reader
re-derives the ordering from it — no duplicated bookkeeping.

XAI SCORE SHAPES:
One score per feature per timestep (the dot product of the feature vector with
the gradient happens upstream when the score is produced), so score arrays mirror
the INDICATOR arrays, not the per-feature value arrays. Text and drugs are the
exception: those are scored per token / per drug.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

# Matches MAX_TOKEN_LENGTH in TransEHR2/constants.py
MAX_TOKEN_LENGTH = 1024
# Matches TOKENIZER_PAD_TOKEN in TransEHR2/constants.py
TOKENIZER_PAD_TOKEN = '[PAD]'
# Matches LLM_NAME in TransEHR2/constants.py
LLM_NAME = 'meta-llama/Llama-3.1-70B'


# ---------------------------------------------------------------------------
# Feature schema. Order here defines array column order (see module docstring).
# ---------------------------------------------------------------------------
# (name, type, size, category_map, description)
VALUED_FEATS = [
    ("CRP",           "numeric",     1, None, "C-reactive protein"),
    ("Hemoglobin",    "numeric",     1, None, "Hemoglobin"),
    ("Creatinine",    "numeric",     1, None, "Creatinine"),
    ("Institution",   "categorical", 4, {0: "University Hospital", 1: "Regional Centre",
                                         2: "Community Clinic", 3: "Emergency Dept"},
                                        "Institution"),
    ("AdmitCategory", "categorical", 3, {0: "Elective", 1: "Urgent", 2: "Emergency"},
                                        "Admission category"),
    ("TriageCode",    "ordinal",     5, {0: 5, 1: 4, 2: 3, 3: 2, 4: 1},
                                        "Triage priority (5 least, 1 most pressing)"),
]
TEXT_FEATS = [
    ("ClinicalNote", "text", 1, None, "Clinical note"),
]
EVENT_FEATS = [
    ("Imaging",   "Imaging performed"),
    ("Procedure", "Procedure performed"),
]
STATIC_FEATS = [
    ("Sex",       "categorical", 2, {0: "F", 1: "M"}, "Sex"),
    ("BirthYear", "numeric",     1, None,             "Birth year"),
]

NUMERIC_RAW_RANGES = {
    "CRP":        (1.0, 60.0),
    "Hemoglobin": (80.0, 170.0),
    "Creatinine": (40.0, 120.0),
    "BirthYear":  (1940.0, 2005.0),
}

# Stand-in clinical sentences. Content is irrelevant (any text works) — what
# matters is that they go through the real tokenizer, so the round-trip is
# identical to the real pipeline.
SAMPLE_NOTES = [
    "Patient presents with abdominal pain and a suspected Crohn disease flare.",
    "Colonoscopy shows moderate inflammation of the terminal ileum. Started on biologic therapy.",
    "Follow-up visit. Symptoms improved, inflammatory markers trending down. Continue maintenance.",
    "Admitted overnight with severe diarrhea and dehydration. IV fluids given.",
]


def _one_hot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.int64)
    if 0 <= idx < size:
        v[idx] = 1
    return v


def _load_tokenizer():
    """Load the real Llama tokenizer with the same settings as LLMTextProcessor."""
    from transformers import AutoTokenizer
    load_dotenv()
    tk = AutoTokenizer.from_pretrained(LLM_NAME, token=os.getenv('HF_READ_TOKEN'))
    tk.add_special_tokens({'pad_token': TOKENIZER_PAD_TOKEN})
    return tk


def _tokenize(tk, text: str):
    """Tokenize exactly as LLMTextProcessor.process_text does."""
    out = tk(
        text,
        max_length=MAX_TOKEN_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='np',
    )
    return out['input_ids'][0], out['attention_mask'][0].astype(np.float32)


def generate(out_dir: Path, n_ep: int = 3, max_ts: int = 12, seed: int = 42):
    rng = np.random.default_rng(seed)
    tk = _load_tokenizer()

    numeric = [f for f in VALUED_FEATS if f[1] == "numeric"]
    categorical = [f for f in VALUED_FEATS if f[1] == "categorical"]
    ordinal = [f for f in VALUED_FEATS if f[1] == "ordinal"]

    n_num, n_cat, n_ord = len(numeric), len(categorical), len(ordinal)
    n_txt, n_evt, n_static = len(TEXT_FEATS), len(EVENT_FEATS), len(STATIC_FEATS)

    ep_lens = rng.integers(low=max(3, max_ts // 2), high=max_ts + 1, size=n_ep)

    # ---- input arrays ----
    val_times = np.zeros((n_ep, max_ts), dtype=np.float32)
    val_masks = np.zeros((n_ep, max_ts), dtype=np.float32)
    num_ind = np.zeros((n_ep, max_ts, n_num), dtype=np.float32)
    cat_ind = np.zeros((n_ep, max_ts, n_cat), dtype=np.float32)
    ord_ind = np.zeros((n_ep, max_ts, n_ord), dtype=np.float32)
    txt_ind = np.zeros((n_ep, max_ts, n_txt), dtype=np.float32)

    num_raw = [np.zeros((n_ep, max_ts, 1), dtype=np.float32) for _ in numeric]
    num_vals = [np.zeros((n_ep, max_ts, 1), dtype=np.float32) for _ in numeric]
    cat_vals = [np.zeros((n_ep, max_ts, f[2]), dtype=np.int64) for f in categorical]
    ord_vals = [np.zeros((n_ep, max_ts, f[2]), dtype=np.int64) for f in ordinal]

    evt_ind = np.zeros((n_ep, max_ts, n_evt), dtype=np.float32)
    evt_times = np.zeros((n_ep, max_ts), dtype=np.float32)
    evt_masks = np.zeros((n_ep, max_ts), dtype=np.float32)

    text_rows, text_mask_rows, text_ts_list, text_counts = [], [], [], []

    for i in range(n_ep):
        ep_len = int(ep_lens[i])
        hours = np.cumsum(rng.integers(1, 12, size=ep_len)).astype(np.float32)
        val_times[i, :ep_len] = hours
        val_masks[i, :ep_len] = 1.0
        evt_times[i, :ep_len] = hours
        evt_masks[i, :ep_len] = 1.0

        n_text_this_ep = 0
        for t in range(ep_len):
            for f, (name, _, _, _, _) in enumerate(numeric):
                if rng.random() < 0.5:
                    num_ind[i, t, f] = 1.0
                    lo, hi = NUMERIC_RAW_RANGES[name]
                    num_raw[f][i, t, 0] = rng.uniform(lo, hi)

            for f, feat in enumerate(categorical):
                if rng.random() < 0.4:
                    cat_ind[i, t, f] = 1.0
                    cat_vals[f][i, t, :] = _one_hot(int(rng.integers(0, feat[2])), feat[2])

            for f, feat in enumerate(ordinal):
                if rng.random() < 0.3:
                    ord_ind[i, t, f] = 1.0
                    ord_vals[f][i, t, :] = _one_hot(int(rng.integers(0, feat[2])), feat[2])

            for f in range(n_txt):
                if rng.random() < 0.2:
                    txt_ind[i, t, f] = 1.0
                    note = SAMPLE_NOTES[int(rng.integers(0, len(SAMPLE_NOTES)))]
                    ids, mask = _tokenize(tk, note)
                    text_rows.append(ids)
                    text_mask_rows.append(mask)
                    text_ts_list.append(t)
                    n_text_this_ep += 1

            for f in range(n_evt):
                if rng.random() < 0.25:
                    evt_ind[i, t, f] = 1.0

        text_counts.append(n_text_this_ep)

    # ---- static ----
    static_dims = [f[2] for f in STATIC_FEATS]
    static_total = int(sum(static_dims))
    static_data = np.zeros((n_ep, static_total), dtype=np.float32)
    for i in range(n_ep):
        offset = 0
        for name, ftype, size, cmap, _ in STATIC_FEATS:
            if ftype == "numeric":
                lo, hi = NUMERIC_RAW_RANGES.get(name, (0.0, 1.0))
                static_data[i, offset] = rng.uniform(lo, hi)
            elif ftype == "categorical":
                static_data[i, offset] = float(rng.integers(0, size))
            offset += size

    # ---- standardize numerics (mirrors standardize_feats) ----
    means = np.zeros(n_num, dtype=np.float32)
    p5 = np.zeros(n_num, dtype=np.float32)
    p95 = np.zeros(n_num, dtype=np.float32)
    for f in range(n_num):
        mask = num_ind[:, :, f] == 1.0
        if mask.any():
            observed = num_raw[f][mask]
            means[f] = observed.mean()
            norms = np.linalg.norm(observed, ord=2, axis=-1)
            p5[f] = np.percentile(norms, 5)
            p95[f] = np.percentile(norms, 95)
        std = num_raw[f].copy()
        if p95[f] != p5[f]:
            std = (std - means[f]) / (p95[f] - p5[f])
        else:
            std[:] = 0.0
        std[num_ind[:, :, f] == 0.0] = 0.0
        num_vals[f][:] = std

    mortality = rng.integers(0, 2, size=n_ep).astype(np.float32)
    length_of_stay = rng.uniform(24, 240, size=n_ep).astype(np.float32)
    phenotype_dim = 2
    phenotype = rng.integers(0, 2, size=(n_ep, phenotype_dim)).astype(np.float32)

    # ---- sparse text (CSR) ----
    offsets = np.zeros(n_ep + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(text_counts)
    if text_rows:
        text_values = np.stack(text_rows, axis=0)
        text_masks = np.stack(text_mask_rows, axis=0)
        text_ts = np.array(text_ts_list, dtype=np.int32)
    else:
        text_values = np.zeros((0, MAX_TOKEN_LENGTH), dtype=np.int64)
        text_masks = np.zeros((0, MAX_TOKEN_LENGTH), dtype=np.float32)
        text_ts = np.zeros(0, dtype=np.int32)

    # -----------------------------------------------------------------
    # XAI score arrays — ONE SCORE PER FEATURE PER TIMESTEP, so these
    # mirror the INDICATOR arrays. Scores are zero where nothing was
    # recorded (gradient x input is zero for absent inputs).
    # Text is the exception: one score per token.
    # -----------------------------------------------------------------
    def scores_like(indicator):
        a = np.zeros(indicator.shape, dtype=np.float32)
        m = indicator == 1.0
        a[m] = rng.uniform(-1, 1, size=int(m.sum()))
        return a

    xai_numeric = scores_like(num_ind)          # (n_ep, max_ts, n_num)
    xai_categorical = scores_like(cat_ind)      # (n_ep, max_ts, n_cat)
    xai_ordinal = scores_like(ord_ind)          # (n_ep, max_ts, n_ord)
    xai_event = scores_like(evt_ind)            # (n_ep, max_ts, n_evt)
    # One score per static feature (not per slot of its encoding).
    xai_static = rng.uniform(-1, 1, size=(n_ep, n_static)).astype(np.float32)
    # One score per real (non-padding) token.
    xai_text = np.zeros_like(text_values, dtype=np.float32)
    real = text_masks == 1.0
    xai_text[real] = rng.uniform(-1, 1, size=int(real.sum()))

    # -----------------------------------------------------------------
    # Write to disk
    # -----------------------------------------------------------------
    train_dir = out_dir / "train"
    xai_dir = out_dir / "train_xai"
    train_dir.mkdir(parents=True, exist_ok=True)
    xai_dir.mkdir(parents=True, exist_ok=True)

    def save(d, name, arr):
        np.save(d / f"{name}.npy", arr)

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

    with open(train_dir / "metadata.pkl", "wb") as fh:
        pickle.dump({
            "max_ts_len": max_ts,
            "text_token_len": [MAX_TOKEN_LENGTH] * n_txt,
            "text_embed_dim": 0,
            "n_numeric_feats": n_num,
            "n_categorical_feats": n_cat,
            "n_ordinal_feats": n_ord,
            "n_text_feats": n_txt,
        }, fh)

    # Score arrays (one file per feature type, mirroring the indicators)
    save(xai_dir, "xai_numeric", xai_numeric)
    save(xai_dir, "xai_categorical", xai_categorical)
    save(xai_dir, "xai_ordinal", xai_ordinal)
    save(xai_dir, "xai_event", xai_event)
    save(xai_dir, "xai_static", xai_static)
    for f in range(n_txt):
        save(xai_dir, f"xai_text_{f}", xai_text)

    episode_ids = [9, 20, 2][:n_ep] + list(range(1000, 1000 + max(0, n_ep - 3)))
    with open(out_dir / "train_ids.pkl", "wb") as fh:
        pickle.dump(episode_ids, fh)

    np.savez(out_dir / "summary_statistics_train.npz", means=means, p5=p5, p95=p95)

    # variable_properties.yaml — feature types/sizes/category maps/descriptions
    var_props = {}
    for name, ftype, size, cmap, desc in VALUED_FEATS + TEXT_FEATS + STATIC_FEATS:
        entry = {"type": ftype, "size": size, "description": desc}
        if cmap is not None:
            entry["category_map"] = cmap
        var_props[name] = entry
    for name, desc in EVENT_FEATS:
        var_props[name] = {"type": "event", "size": 1, "description": desc}
    with open(out_dir / "variable_properties.yaml", "w") as fh:
        yaml.safe_dump(var_props, fh, sort_keys=False)

    # Dataset config — same shape as TransEHR2/configs/datasets/*.yaml.
    # These ordered lists are what dictate array column order; the reader
    # re-derives the per-type ordering from them, so nothing is duplicated.
    dataset_config = {
        "VARIABLE_PROPERTIES_PATH": "variable_properties.yaml",
        "VALUED_FEATS": [f[0] for f in VALUED_FEATS],
        "EVENT_FEATS": [f[0] for f in EVENT_FEATS],
        "TEXT_FEATS": [f[0] for f in TEXT_FEATS],
        "STATIC_FEATS": [f[0] for f in STATIC_FEATS],
        "MAX_EPISODE_LEN_STEPS": max_ts,
        "MAX_HISTORY_LEN_STEPS": 0,
    }
    with open(out_dir / "dataset_config.yaml", "w") as fh:
        yaml.safe_dump(dataset_config, fh, sort_keys=False)

    print(f"Wrote fake dataset for {n_ep} episodes (max_ts={max_ts}) to {out_dir}/")
    print(f"  episodes:      {episode_ids}")
    print(f"  numeric:       {[f[0] for f in numeric]}   -> xai_numeric {xai_numeric.shape}")
    print(f"  categorical:   {[f[0] for f in categorical]} -> xai_categorical {xai_categorical.shape}")
    print(f"  ordinal:       {[f[0] for f in ordinal]}     -> xai_ordinal {xai_ordinal.shape}")
    print(f"  event:         {[f[0] for f in EVENT_FEATS]} -> xai_event {xai_event.shape}")
    print(f"  static:        {[f[0] for f in STATIC_FEATS]} -> xai_static {xai_static.shape}")
    print(f"  text:          {[f[0] for f in TEXT_FEATS]} -> xai_text_0 {xai_text.shape} "
          f"({len(text_values)} notes, real Llama tokenizer, {MAX_TOKEN_LENGTH} tokens)")


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
