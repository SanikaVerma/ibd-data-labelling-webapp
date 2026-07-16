"""
Map TransEHR2 XAI importance scores (numpy arrays on disk) back to intelligible,
per-feature records that can be rendered on the timeline.

Reads the on-disk arrays produced by TransEHR2's preprocessing.py (see
generate_fake_arrays.py for a synthetic stand-in with the identical layout) plus
the parallel XAI score arrays, and walks the chain:

    score at [episode row, timestep, feature column]
      -> patient episode id   (via {suffix}_ids.pkl)
      -> feature name         (via the dataset config's ordered feature lists)
      -> type / description / code map  (via variable_properties.yaml)
      -> original value       (reverse standardization, or category decode)
      -> time                 (via val_times / event_times)

FEATURE ORDER
-------------
Array column order is derived, not stored. It follows the dataset config's
ordered feature lists, split by the `type` declared in variable_properties.yaml,
preserving order of appearance — exactly what DataProcessor.__init__ does:

    numeric_feats     = [f for f in VALUED_FEATS if type(f) == 'numeric']
    categorical_feats = [f for f in VALUED_FEATS if type(f) == 'categorical']
    ordinal_feats     = [f for f in VALUED_FEATS if type(f) == 'ordinal']

Reproducing that rule here keeps the mapping traceable without duplicating any
bookkeeping alongside the arrays.

SCORE SHAPES
------------
One score per feature per timestep — the dot product of a feature's vector with
the gradient happens upstream when the score is produced. So score arrays mirror
the INDICATOR arrays. Text is the exception: one score per token.
"""
import os
import pickle
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

# Matches LLM_NAME / TOKENIZER_PAD_TOKEN in TransEHR2/constants.py — the text
# token IDs in the arrays are from this tokenizer's vocabulary, so decoding
# requires the same one.
LLM_NAME = "meta-llama/Llama-3.1-70B"
TOKENIZER_PAD_TOKEN = "[PAD]"


class XaiArrayReader:
    """Loads on-disk input + XAI score arrays and maps scores back to features."""

    def __init__(self, base_dir: str, suffix: str = "train"):
        self.base = Path(base_dir)
        self.suffix = suffix
        self.data_dir = self.base / suffix
        self.xai_dir = self.base / f"{suffix}_xai"

        with open(self.data_dir / "metadata.pkl", "rb") as f:
            self.meta = pickle.load(f)

        with open(self.base / f"{suffix}_ids.pkl", "rb") as f:
            self.episode_ids = list(pickle.load(f))

        # Dataset config: the ordered feature lists that dictate array columns.
        with open(self.base / "dataset_config.yaml") as f:
            cfg = yaml.safe_load(f)

        var_props_path = self.base / cfg.get("VARIABLE_PROPERTIES_PATH", "variable_properties.yaml")
        with open(var_props_path) as f:
            self.var_props = yaml.safe_load(f)

        # Re-derive per-type ordering exactly as DataProcessor.__init__ does:
        # walk VALUED_FEATS in order, bucket by the type in variable_properties.
        valued = cfg.get("VALUED_FEATS", []) or []
        self.numeric_feats, self.categorical_feats, self.ordinal_feats = [], [], []
        for name in valued:
            ftype = self.var_props[name]["type"]
            if ftype == "numeric":
                self.numeric_feats.append(name)
            elif ftype == "categorical":
                self.categorical_feats.append(name)
            elif ftype == "ordinal":
                self.ordinal_feats.append(name)
        # These keep their config order directly.
        self.text_feats = cfg.get("TEXT_FEATS", []) or []
        self.event_feats = cfg.get("EVENT_FEATS", []) or []
        self.static_feats = cfg.get("STATIC_FEATS", []) or []

        stats = np.load(self.base / f"summary_statistics_{suffix}.npz")
        self.num_means, self.num_p5, self.num_p95 = stats["means"], stats["p5"], stats["p95"]

        self.val_times = self._load(self.data_dir, "val_times")
        self._tokenizer = None  # loaded lazily — only text needs it

    # ----- helpers -----
    def _load(self, d: Path, name: str) -> np.ndarray:
        return np.load(d / f"{name}.npy", mmap_mode="r")

    @property
    def tokenizer(self):
        """The Llama tokenizer, loaded on first use (text decoding only)."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            load_dotenv()
            tk = AutoTokenizer.from_pretrained(LLM_NAME, token=os.getenv("HF_READ_TOKEN"))
            tk.add_special_tokens({"pad_token": TOKENIZER_PAD_TOKEN})
            self._tokenizer = tk
        return self._tokenizer

    def _row_for_episode(self, episode_id) -> int:
        try:
            return self.episode_ids.index(episode_id)
        except ValueError:
            raise ValueError(f"episode_id {episode_id} not found in {self.suffix}_ids.pkl")

    def _cat_map(self, feat: str) -> dict:
        cmap = self.var_props[feat].get("category_map", {}) or {}
        return {int(k): v for k, v in cmap.items()}

    def _describe(self, feat: str) -> str:
        return self.var_props.get(feat, {}).get("description", "") or feat

    def _reverse_standardize(self, feat_idx: int, std_value: float) -> float:
        """Invert standardize_feats: standardized = (value - mean) / (p95 - p5)."""
        spread = self.num_p95[feat_idx] - self.num_p5[feat_idx]
        if spread == 0:
            return float(self.num_means[feat_idx])
        return float(std_value * spread + self.num_means[feat_idx])

    def _record(self, episode_id, feat, kind, t, time_hours, value, score) -> dict:
        return {
            "episode_id": episode_id,
            "feature": feat,
            "description": self._describe(feat),
            "kind": kind,
            "timestep": t,
            "time_hours": time_hours,
            "value": value,
            "score": round(float(score), 3),
        }

    # ----- numeric -----
    def numeric_records(self, episode_id) -> list:
        row = self._row_for_episode(episode_id)
        ind = self._load(self.data_dir, "val_numeric_indicators")
        scores = self._load(self.xai_dir, "xai_numeric")  # (n_ep, ts, n_numeric)
        out = []
        for f, feat in enumerate(self.numeric_feats):
            values = self._load(self.data_dir, f"val_numeric_values_{f}")
            for t in np.nonzero(ind[row, :, f] == 1.0)[0]:
                value = round(self._reverse_standardize(f, float(values[row, t, 0])), 3)
                out.append(self._record(episode_id, feat, "numeric", int(t),
                                        float(self.val_times[row, t]), value,
                                        scores[row, t, f]))
        return out

    # ----- categorical / ordinal (identical storage) -----
    def _onehot_records(self, episode_id, feats, ind_name, val_prefix, xai_name, kind) -> list:
        row = self._row_for_episode(episode_id)
        ind = self._load(self.data_dir, ind_name)
        scores = self._load(self.xai_dir, xai_name)  # (n_ep, ts, n_feats)
        out = []
        for f, feat in enumerate(feats):
            values = self._load(self.data_dir, f"{val_prefix}_{f}")
            cmap = self._cat_map(feat)
            for t in np.nonzero(ind[row, :, f] == 1.0)[0]:
                active = int(np.argmax(values[row, t]))
                value = cmap.get(active, f"<code {active}>")
                out.append(self._record(episode_id, feat, kind, int(t),
                                        float(self.val_times[row, t]), value,
                                        scores[row, t, f]))
        return out

    def categorical_records(self, episode_id) -> list:
        return self._onehot_records(episode_id, self.categorical_feats,
                                    "val_categorical_indicators", "val_categorical_values",
                                    "xai_categorical", "categorical")

    def ordinal_records(self, episode_id) -> list:
        return self._onehot_records(episode_id, self.ordinal_feats,
                                    "val_ordinal_indicators", "val_ordinal_values",
                                    "xai_ordinal", "ordinal")

    # ----- events (indicator only, own time axis) -----
    def event_records(self, episode_id) -> list:
        row = self._row_for_episode(episode_id)
        ind = self._load(self.data_dir, "event_indicators")
        scores = self._load(self.xai_dir, "xai_event")
        event_times = self._load(self.data_dir, "event_times")
        out = []
        for f, feat in enumerate(self.event_feats):
            for t in np.nonzero(ind[row, :, f] == 1.0)[0]:
                out.append(self._record(episode_id, feat, "event", int(t),
                                        float(event_times[row, t]), "occurred",
                                        scores[row, t, f]))
        return out

    # ----- static (patient-level, no timestep) -----
    def static_records(self, episode_id) -> list:
        row = self._row_for_episode(episode_id)
        static = self._load(self.data_dir, "static_data")
        scores = self._load(self.xai_dir, "xai_static")  # (n_ep, n_static_feats)
        out = []
        offset = 0
        for f, feat in enumerate(self.static_feats):
            props = self.var_props[feat]
            size, ftype = int(props["size"]), props["type"]
            raw = float(static[row, offset])  # value sits at the feature's first slot
            if ftype == "numeric":
                value = round(raw, 3)
            else:
                value = self._cat_map(feat).get(int(raw), f"<code {int(raw)}>")
            out.append(self._record(episode_id, feat, "static", None, None,
                                    value, scores[row, f]))
            offset += size
        return out

    # ----- text (per-token scores) -----
    def text_records(self, episode_id) -> list:
        """One record per recorded note, with per-token scores decoded to words.

        Text is stored sparsely (CSR): `offsets` gives this episode's slice of
        the token/score rows, `timesteps` says which timestep each row belongs
        to. Each row is a padded token-ID sequence; the mask marks real tokens.

        Scores are per token, so each record carries the decoded note plus the
        (token, score) pairs behind it — the raw material for the heat-map.

        A single score for the whole note isn't well defined: summing the token
        scores gives the note's total contribution (gradient x input attributions
        are additive), while the largest-magnitude token says which single word
        mattered most. Both are returned; `score` defaults to the sum. Whether
        that is the right choice for ranking is an open question for the model
        author.
        """
        row = self._row_for_episode(episode_id)
        out = []
        for f, feat in enumerate(self.text_feats):
            offsets = self._load(self.data_dir, f"val_text_offsets_{f}")
            start, end = int(offsets[row]), int(offsets[row + 1])
            if end <= start:
                continue
            token_ids = self._load(self.data_dir, f"val_text_values_{f}")
            masks = self._load(self.data_dir, f"val_text_masks_{f}")
            timesteps = self._load(self.data_dir, f"val_text_timesteps_{f}")
            scores = self._load(self.xai_dir, f"xai_text_{f}")

            for j in range(start, end):
                real = np.asarray(masks[j]) == 1.0
                ids = np.asarray(token_ids[j])[real]
                sc = np.asarray(scores[j])[real]
                if ids.size == 0:
                    continue
                t = int(timesteps[j])
                # Special tokens (e.g. <|begin_of_text|>) carry scores because
                # the model sees them, but they aren't words — flag them so the
                # display can skip them rather than surfacing them as findings.
                special = set(self.tokenizer.all_special_ids)
                tokens = [
                    {
                        "token": self.tokenizer.decode([int(i)]),
                        "score": round(float(s), 3),
                        "is_special": int(i) in special,
                    }
                    for i, s in zip(ids, sc)
                ]
                score_sum = float(sc.sum())
                words = [tok for tok in tokens if not tok["is_special"]]
                peak = max(words, key=lambda x: abs(x["score"])) if words else None
                rec = self._record(
                    episode_id, feat, "text", t, float(self.val_times[row, t]),
                    self.tokenizer.decode(ids, skip_special_tokens=True), score_sum,
                )
                rec["tokens"] = tokens
                rec["score_sum"] = round(score_sum, 3)
                rec["top_token"] = peak
                out.append(rec)
        return out

    def top_tokens(self, episode_id, n: int = 10) -> list:
        """The n most important word tokens across all of a patient's notes.

        Special tokens are excluded — they score like any other input but aren't
        words, so surfacing them as findings would be noise.

        Note these are *tokens*, not words: the tokenizer splits longer words
        into pieces (e.g. "Crohn" -> " Cro" + "hn"), so a top token can be a
        fragment. Read alongside the full note rather than in isolation.
        """
        toks = []
        for rec in self.text_records(episode_id):
            for tok in rec["tokens"]:
                if tok["is_special"]:
                    continue
                toks.append({**tok, "feature": rec["feature"],
                             "timestep": rec["timestep"], "time_hours": rec["time_hours"]})
        toks.sort(key=lambda x: abs(x["score"]), reverse=True)
        return toks[:n]

    # ----- top-N across time-associated features -----
    def top_records(self, episode_id, n: int = 15) -> list:
        """Top-n timeline records by absolute importance score.

        Combines numeric, categorical, ordinal and event features. Static
        features are patient-level and returned separately via static_records().
        Text is handled separately (per-token scores).
        """
        records = (self.numeric_records(episode_id)
                   + self.categorical_records(episode_id)
                   + self.ordinal_records(episode_id)
                   + self.event_records(episode_id))
        records.sort(key=lambda r: abs(r["score"]), reverse=True)
        return records[:n]
