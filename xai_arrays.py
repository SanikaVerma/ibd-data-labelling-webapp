"""
Map TransEHR2 XAI importance scores (numpy arrays on disk) back to intelligible,
per-feature records that can be rendered on the timeline.

This is the array-based backend that replaces the earlier CSV mock-up. It reads
the on-disk arrays produced by TransEHR2's preprocessing.py (see
generate_fake_arrays.py for a synthetic stand-in with the identical layout) and
the parallel XAI score arrays, then walks the "importance score -> feature ->
original value" chain the supervisor described.

Implemented so far: numeric features. Categorical/ordinal, event, static and
text follow (in that order of difficulty).

A record looks like:
    {
      "episode_id":  9,
      "feature":     "CRP",
      "description": "C-reactive protein",
      "kind":        "numeric",
      "timestep":    4,
      "time_hours":  27.0,
      "value":       12.4,          # original value, standardization reversed
      "score":       0.83,          # importance score
    }
"""
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


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

        with open(self.base / "variable_properties.yaml") as f:
            self.var_props = yaml.safe_load(f)

        # Feature roles + array-column order. In the real pipeline this comes
        # from the reader config (valued_feats / event_feats / static_feats),
        # NOT from variable_properties.yaml — types alone (numeric/categorical)
        # do not distinguish value-associated vs static vs event features.
        with open(self.base / "feature_layout.json") as f:
            layout = json.load(f)
        self.numeric_feats = layout["numeric_feats"]
        self.categorical_feats = layout["categorical_feats"]
        self.ordinal_feats = layout["ordinal_feats"]
        self.text_feats = layout["text_feats"]
        self.event_feats = layout["event_feats"]
        self.static_feats = layout["static_feats"]

        stats = np.load(self.base / f"summary_statistics_{suffix}.npz")
        self.num_means = stats["means"]
        self.num_p5 = stats["p5"]
        self.num_p95 = stats["p95"]

        self.val_times = self._load(self.data_dir, "val_times")
        self.val_masks = self._load(self.data_dir, "val_masks")

    # ----- loading helpers -----
    def _load(self, d: Path, name: str) -> np.ndarray:
        return np.load(d / f"{name}.npy", mmap_mode="r")

    def _row_for_episode(self, episode_id) -> int:
        try:
            return self.episode_ids.index(episode_id)
        except ValueError:
            raise ValueError(f"episode_id {episode_id} not found in {self.suffix}_ids.pkl")

    # ----- numeric -----
    def _reverse_standardize(self, feat_idx: int, std_value: float) -> float:
        """Recover the original numeric value from its standardized form.

        Mirrors standardize_feats: standardized = (value - mean) / (p95 - p5).
        """
        spread = self.num_p95[feat_idx] - self.num_p5[feat_idx]
        if spread == 0:
            return float(self.num_means[feat_idx])
        return float(std_value * spread + self.num_means[feat_idx])

    def numeric_records(self, episode_id) -> list:
        """Return one record per recorded numeric (timestep, feature) for a patient."""
        row = self._row_for_episode(episode_id)
        indicators = self._load(self.data_dir, "val_numeric_indicators")  # (n_ep, ts, n_num)
        records = []
        for f, feat in enumerate(self.numeric_feats):
            values = self._load(self.data_dir, f"val_numeric_values_{f}")  # (n_ep, ts, dim)
            scores = self._load(self.xai_dir, f"xai_numeric_{f}")          # (n_ep, ts, dim)
            ind_f = indicators[row, :, f]
            for t in np.nonzero(ind_f == 1.0)[0]:
                std_val = float(values[row, t, 0])
                records.append({
                    "episode_id": episode_id,
                    "feature": feat,
                    "description": self.var_props[feat].get("description", "") or feat,
                    "kind": "numeric",
                    "timestep": int(t),
                    "time_hours": float(self.val_times[row, t]),
                    "value": round(self._reverse_standardize(f, std_val), 3),
                    "score": round(float(scores[row, t, 0]), 3),
                })
        return records

    # ----- categorical / ordinal (one-hot -> decoded category) -----
    def _cat_map(self, feat: str) -> dict:
        """Index -> original value map, with int keys (yaml may load them as int)."""
        cmap = self.var_props[feat].get("category_map", {}) or {}
        return {int(k): v for k, v in cmap.items()}

    def _onehot_records(self, episode_id, feats, ind_name, val_prefix,
                        xai_prefix, kind) -> list:
        """Shared logic for categorical and ordinal (identical storage)."""
        row = self._row_for_episode(episode_id)
        indicators = self._load(self.data_dir, ind_name)  # (n_ep, ts, n_feats)
        records = []
        for f, feat in enumerate(feats):
            values = self._load(self.data_dir, f"{val_prefix}_{f}")  # (n_ep, ts, n_classes)
            scores = self._load(self.xai_dir, f"{xai_prefix}_{f}")   # same shape
            cmap = self._cat_map(feat)
            for t in np.nonzero(indicators[row, :, f] == 1.0)[0]:
                active = int(np.argmax(values[row, t]))
                # Sum over one-hot classes: only the active class contributes.
                score = float(scores[row, t].sum())
                records.append({
                    "episode_id": episode_id,
                    "feature": feat,
                    "description": self.var_props[feat].get("description", "") or feat,
                    "kind": kind,
                    "timestep": int(t),
                    "time_hours": float(self.val_times[row, t]),
                    "value": cmap.get(active, f"<code {active}>"),
                    "score": round(score, 3),
                })
        return records

    def categorical_records(self, episode_id) -> list:
        return self._onehot_records(
            episode_id, self.categorical_feats, "val_categorical_indicators",
            "val_categorical_values", "xai_categorical", "categorical")

    def ordinal_records(self, episode_id) -> list:
        return self._onehot_records(
            episode_id, self.ordinal_feats, "val_ordinal_indicators",
            "val_ordinal_values", "xai_ordinal", "ordinal")

    # ----- events (indicator only, own time axis, no value) -----
    def event_records(self, episode_id) -> list:
        row = self._row_for_episode(episode_id)
        indicators = self._load(self.data_dir, "event_indicators")  # (n_ep, ts, n_event)
        scores = self._load(self.xai_dir, "xai_event")              # same shape
        event_times = self._load(self.data_dir, "event_times")
        records = []
        for f, feat in enumerate(self.event_feats):
            for t in np.nonzero(indicators[row, :, f] == 1.0)[0]:
                records.append({
                    "episode_id": episode_id,
                    "feature": feat,
                    "description": self.var_props.get(feat, {}).get("description", "") or feat,
                    "kind": "event",
                    "timestep": int(t),
                    "time_hours": float(event_times[row, t]),
                    "value": "occurred",
                    "score": round(float(scores[row, t, f]), 3),
                })
        return records

    # ----- static (patient-level, no timestep) -----
    def static_records(self, episode_id) -> list:
        """One record per static feature. Static is time-invariant, so these are
        patient-level attributes rather than timeline events."""
        row = self._row_for_episode(episode_id)
        static = self._load(self.data_dir, "static_data")  # (n_ep, static_total_dim)
        scores = self._load(self.xai_dir, "xai_static")    # same shape
        records = []
        offset = 0
        for feat in self.static_feats:
            props = self.var_props[feat]
            size = int(props["size"])
            ftype = props["type"]
            raw = float(static[row, offset])  # value sits at the feature's first slot
            if ftype == "numeric":
                value = round(raw, 3)
            else:  # categorical/ordinal static: stored as the category index
                value = self._cat_map(feat).get(int(raw), f"<code {int(raw)}>")
            # Sum the feature's slot range for its total contribution.
            score = float(scores[row, offset:offset + size].sum())
            records.append({
                "episode_id": episode_id,
                "feature": feat,
                "description": props.get("description", "") or feat,
                "kind": "static",
                "timestep": None,
                "time_hours": None,
                "value": value,
                "score": round(score, 3),
            })
            offset += size
        return records

    # ----- top-N across all time-associated features -----
    def top_records(self, episode_id, n: int = 15) -> list:
        """Top-n timeline records by absolute importance score for a patient.

        Combines numeric, categorical, ordinal and event features (all
        time-associated). Static features are patient-level and returned
        separately via static_records().
        """
        records = (self.numeric_records(episode_id)
                   + self.categorical_records(episode_id)
                   + self.ordinal_records(episode_id)
                   + self.event_records(episode_id))
        records.sort(key=lambda r: abs(r["score"]), reverse=True)
        return records[:n]
