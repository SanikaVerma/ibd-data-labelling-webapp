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

    # ----- helpers -----
    def _load(self, d: Path, name: str) -> np.ndarray:
        return np.load(d / f"{name}.npy", mmap_mode="r")

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
