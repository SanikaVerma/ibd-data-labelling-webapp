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
Array column order follows the order features appear in variable_properties.yaml
(the model author's assumption: TransEHR2 builds its arrays in that order, so a
feature's array column matches its position in that file, with no need to look at
the source CSVs). The dataset config still supplies each feature's ROLE
(value-associated / event / static / text), because variable_properties.yaml
records a feature's type but not its role. Within a role, features are ordered by
their position in variable_properties.yaml.

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
    # loads metadata, patient ID list, standardization stats, derives feature order
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

        # Dataset config: tells us each feature's ROLE (value-associated / event /
        # static / text). variable_properties.yaml gives a feature's type but not
        # its role, so the role lists still come from here. (Open question with the
        # model author on whether role could live in variable_properties.yaml too.)
        with open(self.base / "dataset_config.yaml") as f:
            cfg = yaml.safe_load(f)

        var_props_path = self.base / cfg.get("VARIABLE_PROPERTIES_PATH", "variable_properties.yaml")
        with open(var_props_path) as f:
            self.var_props = yaml.safe_load(f)

        # ORDER assumption (per the model author): TransEHR2 builds the arrays with
        # features in the order they APPEAR IN variable_properties.yaml. So a
        # feature's array column follows its position in that file — no need to
        # look at the source CSVs. We reorder each role's features accordingly.
        vp_pos = {name: i for i, name in enumerate(self.var_props.keys())}
        def in_vp_order(names):
            return sorted(names, key=lambda n: vp_pos.get(n, len(vp_pos)))

        valued = in_vp_order(cfg.get("VALUED_FEATS", []) or [])
        self.numeric_feats = [n for n in valued if self.var_props[n]["type"] == "numeric"]
        self.categorical_feats = [n for n in valued if self.var_props[n]["type"] == "categorical"]
        self.ordinal_feats = [n for n in valued if self.var_props[n]["type"] == "ordinal"]
        self.text_feats = in_vp_order(cfg.get("TEXT_FEATS", []) or [])
        self.event_feats = in_vp_order(cfg.get("EVENT_FEATS", []) or [])
        self.static_feats = in_vp_order(cfg.get("STATIC_FEATS", []) or [])

        # Path to the DPD file (din_drug_info.csv) for DIN -> brand-name lookup.
        self.drug_info_path = cfg.get("DRUG_INFO_PATH")
        self._dpd = None  # loaded lazily on first drug lookup

        stats = np.load(self.base / f"summary_statistics_{suffix}.npz")
        self.num_means, self.num_p5, self.num_p95 = stats["means"], stats["p5"], stats["p95"]

        self.val_times = self._load(self.data_dir, "val_times")
        # Absolute admission datetime per episode (row-aligned), if the extractor
        # wrote it (index_times.npy). Lets us recover real dates: event datetime =
        # index_time + time_hours. None when absent (older arrays) -> show hours.
        try:
            self.index_times = np.load(self.data_dir / "index_times.npy")
        except FileNotFoundError:
            self.index_times = None
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

    def index_time(self, episode_id):
        """Absolute admission datetime (pandas Timestamp) for this episode, or
        None if index_times.npy wasn't written (then times stay relative hours)."""
        if self.index_times is None:
            return None
        import pandas as pd
        return pd.Timestamp(self.index_times[self._row_for_episode(episode_id)])

    def event_datetime(self, episode_id, time_hours):
        """Real datetime of an event = index_time + time_hours, or None if there's
        no index_times / no time."""
        if time_hours is None:
            return None
        idx = self.index_time(episode_id)
        if idx is None:
            return None
        import pandas as pd
        return idx + pd.Timedelta(hours=float(time_hours))

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

    # ----- drugs (per-drug scores, DIN -> brand name via the DPD file) -----
    def _dpd_lookup(self) -> dict:
        """DIN (int) -> {brand, ingredient, strength, class} from din_drug_info.csv.

        The DPD lists one row per ingredient, so a DIN can repeat; we keep the
        first row per DIN and join ingredient names. Non-numeric DINs (the file
        uses 'Not Applicable' in places) are skipped.
        """
        if self._dpd is not None:
            return self._dpd
        self._dpd = {}
        if not self.drug_info_path:
            return self._dpd
        import pandas as pd
        cols = ["DRUG_IDENTIFICATION_NUMBER", "BRAND_NAME", "INGREDIENT",
                "STRENGTH", "STRENGTH_UNIT", "CLASS"]
        df = pd.read_csv(self.drug_info_path, usecols=lambda c: c in cols, dtype=str)
        df = df[df["DRUG_IDENTIFICATION_NUMBER"].str.fullmatch(r"\d+", na=False)].fillna("")
        for din, g in df.groupby("DRUG_IDENTIFICATION_NUMBER"):
            first = g.iloc[0]
            self._dpd[int(din)] = {
                "brand": str(first.get("BRAND_NAME", "")).strip(),
                "ingredient": " / ".join(sorted({s for s in g["INGREDIENT"] if s})),
                "strength": str(first.get("STRENGTH", "")).strip(),
                "strength_unit": str(first.get("STRENGTH_UNIT", "")).strip(),
                "class": str(first.get("CLASS", "")).strip(),
            }
        return self._dpd

    def drug_records(self, episode_id) -> list:
        """One record per dispensed drug for a patient, with its brand name.

        Drugs are stored sparsely (CSR) like text: `drug_offsets` gives the
        patient's slice, `drug_timesteps` the timestep of each dispensing entry,
        and each entry has up to 30 slots. For each real slot we read the DIN,
        dose and score, then look the DIN up in the DPD for the brand name.
        """
        row = self._row_for_episode(episode_id)
        try:
            offsets = self._load(self.data_dir, "drug_offsets")
        except FileNotFoundError:
            return []
        start, end = int(offsets[row]), int(offsets[row + 1])
        if end <= start:
            return []
        dins = self._load(self.data_dir, "drug_dins")
        doses = self._load(self.data_dir, "drug_doses")
        masks = self._load(self.data_dir, "drug_masks")
        timesteps = self._load(self.data_dir, "drug_timesteps")
        scores = self._load(self.xai_dir, "xai_drug")
        dpd = self._dpd_lookup()

        out = []
        for j in range(start, end):
            t = int(timesteps[j])
            for k in np.nonzero(np.asarray(masks[j]) == 1.0)[0]:
                din = int(dins[j, k])
                info = dpd.get(din)
                if info and info["brand"]:
                    label = info["brand"]
                    detail = f"{info['ingredient']} {info['strength']}{info['strength_unit']}".strip()
                    value = f"{detail} (DIN {din})" if detail else f"DIN {din}"
                else:
                    label = f"DIN {din}"
                    value = f"DIN {din}"
                rec = self._record(episode_id, label, "drug", t,
                                   float(self.val_times[row, t]) if t < self.val_times.shape[1] else None,
                                   value, scores[j, k])
                rec["din"] = din
                rec["dose"] = round(float(doses[j, k]), 3)
                out.append(rec)
        return out

    # ----- array data -> timeline-renderable events -----
    def to_timeline_events(self, episode_id, admission_time=None, include_text=True) -> list:
        """Convert this episode's array records into the shape the timeline renders.

        The timeline draws events shaped like load_all_events' output
        (patient_id / start_date / end_date / event_type / event_info /
        source_dataset), so mapping array records into that shape lets the same
        rendering path draw model-input data instead of raw CSV events. This is
        the link needed before any hover/highlight work, since importance scores
        are always with respect to the arrays, not the CSVs.

        Args:
            episode_id: patient episode to convert.
            admission_time: the episode's real admission datetime. Array times
                are hours relative to it. If None, calendar dates cannot be
                recovered — start_date/end_date are left None and `time_hours`
                carries the relative time instead. (Open question with the model
                author: the admission timestamp isn't stored in the arrays.)
            include_text: include note records (one event per note).

        Returns:
            List of dicts, each also carrying `score`, `feature` and `kind` so a
            timeline item can be tied back to its importance score.
        """
        import pandas as pd

        records = (self.numeric_records(episode_id)
                   + self.categorical_records(episode_id)
                   + self.ordinal_records(episode_id)
                   + self.event_records(episode_id))
        if include_text:
            records += self.text_records(episode_id)

        events = []
        for r in records:
            hours = r["time_hours"]
            if admission_time is not None and hours is not None:
                start = pd.Timestamp(admission_time) + pd.Timedelta(hours=hours)
            else:
                start = None
            events.append({
                "patient_id": r["episode_id"],
                "start_date": start,
                "end_date": start,          # array records are point-in-time
                "time_hours": hours,        # kept so relative time survives
                "event_type": r["kind"],    # lane: numeric / categorical / event / text ...
                "event_info": f"{r['description']}: {r['value']}",
                "source_dataset": "TransEHR2",
                "feature": r["feature"],
                "kind": r["kind"],
                "score": r["score"],
            })
        events.sort(key=lambda e: (e["time_hours"] is None, e["time_hours"]))
        return events

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
                   + self.event_records(episode_id)
                   + self.drug_records(episode_id))
        records.sort(key=lambda r: abs(r["score"]), reverse=True)
        return records[:n]
