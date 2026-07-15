"""
Generate mock XAI importance scores for a patient.

The event data (timestamp, source, description) is real, loaded the same way
the timeline viewer loads it. Only the importance_score column is fabricated,
standing in for scores that would eventually come from a real trained
TransEHR2 model + XAI method. No real model exists yet, so this lets the
webapp's XAI list/highlight routines be built and tested now.

Run with:  python generate_xai_scores.py --patient_id 20 --n_items 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rmt23345_events import load_all_events


def generate_fake_importance_scores(
    patient_id: str,
    config: dict,
    rng: np.random.Generator,
    n_items: int = 20,
) -> pd.DataFrame:
    """Sample real events for a patient and attach a fake importance score to each.

    `rng` is shared across patients (passed in, not re-seeded per call) so
    different patients get different scores instead of an identical sequence.
    """
    events = load_all_events(config=config, patient_id=patient_id)
    if events.empty:
        raise ValueError(f"No events found for patient {patient_id}")

    n = min(n_items, len(events))
    sample = events.sample(n=n, random_state=rng.integers(2**31)).reset_index(drop=True)

    return pd.DataFrame({
        "patient_id":       sample["patient_id"],
        "timestamp":        sample["start_date"],
        "feature_source":   sample["source_dataset"],
        "feature_name":     sample["event_type"],
        "feature_value":    sample["event_info"],
        "importance_score": rng.uniform(-1, 1, size=n).round(3),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient_id", required=True, nargs="+", help="One or more patient IDs")
    parser.add_argument("--config", default="study_config.yaml", help="Path to study_config.yaml")
    parser.add_argument("--n_items", type=int, default=20, help="Number of fake-scored items per patient")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="xai_scores/importance_scores.csv", help="Output CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    all_scores = pd.concat(
        [generate_fake_importance_scores(pid, config, rng, n_items=args.n_items)
         for pid in args.patient_id],
        ignore_index=True,
    )
    all_scores.to_csv(out_path, index=False)
    print(f"wrote {len(all_scores)} rows for {len(args.patient_id)} patient(s) -> {out_path}")


if __name__ == "__main__":
    main()
