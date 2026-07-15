"""
XAI module backend routines (mock-up).

Loads pre-computed (fake, for now) importance scores for a patient and
ranks them for display. See generate_xai_scores.py for how the scores file
is produced, and study_config.yaml's `xai_scores` section for its location.
"""
import pandas as pd


def get_top_importance_items(patient_id: str, config: dict, n: int = 15) -> list[dict]:
    """Return the top-n (by absolute importance score) items for a patient.

    Args:
        patient_id: Patient to filter to.
        config:     Dict loaded from study_config.yaml. Must have an
                    `xai_scores.file` key pointing to the scores CSV.
        n:          Max number of items to return (10-20 recommended, per
                    the XAI plan doc, so the list doesn't overwhelm the user).

    Returns:
        List of dicts sorted by decreasing absolute importance score, each
        with: timestamp, feature_source, feature_name, feature_value, score.
        Empty list if the patient has no scored items.
    """
    xai_cfg = (config or {}).get("xai_scores") or {}
    scores_path = xai_cfg.get("file")
    if not scores_path:
        return []

    df = pd.read_csv(scores_path, parse_dates=["timestamp"])
    df = df[df["patient_id"].astype(str) == str(patient_id)]
    if df.empty:
        return []

    df = df.reindex(df["importance_score"].abs().sort_values(ascending=False).index)
    top = df.head(n)

    return [
        {
            "timestamp": row.timestamp,
            "feature_source": row.
        feature_source,
            "feature_name": row.feature_name,
            "feature_value": row.feature_value,
            "score": row.importance_score,
        }
        for row in top.itertuples()
    ]
