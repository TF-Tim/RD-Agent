from pathlib import Path
from typing import Optional, List

import pandas as pd
import qlib

# Initialize qlib with default config; rely on environment that RD-Agent sets up
qlib.init()

from qlib.workflow import R  # noqa: E402


def _safe_list_metrics(recorder) -> pd.DataFrame:
    """
    Return metrics as a tidy DataFrame with columns ['metric', 'value'].
    Works whether list_metrics() returns a dict or list-like.
    """
    try:
        m = recorder.list_metrics()
        if isinstance(m, dict):
            return pd.DataFrame(list(m.items()), columns=["metric", "value"])
        # Fallback: try to coerce to Series then DataFrame
        s = pd.Series(m)
        if s.index.dtype == "object":
            return s.reset_index().rename(columns={"index": "metric", 0: "value"})
        return pd.DataFrame({"metric": s.index.astype(str), "value": s.values})
    except Exception:
        return pd.DataFrame(columns=["metric", "value"])


def _choose_latest_completed_recorder() -> Optional[object]:
    """
    Iterate all experiments & recorders; pick the one with the latest valid end_time.
    Fallback to the one with latest start_time if end_time is missing.
    """
    latest = None
    latest_end = None
    latest_start = None

    exps: List[str] = R.list_experiments() or []
    for exp in exps:
        rec_ids = R.list_recorders(experiment_name=exp) or []
        for rid in rec_ids:
            try:
                rec = R.get_recorder(recorder_id=rid, experiment_name=exp)
                info = getattr(rec, "info", {}) or {}
                end_time = info.get("end_time")
                start_time = info.get("start_time")
                # Prefer completed runs with end_time
                if end_time is not None:
                    if latest_end is None or end_time > latest_end:
                        latest, latest_end, latest_start = rec, end_time, start_time
                else:
                    # Fallback by start_time when no end_time is available
                    if latest is None and start_time is not None:
                        if latest_start is None or start_time > latest_start:
                            latest, latest_start = rec, start_time
            except Exception:
                continue

    return latest


def _load_any_report(recorder) -> Optional[pd.DataFrame]:
    """
    Try several common report artifact keys and return the first one that loads.
    """
    candidates = [
        "portfolio_analysis/report_normal_1day.pkl",
        "portfolio_analysis/report_normal.pkl",
        "portfolio_analysis/analysis.pkl",
    ]
    for key in candidates:
        try:
            obj = recorder.load_object(key)
            if isinstance(obj, pd.DataFrame):
                return obj
            # some qlib versions return dict-like with 'summary' or 'report'
            if hasattr(obj, "get"):
                df = obj.get("report") or obj.get("summary")
                if isinstance(df, pd.DataFrame):
                    return df
        except Exception:
            pass
    return None


def main():
    out_dir = Path(__file__).resolve().parent

    recorder = _choose_latest_completed_recorder()
    if recorder is None:
        print("No recorders found")
        return

    print(f"Latest recorder: {recorder}")

    # Save metrics
    metrics_df = _safe_list_metrics(recorder)
    metrics_csv = out_dir / "qlib_res.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to {metrics_csv}")

    # Save return/analysis report if available
    report_df = _load_any_report(recorder)
    if report_df is not None:
        ret_pkl = out_dir / "ret.pkl"
        report_df.to_pickle(ret_pkl)
        print(f"Saved portfolio report to {ret_pkl}")
    else:
        print("No portfolio report artifact found among known keys.")

if __name__ == "__main__":
    main()
