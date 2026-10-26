"""PowerWatch anomaly-detection Lambda handler (Python standard library only).

Method (explainable rolling baseline + Z-score):
  For every meter, each reading is compared with the previous `baseline_days`
  readings taken at the same time of day on the same day type (weekday/weekend).
  z = (observed - baseline_mean) / sigma, where
  sigma = max(sample std-dev, 5% of baseline mean, 0.01 kWh).
  The sigma floor keeps zero-variance history from producing division by zero or
  absurd scores. Only HIGH readings (z >= threshold) are flagged. Flagged readings
  are not added to the baseline, so a spike does not hide later spikes.

Input (direct invocation):
  {"records": [{"timestamp": "2025-09-01T00:00:00", "building": "...",
                "meter_id": "...", "consumption_kwh": 5.2}, ...],
   "threshold": 3.0, "baseline_days": 10, "min_history": 3}

Output:
  {"statusCode": 200, "body": {"engine", "parameters", "summary", "anomalies": [...]}}
  or {"statusCode": 400, "body": {"error": "..."}} for invalid input.

The Streamlit app calls lambda_handler() directly when AWS is not configured, so
local (demo) and cloud results come from the same code.
"""
import json
import math
import statistics
from collections import defaultdict, deque
from datetime import datetime

ENGINE = "powerwatch-zscore-v1"
DEFAULT_THRESHOLD = 3.0
DEFAULT_BASELINE_DAYS = 10
DEFAULT_MIN_HISTORY = 3
REL_SIGMA_FLOOR = 0.05
ABS_SIGMA_FLOOR = 0.01


def _parse_timestamp(value):
    text = str(value).strip().replace("Z", "+00:00")
    return datetime.fromisoformat(text).replace(tzinfo=None)


def severity_for(z, threshold):
    ratio = z / threshold
    if ratio >= 2.5:
        return "Critical"
    if ratio >= 1.5:
        return "High"
    return "Moderate"


def detect_anomalies(records, threshold=DEFAULT_THRESHOLD,
                     baseline_days=DEFAULT_BASELINE_DAYS,
                     min_history=DEFAULT_MIN_HISTORY):
    """Return (anomalies, summary) for a list of reading dicts."""
    valid, skipped = [], 0
    for rec in records:
        try:
            ts = _parse_timestamp(rec["timestamp"])
            value = float(rec["consumption_kwh"])
        except (KeyError, TypeError, ValueError, AttributeError):
            skipped += 1
            continue
        if not math.isfinite(value) or value < 0:
            skipped += 1
            continue
        building = str(rec.get("building") or "Unspecified")
        meter = str(rec.get("meter_id") or building)
        valid.append((ts, meter, building, value))
    valid.sort(key=lambda r: (r[0], r[1]))

    history = defaultdict(lambda: deque(maxlen=baseline_days))
    anomalies, evaluated, insufficient = [], 0, 0
    for ts, meter, building, value in valid:
        weekend = ts.weekday() >= 5
        hist = history[(meter, ts.hour * 60 + ts.minute, weekend)]
        if len(hist) < min_history:
            insufficient += 1
            hist.append(value)
            continue
        evaluated += 1
        mean = statistics.fmean(hist)
        std = statistics.stdev(hist)
        sigma = max(std, REL_SIGMA_FLOOR * abs(mean), ABS_SIGMA_FLOOR)
        z = (value - mean) / sigma
        if z >= threshold:
            deviation = value - mean
            pct = (deviation / mean * 100.0) if mean > 0 else None
            daytype = "weekend" if weekend else "weekday"
            pct_txt = f"{pct:+.0f}% vs baseline, " if pct is not None else ""
            anomalies.append({
                "timestamp": ts.isoformat(),
                "building": building,
                "meter_id": meter,
                "observed_kwh": round(value, 3),
                "baseline_kwh": round(mean, 3),
                "std_kwh": round(std, 3),
                "deviation_kwh": round(deviation, 3),
                "deviation_pct": None if pct is None else round(pct, 1),
                "z_score": round(z, 2),
                "severity": severity_for(z, threshold),
                "explanation": (
                    f"Observed {value:.2f} kWh against a typical {mean:.2f} kWh for this "
                    f"time of day on {daytype}s (mean of the previous {len(hist)} "
                    f"comparable readings). {pct_txt}z-score {z:.1f} meets the "
                    f"threshold of {threshold:g}. This is unusual relative to history; "
                    f"it does not by itself identify the cause."
                ),
            })
        else:
            hist.append(value)

    summary = {
        "records_received": len(records),
        "records_valid": len(valid),
        "records_skipped_invalid": skipped,
        "records_evaluated": evaluated,
        "records_insufficient_history": insufficient,
        "anomalies_detected": len(anomalies),
    }
    return anomalies, summary


def _read_parameters(event):
    threshold = float(event.get("threshold", DEFAULT_THRESHOLD))
    baseline_days = int(event.get("baseline_days", DEFAULT_BASELINE_DAYS))
    min_history = int(event.get("min_history", min(DEFAULT_MIN_HISTORY, baseline_days)))
    if not 0.5 <= threshold <= 20:
        raise ValueError("threshold must be between 0.5 and 20")
    if not 2 <= baseline_days <= 60:
        raise ValueError("baseline_days must be between 2 and 60")
    if not 2 <= min_history <= baseline_days:
        raise ValueError("min_history must be between 2 and baseline_days")
    return threshold, baseline_days, min_history


def lambda_handler(event, context):
    try:
        if not isinstance(event, dict):
            raise ValueError("event must be a JSON object")
        if isinstance(event.get("body"), str):  # tolerate API-Gateway style wrapping
            event = {**event, **json.loads(event["body"])}
        records = event.get("records")
        if not isinstance(records, list):
            raise ValueError("'records' must be a list of reading objects")
        threshold, baseline_days, min_history = _read_parameters(event)
    except (ValueError, TypeError) as exc:
        return {"statusCode": 400, "body": {"error": str(exc)}}

    anomalies, summary = detect_anomalies(records, threshold, baseline_days, min_history)
    return {
        "statusCode": 200,
        "body": {
            "engine": ENGINE,
            "parameters": {"threshold": threshold, "baseline_days": baseline_days,
                           "min_history": min_history},
            "summary": summary,
            "anomalies": anomalies,
        },
    }
