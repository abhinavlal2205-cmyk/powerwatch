"""PowerWatch core logic. No Streamlit imports, so everything here is testable."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import lambda_function as lf

REQUIRED_COLUMNS = ["timestamp", "consumption_kwh"]
TEMPLATE_COLUMNS = ["timestamp", "building", "meter_id", "consumption_kwh", "temperature_c"]
SAMPLE_START = "2025-09-01"  # a Monday

# Deliberate spikes in the synthetic dataset: (meter, start, hours, multiplier, note)
SAMPLE_SPIKES = [
    ("ACD-01", "2025-09-18 01:00", 3, 6.0, "Academic block overnight load"),
    ("HST-01", "2025-09-23 13:00", 3, 2.3, "Hostel afternoon surge"),
    ("LIB-01", "2025-09-25 03:00", 3, 4.0, "Library overnight load"),
    ("HST-01", "2025-09-27 10:00", 3, 2.5, "Hostel weekend morning surge"),
    ("ACD-01", "2025-09-29 11:00", 2, 1.9, "Academic block daytime surge"),
]

CONFIG_KEYS = [
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_REGION",
    "S3_BUCKET_NAME", "LAMBDA_FUNCTION_NAME", "QUICKSIGHT_DASHBOARD_ID",
    "QUICKSIGHT_ACCOUNT_ID", "QUICKSIGHT_USER_ARN",
]


@dataclass
class OpResult:
    ok: bool
    message: str
    data: Any = None


# --------------------------------------------------------------------------- data
def _window(h, start, end, k=1.5):
    return 1 / (1 + np.exp(-(h - start) * k)) * 1 / (1 + np.exp((h - end) * k))


def generate_sample_data(days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Hourly synthetic readings for three buildings with daily/weekly patterns and spikes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(SAMPLE_START, periods=days * 24, freq="h")
    hour = idx.hour.values.astype(float)
    weekend = idx.dayofweek.values >= 5
    n = len(idx)
    temp = (28 + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
            + 2 * np.sin(2 * np.pi * np.arange(n) / (24 * 7)) + rng.normal(0, 0.6, n))
    profiles = {
        "Hostel Block A": ("HST-01", 8 + 14 * np.exp(-((hour - 20.5) / 2.5) ** 2)
                           + 6 * np.exp(-((hour - 7) / 1.5) ** 2)
                           + np.where(weekend, 4 * np.exp(-((hour - 13) / 3) ** 2), 0)),
        "Academic Block": ("ACD-01", 4 + np.where(weekend, 3, 26) * _window(hour, 8.5, 17.5)),
        "Central Library": ("LIB-01", 5 + np.where(weekend, 7, 12) * _window(hour, 9, 20.5)),
    }
    frames = []
    for building, (meter, base) in profiles.items():
        kwh = base * (1 + 0.015 * (temp - 28)) * rng.normal(1, 0.04, n)
        for m, start, hours, mult, _ in SAMPLE_SPIKES:
            if m == meter:
                s = pd.Timestamp(start)
                mask = (idx >= s) & (idx < s + pd.Timedelta(hours=hours))
                kwh = np.where(mask, kwh * mult, kwh)
        frames.append(pd.DataFrame({
            "timestamp": idx, "building": building, "meter_id": meter,
            "consumption_kwh": np.round(kwh, 3), "temperature_c": np.round(temp, 1),
        }))
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "meter_id"]).reset_index(drop=True)


def template_csv() -> str:
    rows = [
        "2025-09-01 00:00:00,Hostel Block A,HST-01,8.20,26.4",
        "2025-09-01 01:00:00,Hostel Block A,HST-01,7.90,26.1",
        "2025-09-01 02:00:00,Hostel Block A,HST-01,7.80,25.8",
    ]
    return ",".join(TEMPLATE_COLUMNS) + "\n" + "\n".join(rows) + "\n"


def validate_dataframe(raw: pd.DataFrame):
    """Validate an uploaded frame. Returns (clean_df or None, report).

    Invalid rows are removed and counted; nothing is imputed or invented.
    """
    report = {"errors": [], "warnings": [], "rows_in": len(raw), "rows_kept": 0, "dropped": {}}
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        report["errors"].append(f"Missing required column(s): {', '.join(missing)}. "
                                f"Required: {', '.join(REQUIRED_COLUMNS)}.")
        return None, report

    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    except (ValueError, TypeError):
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_localize(None)
        report["warnings"].append("Timestamps had time-zone information; it was removed and local wall-clock time is used.")

    orig = df["consumption_kwh"]
    num = pd.to_numeric(orig, errors="coerce")
    blank = orig.isna() | (orig.astype(str).str.strip() == "")
    non_numeric = ~blank & num.isna()
    infinite = np.isinf(num.fillna(0))
    negative = num < 0
    bad_ts = ts.isna()
    drop = bad_ts | blank | non_numeric | negative | infinite
    for label, mask in [("invalid or missing timestamp", bad_ts), ("missing consumption value", blank),
                        ("non-numeric consumption value", non_numeric), ("negative consumption", negative),
                        ("infinite consumption", infinite)]:
        if int(mask.sum()):
            report["dropped"][label] = int(mask.sum())

    out = pd.DataFrame({"timestamp": ts, "consumption_kwh": num})
    if "building" in df.columns:
        b = df["building"].astype(str).str.strip().replace({"": "Unspecified", "nan": "Unspecified"})
    else:
        b = pd.Series("Unspecified", index=df.index)
        report["warnings"].append("Optional column 'building' not found; all rows are grouped as 'Unspecified'.")
    if "meter_id" in df.columns:
        m = df["meter_id"].astype(str).str.strip()
        m = m.where(~m.isin(["", "nan"]), b)
    else:
        m = b.copy()
        report["warnings"].append("Optional column 'meter_id' not found; the building name is used as the meter ID.")
    out["building"], out["meter_id"] = b, m
    if "temperature_c" in df.columns:
        out["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
        bad_t = int((out["temperature_c"].isna() & df["temperature_c"].astype(str).str.strip().ne("")).sum())
        if bad_t:
            report["warnings"].append(f"{bad_t} temperature value(s) were not numeric and were left empty.")
    else:
        out["temperature_c"] = np.nan
        report["warnings"].append("Optional column 'temperature_c' not found; temperature analysis is skipped.")

    out = out[~drop]
    dup = out.duplicated(["meter_id", "timestamp"], keep="first")
    if int(dup.sum()):
        report["dropped"]["duplicate meter/timestamp"] = int(dup.sum())
        out = out[~dup]
    if report["dropped"]:
        report["warnings"].append("Rows with the problems listed below were removed, not repaired.")
    if out.empty:
        report["errors"].append("No valid rows remain after validation.")
        return None, report
    out = out.sort_values(["timestamp", "meter_id"]).reset_index(drop=True)[TEMPLATE_COLUMNS]
    report["rows_kept"] = len(out)
    return out, report


# ------------------------------------------------------------------ anomaly engine
def build_payload(df: pd.DataFrame, threshold: float, baseline_days: int) -> dict:
    recs = df[["timestamp", "building", "meter_id", "consumption_kwh"]].copy()
    recs["timestamp"] = recs["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return {"records": recs.to_dict("records"), "threshold": float(threshold),
            "baseline_days": int(baseline_days)}


def run_local(payload: dict) -> dict:
    """Local fallback: runs the exact Lambda handler in-process."""
    return lf.lambda_handler(payload, None)


ANOMALY_COLUMNS = ["timestamp", "building", "meter_id", "observed_kwh", "baseline_kwh", "std_kwh",
                   "deviation_kwh", "deviation_pct", "z_score", "severity", "explanation"]


def response_to_frame(resp: dict):
    """Validate a handler response and return (anomaly DataFrame, summary dict)."""
    if not isinstance(resp, dict) or resp.get("statusCode") != 200:
        err = resp.get("body", {}).get("error") if isinstance(resp, dict) and isinstance(resp.get("body"), dict) else resp
        raise ValueError(f"Detection engine returned an error: {err}")
    body = resp["body"]
    frame = pd.DataFrame(body["anomalies"], columns=ANOMALY_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame, body["summary"]


# ------------------------------------------------------------------------ analytics
def filter_data(df, building=None, meter=None, start=None, end=None):
    out = df
    if building and building != "All buildings":
        out = out[out["building"] == building]
    if meter and meter != "All meters":
        out = out[out["meter_id"] == meter]
    if start is not None:
        out = out[out["timestamp"].dt.date >= start]
    if end is not None:
        out = out[out["timestamp"].dt.date <= end]
    return out


def compute_kpis(scope_df, start, end, tariff, n_anomalies):
    """scope_df is filtered by building/meter only, so the previous period can be found."""
    d = scope_df["timestamp"].dt.date
    cur = scope_df[(d >= start) & (d <= end)]
    total = float(cur["consumption_kwh"].sum())
    days = int(cur["timestamp"].dt.normalize().nunique())
    avg_daily = total / days if days else 0.0
    length = (end - start).days + 1
    p_end = start - pd.Timedelta(days=1)
    p_start = start - pd.Timedelta(days=length)
    prev = scope_df[(d >= p_start) & (d <= p_end)]
    prev_days = int(prev["timestamp"].dt.normalize().nunique())
    pct = None
    if prev_days >= max(1, int(0.8 * days)) and prev["consumption_kwh"].sum() > 0:
        prev_avg = prev["consumption_kwh"].sum() / prev_days
        pct = (avg_daily - prev_avg) / prev_avg * 100
    return {"total_kwh": total, "avg_daily_kwh": avg_daily, "cost": total * tariff,
            "anomalies": n_anomalies, "pct_change": pct, "days": days, "period_days": length}


def aggregate_total(df, freq="h"):
    return df.groupby(pd.Grouper(key="timestamp", freq=freq))["consumption_kwh"].sum().reset_index()


def hourly_profile(df):
    tot = df.groupby("timestamp")["consumption_kwh"].sum()
    return tot.groupby(tot.index.hour).mean().reindex(range(24)).fillna(0)


def dow_hour_matrix(df):
    tot = df.groupby("timestamp")["consumption_kwh"].sum()
    mat = tot.groupby([tot.index.dayofweek, tot.index.hour]).mean().unstack().reindex(range(7)).reindex(columns=range(24))
    mat.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return mat


def dataset_summary(df, anoms, tariff):
    if df.empty:
        return "No data in the selected range."
    total = df["consumption_kwh"].sum()
    prof = hourly_profile(df)
    peak_h = int(prof.idxmax())
    daily = df.groupby(df["timestamp"].dt.date)["consumption_kwh"].sum()
    top_day = daily.idxmax()
    s = (f"{len(df):,} readings from {df['meter_id'].nunique()} meter(s) in {df['building'].nunique()} "
         f"building(s) between {df['timestamp'].min():%d %b %Y} and {df['timestamp'].max():%d %b %Y} "
         f"total {total:,.0f} kWh (estimated cost ₹{total * tariff:,.0f} at ₹{tariff:g}/kWh). "
         f"Demand usually peaks around {peak_h:02d}:00, and the highest-use day was {top_day:%d %b %Y} "
         f"({daily.max():,.0f} kWh). ")
    if len(anoms):
        worst = anoms.sort_values("z_score", ascending=False).iloc[0]
        s += (f"{len(anoms)} reading(s) were flagged as unusually high; the largest was on "
              f"{worst['meter_id']} at {worst['timestamp']:%d %b %H:%M} ({worst['observed_kwh']:.1f} kWh "
              f"against a baseline of {worst['baseline_kwh']:.1f} kWh).")
    else:
        s += "No unusually high readings were flagged at the current sensitivity."
    return s


def build_recommendations(df, anoms, tariff, reduction_pct):
    """Rule-based recommendations. Savings are scenario estimates under stated assumptions."""
    recs = []
    if df.empty:
        return recs
    frac = reduction_pct / 100.0
    hour = df["timestamp"].dt.hour
    night = df[hour < 6]
    day = df[(hour >= 9) & (hour < 17)]
    n_days = max(1, df["timestamp"].dt.normalize().nunique())

    # 1. overnight consumption
    if len(night) and len(day):
        rows = []
        for b, g in df.groupby("building"):
            gh = g["timestamp"].dt.hour
            n_mean = g.loc[gh < 6, "consumption_kwh"].mean()
            d_mean = g.loc[(gh >= 9) & (gh < 17), "consumption_kwh"].mean()
            if d_mean and d_mean > 0:
                rows.append((b, n_mean / d_mean, g.loc[gh < 6, "consumption_kwh"].sum()))
        if rows:
            b, ratio, kwh = max(rows, key=lambda r: r[1])
            if ratio >= 0.5:
                recs.append({
                    "priority": "High" if ratio >= 0.75 else "Medium",
                    "title": "Investigate overnight consumption",
                    "finding": f"{b} uses about {ratio * 100:.0f}% as much power between 00:00 and 06:00 as it does "
                               f"between 09:00 and 17:00. Overnight use totals {kwh:,.0f} kWh in this period.",
                    "action": "Walk through the building after hours and check for equipment left on, "
                              "air-conditioning schedules and standby loads. Some buildings (hostels, labs) "
                              "legitimately run at night.",
                    "saving_kwh": kwh * frac,
                    "assumption": f"Assumes {reduction_pct}% of overnight energy in the selected range is avoidable.",
                })
    # 2. repeated peak hours
    prof = hourly_profile(df)
    top3 = prof.nlargest(3).index.tolist()
    share = prof[top3].sum() / prof.sum() if prof.sum() else 0
    if share >= 0.20:
        recs.append({
            "priority": "Medium", "title": "Review repeated peak-hour usage",
            "finding": f"The three busiest hours ({', '.join(f'{h:02d}:00' for h in sorted(top3))}) account for "
                       f"{share * 100:.0f}% of average daily energy (an even spread would be 12.5%).",
            "action": "Stagger flexible loads such as water heaters, pumps and laundry outside these hours.",
            "saving_kwh": None,
            "assumption": "Under a flat tariff, shifting load does not reduce kWh cost. Any financial benefit "
                          "depends on time-of-day pricing or demand charges from your utility.",
        })
    # 3. building comparison
    if df["building"].nunique() > 1:
        per = df.groupby("building")["consumption_kwh"].sum().sort_values(ascending=False)
        top, share_b = per.index[0], per.iloc[0] / per.sum() * 100
        recs.append({
            "priority": "Low", "title": "Compare consumption across buildings",
            "finding": f"{top} consumed the most ({per.iloc[0]:,.0f} kWh, {share_b:.0f}% of the selected total); "
                       f"{per.index[-1]} consumed the least ({per.iloc[-1]:,.0f} kWh).",
            "action": "Normalise by floor area or occupancy before drawing conclusions, then investigate "
                      "the building with the highest energy per square metre or per person.",
            "saving_kwh": None,
            "assumption": "The dataset has no floor-area or occupancy data, so the buildings are compared on absolute kWh only.",
        })
    # 4. persistent increase
    daily = df.groupby(df["timestamp"].dt.normalize())["consumption_kwh"].sum()
    if len(daily) >= 14:
        last, prior = daily.iloc[-7:].mean(), daily.iloc[-14:-7].mean()
        change = (last - prior) / prior * 100 if prior else 0
        if change >= 10:
            recs.append({
                "priority": "Medium", "title": "Check for a persistent increase in daily usage",
                "finding": f"Average daily use over the last 7 days ({last:,.0f} kWh) is {change:.0f}% higher "
                           f"than the 7 days before ({prior:,.0f} kWh).",
                "action": "Compare against weather, occupancy and new equipment. If none explain it, inspect "
                          "high-load equipment and schedules.",
                "saving_kwh": None,
                "assumption": "Seasonal or occupancy changes can raise consumption without any waste.",
            })
    # 5. flagged excess
    if len(anoms):
        excess = float(anoms["deviation_kwh"].sum())
        top_meter = anoms["meter_id"].value_counts()
        recs.append({
            "priority": "High" if len(anoms) >= 5 else "Medium",
            "title": "Follow up on flagged readings",
            "finding": f"{len(anoms)} reading(s) exceeded their baseline by {excess:,.0f} kWh in total "
                       f"(about ₹{excess * tariff:,.0f} at ₹{tariff:g}/kWh). Most flags: {top_meter.index[0]} "
                       f"({int(top_meter.iloc[0])}).",
            "action": "Check the Anomaly Detection page for timestamps and look for scheduled events, "
                      "maintenance, or equipment left running at those times.",
            "saving_kwh": None,
            "assumption": "This is the measured excess over a statistical baseline, not a confirmed waste or a guaranteed saving.",
        })
    if not recs:
        recs.append({"priority": "Info", "title": "No major patterns found",
                     "finding": "None of the rule-based checks found a notable pattern in the selected data.",
                     "action": "Continue monitoring. Upload more history for stronger baselines.",
                     "saving_kwh": None, "assumption": ""})
    return recs


# ------------------------------------------------------------------------------ AWS
def load_config(secrets: dict | None = None, environ: dict | None = None) -> dict:
    import os
    env = os.environ if environ is None else environ
    secrets = secrets or {}
    cfg = {}
    for k in CONFIG_KEYS:
        v = secrets.get(k) or env.get(k) or ""
        cfg[k] = str(v).strip()
    return cfg


def aws_readiness(cfg: dict) -> dict:
    creds = bool(cfg.get("AWS_ACCESS_KEY_ID") and cfg.get("AWS_SECRET_ACCESS_KEY"))
    region = bool(cfg.get("AWS_REGION"))
    s3 = creds and region and bool(cfg.get("S3_BUCKET_NAME"))
    fn = creds and region and bool(cfg.get("LAMBDA_FUNCTION_NAME"))
    qs = creds and region and bool(cfg.get("QUICKSIGHT_DASHBOARD_ID") and cfg.get("QUICKSIGHT_ACCOUNT_ID")
                                    and cfg.get("QUICKSIGHT_USER_ARN"))
    return {"credentials": creds, "region": region, "s3": s3, "lambda": fn,
            "quicksight": qs, "configured": s3 or fn}


def _client(service, cfg, read_timeout=60):
    import boto3
    from botocore.config import Config
    session = boto3.Session(
        aws_access_key_id=cfg.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=cfg.get("AWS_SECRET_ACCESS_KEY") or None,
        aws_session_token=cfg.get("AWS_SESSION_TOKEN") or None,
        region_name=cfg.get("AWS_REGION") or None)
    return session.client(service, config=Config(connect_timeout=5, read_timeout=read_timeout,
                                                 retries={"max_attempts": 2}))


def _explain_error(exc, what):
    from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError, BotoCoreError
    if isinstance(exc, NoCredentialsError):
        return f"{what} failed: no AWS credentials were found."
    if isinstance(exc, EndpointConnectionError):
        return f"{what} failed: could not reach the AWS endpoint (check AWS_REGION and network)."
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        hints = {
            "NoSuchBucket": "the bucket does not exist (check S3_BUCKET_NAME and region)",
            "AccessDenied": "the IAM identity lacks the required permission",
            "InvalidAccessKeyId": "the access key ID is not valid",
            "SignatureDoesNotMatch": "the secret access key is wrong",
            "ExpiredToken": "the session token has expired",
            "ResourceNotFoundException": "the Lambda function was not found in this region",
            "AccessDeniedException": "the IAM identity lacks the required permission",
            "404": "the resource was not found", "403": "access was denied",
        }
        return f"{what} failed ({code}): {hints.get(code, exc.response.get('Error', {}).get('Message', 'see AWS error'))}."
    if isinstance(exc, BotoCoreError):
        return f"{what} failed: {exc}"
    return f"{what} failed: {type(exc).__name__}: {exc}"


def upload_to_s3(data: bytes, filename: str, cfg: dict) -> OpResult:
    r = aws_readiness(cfg)
    if not r["s3"]:
        return OpResult(False, "S3 upload skipped: AWS credentials, AWS_REGION or S3_BUCKET_NAME are not configured.")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "upload.csv"
    now = datetime.now(timezone.utc)
    key = f"uploads/{now:%Y/%m/%d}/{now:%H%M%S}_{safe}"
    try:
        resp = _client("s3", cfg).put_object(Bucket=cfg["S3_BUCKET_NAME"], Key=key, Body=data,
                                             ContentType="text/csv", ServerSideEncryption="AES256")
        if resp.get("ResponseMetadata", {}).get("HTTPStatusCode") != 200:
            return OpResult(False, "S3 did not confirm the upload (unexpected HTTP status).")
        return OpResult(True, f"Stored s3://{cfg['S3_BUCKET_NAME']}/{key} ({len(data):,} bytes).", {"key": key})
    except Exception as exc:  # noqa: BLE001 - mapped to a readable message
        return OpResult(False, _explain_error(exc, "S3 upload"))


def invoke_lambda(payload: dict, cfg: dict) -> OpResult:
    r = aws_readiness(cfg)
    if not r["lambda"]:
        return OpResult(False, "Lambda invocation skipped: AWS credentials, AWS_REGION or LAMBDA_FUNCTION_NAME are not configured.")
    body = json.dumps(payload).encode()
    if len(body) > 5_500_000:
        return OpResult(False, f"Payload is {len(body) / 1e6:.1f} MB; synchronous Lambda invocation allows 6 MB.")
    try:
        resp = _client("lambda", cfg).invoke(FunctionName=cfg["LAMBDA_FUNCTION_NAME"],
                                             InvocationType="RequestResponse", Payload=body)
        raw = resp["Payload"].read()
        if resp.get("FunctionError"):
            return OpResult(False, f"Lambda reported a function error: {raw[:300].decode(errors='replace')}")
        out = json.loads(raw)
        if not isinstance(out, dict) or out.get("statusCode") != 200 or "anomalies" not in out.get("body", {}):
            return OpResult(False, f"Lambda returned an unexpected response: {str(out)[:300]}")
        return OpResult(True, f"Lambda '{cfg['LAMBDA_FUNCTION_NAME']}' processed {len(payload['records']):,} records.", out)
    except Exception as exc:  # noqa: BLE001
        return OpResult(False, _explain_error(exc, "Lambda invocation"))


def test_connection(cfg: dict) -> list[tuple[str, bool, str]]:
    """Real, read-only checks. Returns (check name, ok, message)."""
    r = aws_readiness(cfg)
    if not (r["credentials"] and r["region"]):
        return [("Credentials and region", False, "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY or AWS_REGION is not set.")]
    out = []
    try:
        ident = _client("sts", cfg).get_caller_identity()
        out.append(("Credentials (STS)", True, f"Authenticated as {ident['Arn']}"))
    except Exception as exc:  # noqa: BLE001
        return [("Credentials (STS)", False, _explain_error(exc, "Credential check"))]
    if cfg.get("S3_BUCKET_NAME"):
        try:
            _client("s3", cfg).head_bucket(Bucket=cfg["S3_BUCKET_NAME"])
            out.append(("S3 bucket", True, f"Bucket '{cfg['S3_BUCKET_NAME']}' is reachable."))
        except Exception as exc:  # noqa: BLE001
            out.append(("S3 bucket", False, _explain_error(exc, "Bucket check")))
    if cfg.get("LAMBDA_FUNCTION_NAME"):
        try:
            c = _client("lambda", cfg).get_function_configuration(FunctionName=cfg["LAMBDA_FUNCTION_NAME"])
            out.append(("Lambda function", True, f"Found '{cfg['LAMBDA_FUNCTION_NAME']}' ({c.get('Runtime')}, state {c.get('State', 'n/a')})."))
        except Exception as exc:  # noqa: BLE001
            out.append(("Lambda function", False, _explain_error(exc, "Function check")))
    return out


def quicksight_embed_url(cfg: dict) -> OpResult:
    if not aws_readiness(cfg)["quicksight"]:
        return OpResult(False, "QuickSight is not configured (needs account ID, dashboard ID and registered user ARN).")
    try:
        resp = _client("quicksight", cfg).generate_embed_url_for_registered_user(
            AwsAccountId=cfg["QUICKSIGHT_ACCOUNT_ID"], UserArn=cfg["QUICKSIGHT_USER_ARN"],
            SessionLifetimeInMinutes=60,
            ExperienceConfiguration={"Dashboard": {"InitialDashboardId": cfg["QUICKSIGHT_DASHBOARD_ID"]}})
        return OpResult(True, "Embed URL generated (valid for a short time).", resp["EmbedUrl"])
    except Exception as exc:  # noqa: BLE001
        return OpResult(False, _explain_error(exc, "QuickSight embed URL"))
