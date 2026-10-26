"""PowerWatch - Cloud-Based Electricity Consumption Monitoring and Anomaly Detection."""
import io
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core

NAVY, GREEN, GREY, RED, LIGHT = "#0B1F3A", "#1F9D55", "#6B7A90", "#C0392B", "#E6EBF2"
PAGES = ["Overview", "Consumption Analytics", "Anomaly Detection", "Recommendations",
         "Data Upload", "AWS Architecture and System Status", "About the Project"]

st.set_page_config(page_title="PowerWatch", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
<style>
html, body, [class*="css"], .stApp {{ color: {NAVY}; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ background: #F5F8FB; border-right: 1px solid {LIGHT}; }}
.pw-brand {{ font-size: 1.6rem; font-weight: 700; color: {NAVY}; margin-bottom: 0; }}
.pw-brand span {{ color: {GREEN}; }}
.pw-tag {{ color: {GREY}; font-size: 0.85rem; margin-bottom: 1rem; }}
.pw-card {{ background: #fff; border: 1px solid {LIGHT}; border-radius: 10px; padding: 14px 16px;
           border-top: 3px solid {GREEN}; height: 100%; }}
.pw-label {{ color: {GREY}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; }}
.pw-value {{ font-size: 1.7rem; font-weight: 700; color: {NAVY}; line-height: 1.3; }}
.pw-sub {{ color: {GREY}; font-size: 0.8rem; }}
.pw-pill {{ display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
.pw-demo {{ background: #FFF4DB; color: #7A5200; border: 1px solid #F0D58C; }}
.pw-aws {{ background: #E3F5EA; color: #11643A; border: 1px solid #A8DDBE; }}
.pw-note {{ background: #F5F8FB; border-left: 3px solid {GREEN}; padding: 10px 14px; border-radius: 4px; margin: 8px 0; }}
.pw-footer {{ margin-top: 3rem; padding-top: 1rem; border-top: 1px solid {LIGHT}; color: {GREY}; font-size: 0.8rem; }}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- state
@st.cache_data(show_spinner=False)
def sample_df():
    return core.generate_sample_data()


def load_sample():
    st.session_state.update(raw_df=sample_df(), source_name="Built-in synthetic dataset (30 days, 3 buildings)",
                            dataset_id=st.session_state.get("dataset_id", 0) + 1)


def start_demo():
    load_sample()
    st.session_state["page"] = "Anomaly Detection"
    st.session_state["threshold"] = 3.0


def use_dataset(df, name):
    st.session_state.update(raw_df=df, source_name=name, dataset_id=st.session_state.get("dataset_id", 0) + 1)


def read_secrets():
    try:
        return {k: str(v) for k, v in st.secrets.items() if not isinstance(v, dict)}
    except Exception:  # no secrets file configured
        return {}


def log_op(operation, res):
    st.session_state.setdefault("aws_log", []).insert(0, {
        "Time": datetime.now().strftime("%H:%M:%S"), "Operation": operation,
        "Result": "Success" if res.ok else "Failed / skipped", "Detail": res.message})
    del st.session_state["aws_log"][50:]


if "raw_df" not in st.session_state:
    load_sample()
if "page" not in st.session_state:
    st.session_state["page"] = "Overview"

cfg = core.load_config(read_secrets())
ready = core.aws_readiness(cfg)


# ---------------------------------------------------------------------- analysis
def get_analysis(df, threshold, baseline_days, use_aws):
    """Run detection (Lambda if enabled, otherwise local) and cache the result for this input."""
    sig = int(pd.util.hash_pandas_object(df, index=False).sum())
    key = (sig, float(threshold), int(baseline_days), bool(use_aws))
    cache = st.session_state.setdefault("analysis_cache", {})
    if key in cache:
        return cache[key]
    payload = core.build_payload(df, threshold, baseline_days)
    resp, source, note = None, "Local Python (demo mode)", None
    if use_aws:
        res = core.invoke_lambda(payload, cfg)
        log_op("Lambda invoke", res)
        if res.ok:
            resp, source = res.data, f"AWS Lambda ({cfg['LAMBDA_FUNCTION_NAME']})"
        else:
            note = f"{res.message} The local detection engine was used instead."
    if resp is None:
        resp = core.run_local(payload)
        if use_aws and note is None:
            note = "Lambda is not configured; the local detection engine was used."
    frame, summary = core.response_to_frame(resp)
    if len(cache) > 20:
        cache.clear()
    cache[key] = {"anoms": frame, "summary": summary, "source": source, "note": note}
    return cache[key]


# ---------------------------------------------------------------------- ui helpers
def style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#fff",
                      plot_bgcolor="#fff", font=dict(color=NAVY, family="Inter, Segoe UI, sans-serif"),
                      legend=dict(orientation="h", y=-0.2), title_font=dict(size=15))
    fig.update_xaxes(gridcolor=LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor=LIGHT, zeroline=False)
    return fig


def card(col, label, value, sub=""):
    col.markdown(f'<div class="pw-card"><div class="pw-label">{label}</div><div class="pw-value">{value}</div>'
                 f'<div class="pw-sub">{sub}</div></div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="pw-footer">PowerWatch is a student cloud computing project (B.Tech). '
                'Costs and savings shown are estimates, not utility bills. Detected anomalies are statistical '
                'flags, not proof of equipment failure or theft.</div>', unsafe_allow_html=True)


def csv_bytes(df):
    return df.to_csv(index=False).encode()


# ---------------------------------------------------------------------- sidebar
raw = st.session_state["raw_df"]
ds = st.session_state["dataset_id"]
with st.sidebar:
    st.markdown('<div class="pw-brand">Power<span>Watch</span></div>'
                '<div class="pw-tag">Detect unusual energy usage. Make smarter energy decisions.</div>',
                unsafe_allow_html=True)
    st.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
    st.divider()

    use_aws = False
    if ready["configured"]:
        use_aws = st.toggle("Use AWS services", value=True, key="use_aws")
    if use_aws:
        st.markdown('<span class="pw-pill pw-aws">AWS-CONNECTED MODE</span>', unsafe_allow_html=True)
        st.caption("AWS is configured. Each operation reports its own result; failures fall back to local processing.")
    else:
        st.markdown('<span class="pw-pill pw-demo">DEMO MODE</span>', unsafe_allow_html=True)
        st.caption("Processing runs locally. No data is sent to AWS." if not ready["configured"]
                   else "AWS is configured but switched off.")
    st.button("Try a demo scenario", on_click=start_demo, help="Loads the sample data with known spikes.")
    st.divider()

    st.caption(f"Data source: {st.session_state['source_name']}")
    buildings = ["All buildings"] + sorted(raw["building"].unique())
    building = st.selectbox("Building", buildings, key=f"b_{ds}")
    scope_b = raw if building == "All buildings" else raw[raw["building"] == building]
    meter = st.selectbox("Meter", ["All meters"] + sorted(scope_b["meter_id"].unique()), key=f"m_{ds}_{building}")
    dmin, dmax = raw["timestamp"].dt.date.min(), raw["timestamp"].dt.date.max()
    dates = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax, key=f"d_{ds}")
    tariff = st.number_input("Tariff (₹ per kWh)", min_value=0.0, value=8.0, step=0.5, key="tariff")
    st.session_state.setdefault("threshold", 3.0)
    threshold = st.slider("Detection sensitivity (Z-score threshold)", 2.0, 6.0, step=0.5, key="threshold",
                          help="Lower values flag more readings. Higher values flag only large deviations.")
    baseline_days = st.slider("Baseline length (days)", 5, 20, 10, key="baseline_days",
                              help="Previous same-type days used to build the baseline for each hour.")

if not isinstance(dates, (list, tuple)) or len(dates) != 2:
    st.info("Select both a start and an end date in the sidebar.")
    st.stop()
start, end = dates

scope = core.filter_data(raw, building, meter)
view = core.filter_data(scope, start=start, end=end)
analysis = get_analysis(raw, threshold, baseline_days, use_aws)  # full data keeps baselines intact
all_anoms = analysis["anoms"]
a_scope = all_anoms
if building != "All buildings":
    a_scope = a_scope[a_scope["building"] == building]
if meter != "All meters":
    a_scope = a_scope[a_scope["meter_id"] == meter]
anoms = a_scope[(a_scope["timestamp"].dt.date >= start) & (a_scope["timestamp"].dt.date <= end)]
page = st.session_state["page"]

if view.empty:
    st.warning("No readings match the selected building, meter and date range.")
    footer()
    st.stop()


def source_banner():
    if analysis["note"]:
        st.warning(analysis["note"])
    st.caption(f"Detection engine: {analysis['source']}")


# ------------------------------------------------------------------------ pages
def page_overview():
    st.title("Overview")
    source_banner()
    k = core.compute_kpis(scope, start, end, tariff, len(anoms))
    c = st.columns(5)
    card(c[0], "Total consumption", f"{k['total_kwh']:,.0f} kWh", f"{k['days']} day(s) with data")
    card(c[1], "Average daily", f"{k['avg_daily_kwh']:,.0f} kWh", "per day with data")
    card(c[2], "Estimated cost", f"₹{k['cost']:,.0f}", f"estimate at ₹{tariff:g}/kWh")
    card(c[3], "Anomalies detected", f"{k['anomalies']}", f"threshold z ≥ {threshold:g}")
    if k["pct_change"] is None:
        card(c[4], "Change vs previous period", "n/a", "no comparable earlier period in data")
    else:
        card(c[4], "Change vs previous period", f"{k['pct_change']:+.1f}%", f"avg daily, vs prior {k['period_days']} days")
    st.write("")
    st.markdown(f'<div class="pw-note"><b>Dataset summary.</b> {core.dataset_summary(view, anoms, tariff)}</div>',
                unsafe_allow_html=True)

    daily_view = (end - start).days + 1 > 14
    tot = core.aggregate_total(view, "D" if daily_view else "h")
    fig = go.Figure(go.Scatter(x=tot["timestamp"], y=tot["consumption_kwh"], mode="lines", name="Consumption",
                               line=dict(color=GREEN, width=2)))
    if len(anoms):
        mark = anoms.assign(t=anoms["timestamp"].dt.normalize() if daily_view else anoms["timestamp"])
        pts = tot.set_index("timestamp").reindex(mark["t"].unique()).dropna()
        fig.add_trace(go.Scatter(x=pts.index, y=pts["consumption_kwh"], mode="markers", name="Contains anomaly",
                                 marker=dict(color=RED, size=8)))
    fig.update_layout(title=f"Electricity consumption over time ({'daily' if daily_view else 'hourly'} totals, kWh)")
    st.plotly_chart(style(fig))

    c1, c2 = st.columns(2)
    prof = core.hourly_profile(view)
    top3 = set(prof.nlargest(3).index)
    fig = go.Figure(go.Bar(x=[f"{h:02d}" for h in prof.index], y=prof.values,
                           marker_color=[GREEN if h in top3 else "#B8C4D6" for h in prof.index]))
    fig.update_layout(title="Average consumption by hour of day (peak hours in green)",
                      xaxis_title="Hour", yaxis_title="kWh")
    c1.plotly_chart(style(fig, 330))
    if view["building"].nunique() > 1:
        by_b = view.groupby("building")["consumption_kwh"].sum().sort_values().reset_index()
        fig = px.bar(by_b, x="consumption_kwh", y="building", orientation="h", title="Consumption by building (kWh)")
        fig.update_traces(marker_color=NAVY)
        c2.plotly_chart(style(fig, 330))
    else:
        c2.info("Only one building in the selected data, so no building comparison is shown.")

    st.subheader("Recent anomalies")
    if anoms.empty:
        st.write("No anomalies were flagged in the selected range.")
    else:
        recent = anoms.sort_values("timestamp", ascending=False).head(10)
        st.dataframe(recent[["timestamp", "building", "meter_id", "observed_kwh", "baseline_kwh", "z_score", "severity"]],
                     hide_index=True)


def page_analytics():
    st.title("Consumption Analytics")
    t1, t2, t3, t4 = st.tabs(["Trend", "Weekly pattern", "Temperature", "Daily table"])
    with t1:
        res = st.radio("Resolution", ["Hourly", "Daily"], horizontal=True, index=1 if (end - start).days > 14 else 0)
        g = view.groupby([pd.Grouper(key="timestamp", freq="D" if res == "Daily" else "h"), "meter_id"])["consumption_kwh"].sum().reset_index()
        fig = px.line(g, x="timestamp", y="consumption_kwh", color="meter_id",
                      color_discrete_sequence=[GREEN, NAVY, "#7FB5A0", GREY], title=f"{res} consumption by meter (kWh)")
        st.plotly_chart(style(fig))
    with t2:
        m = core.dow_hour_matrix(view)
        fig = px.imshow(m, aspect="auto", color_continuous_scale=["#F2F7F4", GREEN, NAVY],
                        labels=dict(x="Hour of day", y="Day", color="kWh"), title="Average kWh by weekday and hour")
        st.plotly_chart(style(fig, 360))
        st.caption("Darker cells show recurring high-use periods.")
    with t3:
        if view["temperature_c"].notna().any():
            tot = view.groupby("timestamp").agg(kwh=("consumption_kwh", "sum"), temp=("temperature_c", "mean")).dropna()
            fig = px.scatter(tot, x="temp", y="kwh", opacity=0.5, title="Consumption versus temperature",
                             labels={"temp": "Temperature (°C)", "kwh": "kWh per hour"}, color_discrete_sequence=[GREEN])
            st.plotly_chart(style(fig))
            st.caption(f"Correlation: {tot['temp'].corr(tot['kwh']):.2f}. Correlation does not imply causation.")
        else:
            st.info("This dataset has no temperature_c values, so temperature analysis is unavailable.")
    with t4:
        d = view.groupby([view["timestamp"].dt.date.rename("date"), "building"])["consumption_kwh"].sum().round(1).reset_index()
        d["estimated_cost_inr"] = (d["consumption_kwh"] * tariff).round(0)
        st.dataframe(d, hide_index=True)
        st.download_button("Download daily summary (CSV)", csv_bytes(d), "daily_summary.csv", "text/csv")


def page_anomalies():
    st.title("Anomaly Detection")
    source_banner()
    c = st.columns([3, 1])
    c[0].markdown(
        f'<div class="pw-note"><b>How it works.</b> Each reading is compared with the previous {baseline_days} readings '
        f'at the same time of day on the same day type (weekday or weekend). A reading is flagged when it is at least '
        f'{threshold:g} standard deviations above that baseline. Only high usage is flagged. A flag means '
        f'"unusual compared with history", not "equipment failure" or "theft".</div>', unsafe_allow_html=True)
    c[1].button("Load demo scenario", on_click=start_demo, key="demo2")
    s = analysis["summary"]
    m = st.columns(4)
    m[0].metric("Readings analysed", f"{s['records_received']:,}")
    m[1].metric("Evaluated against a baseline", f"{s['records_evaluated']:,}")
    m[2].metric("Too little history", f"{s['records_insufficient_history']:,}",
                help="The first few days of each meter cannot be judged yet.")
    m[3].metric("Flagged in current view", len(anoms))
    if s["records_skipped_invalid"]:
        st.warning(f"{s['records_skipped_invalid']} record(s) were skipped as invalid by the engine.")

    meters = sorted(view["meter_id"].unique())
    sel = st.selectbox("Meter to plot", meters)
    d = view[view["meter_id"] == sel]
    a = anoms[anoms["meter_id"] == sel]
    fig = go.Figure(go.Scatter(x=d["timestamp"], y=d["consumption_kwh"], mode="lines", name="Observed",
                               line=dict(color=GREEN, width=1.5)))
    if len(a):
        fig.add_trace(go.Scatter(x=a["timestamp"], y=a["observed_kwh"], mode="markers", name="Flagged",
                                 marker=dict(color=RED, size=9), customdata=a[["baseline_kwh", "z_score", "severity"]].values,
                                 hovertemplate="%{x}<br>Observed %{y:.1f} kWh<br>Baseline %{customdata[0]:.1f} kWh"
                                               "<br>z = %{customdata[1]:.1f} (%{customdata[2]})<extra></extra>"))
    fig.update_layout(title=f"{sel}: observed consumption and flagged readings", yaxis_title="kWh")
    st.plotly_chart(style(fig))

    st.subheader("Flagged readings")
    if anoms.empty:
        st.success("No readings exceeded the threshold in the current view. Try lowering the sensitivity threshold.")
    else:
        sev = st.multiselect("Severity", ["Moderate", "High", "Critical"], default=["Moderate", "High", "Critical"])
        shown = anoms[anoms["severity"].isin(sev)].sort_values("timestamp")
        st.dataframe(shown, hide_index=True, column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="DD MMM YYYY, HH:mm"),
            "explanation": st.column_config.TextColumn("Why it was flagged", width="large")})
        st.download_button("Download anomaly report (CSV)", csv_bytes(shown), "powerwatch_anomaly_report.csv", "text/csv")
        st.caption("Severity compares the z-score with your threshold: Moderate < 1.5x, High 1.5x to 2.5x, Critical ≥ 2.5x.")


def page_recs():
    st.title("Recommendations")
    st.caption(f"Based on the selected data ({st.session_state['source_name']}). All figures are estimates at ₹{tariff:g}/kWh.")
    red = st.slider("Assumed reduction in targeted usage (%)", 0, 50, 10,
                    help="A scenario assumption used to size possible savings. It is not a measured result.")
    for r in core.build_recommendations(view, anoms, tariff, red):
        with st.container(border=True):
            st.markdown(f"**{r['title']}**  ·  Priority: {r['priority']}")
            st.write(r["finding"])
            st.write(f"**Suggested action:** {r['action']}")
            if r["saving_kwh"] is not None:
                st.write(f"**Estimated potential saving (scenario):** {r['saving_kwh']:,.0f} kWh, about "
                         f"₹{r['saving_kwh'] * tariff:,.0f} over the selected period.")
            if r["assumption"]:
                st.caption(f"Assumption: {r['assumption']}")
    st.caption("Savings are not guaranteed and have not been measured. Verify on site before acting.")


def page_upload():
    st.title("Data Upload")
    c = st.columns(2)
    c[0].download_button("Download CSV template", core.template_csv().encode(), "powerwatch_template.csv", "text/csv")
    c[1].download_button("Download sample dataset", csv_bytes(sample_df()), "sample_meter_data.csv", "text/csv")
    st.markdown("**Required columns:** `timestamp`, `consumption_kwh`. **Optional:** `building`, `meter_id`, `temperature_c`. "
                "Rows with invalid timestamps, missing, non-numeric or negative consumption are removed and reported, never repaired.")
    up = st.file_uploader("Upload a CSV file", type=["csv"])
    if up is not None:
        data = up.getvalue()
        try:
            parsed = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read this file as CSV: {exc}")
            parsed = None
        if parsed is not None:
            clean, rep = core.validate_dataframe(parsed)
            m = st.columns(3)
            m[0].metric("Rows in file", f"{rep['rows_in']:,}")
            m[1].metric("Valid rows", f"{rep['rows_kept']:,}")
            m[2].metric("Rows removed", f"{rep['rows_in'] - rep['rows_kept']:,}")
            for e in rep["errors"]:
                st.error(e)
            for w in rep["warnings"]:
                st.warning(w)
            for reason, n in rep["dropped"].items():
                st.write(f"- {n:,} row(s) removed: {reason}")
            if clean is not None:
                st.subheader("Preview")
                st.dataframe(clean.head(20), hide_index=True)
                b = st.columns(2)
                b[0].button("Use this dataset in the dashboard", type="primary", on_click=use_dataset,
                            args=(clean, f"Uploaded: {up.name}"))
                if b[1].button("Store original file in Amazon S3", disabled=not ready["s3"]):
                    res = core.upload_to_s3(data, up.name, cfg)
                    log_op("S3 upload", res)
                    (st.success if res.ok else st.error)(res.message)
                if not ready["s3"]:
                    st.caption("S3 upload is disabled because AWS credentials, region or bucket are not configured (demo mode).")
    st.divider()
    st.subheader("Current dataset")
    st.write(f"{st.session_state['source_name']}: {len(raw):,} readings.")
    if st.button("Store current dataset in Amazon S3", disabled=not ready["s3"]):
        res = core.upload_to_s3(csv_bytes(raw), "current_dataset.csv", cfg)
        log_op("S3 upload", res)
        (st.success if res.ok else st.error)(res.message)
    st.button("Reset to built-in sample data", on_click=load_sample)


def page_arch():
    st.title("AWS Architecture and System Status")
    st.subheader("System status")
    rows = [("AWS credentials", ready["credentials"]), ("AWS_REGION", ready["region"]),
            ("S3_BUCKET_NAME", bool(cfg["S3_BUCKET_NAME"])), ("LAMBDA_FUNCTION_NAME", bool(cfg["LAMBDA_FUNCTION_NAME"])),
            ("QuickSight dashboard (optional)", ready["quicksight"])]
    st.table(pd.DataFrame({"Setting": [r[0] for r in rows], "Status": ["Configured" if r[1] else "Not configured" for r in rows]}))
    st.write("**Current mode:** " + ("AWS-connected (configured)" if use_aws else "Demo mode"))
    st.caption("This table only shows whether settings exist. Use the test below to verify them against AWS.")
    if st.button("Test AWS connection", disabled=not ready["credentials"]):
        for name, ok, msg in core.test_connection(cfg):
            (st.success if ok else st.error)(f"{name}: {msg}")
    if not ready["credentials"]:
        st.caption("The test is disabled in demo mode because no credentials are configured.")
    st.markdown("**Operation log (this session)**")
    log = st.session_state.get("aws_log")
    st.dataframe(pd.DataFrame(log), hide_index=True) if log else st.caption("No AWS operations have been attempted yet.")

    st.subheader("Architecture")
    st.graphviz_chart('''digraph G { rankdir=LR; bgcolor="transparent"; nodesep=0.5;
      node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#0B1F3A", fillcolor="#FFFFFF", fontcolor="#0B1F3A"];
      edge [color="#0B1F3A", fontname="Helvetica", fontsize=9, fontcolor="#6B7A90"];
      User -> App [label="upload CSV"];
      App [label="Streamlit application\\n(Streamlit Community Cloud)", fillcolor="#E3F5EA"];
      S3 [label="Amazon S3\\nraw CSV storage"]; Lambda [label="AWS Lambda\\nanomaly detection"];
      App -> S3 [label="PutObject (original file)"];
      App -> Lambda [label="Invoke: JSON records"];
      Lambda -> App [label="JSON: anomalies"];
      Results [label="Processed results", fillcolor="#F5F8FB"]; Dash [label="PowerWatch analytics\\ndashboard (Plotly)", fillcolor="#E3F5EA"];
      App -> Results -> Dash;
      QS [label="Amazon QuickSight\\n(optional, not auto-connected)", style="rounded,dashed", fillcolor="#FFFFFF"];
      S3 -> QS [style=dashed, label="optional dataset"]; QS -> Dash [style=dashed, label="optional embed"]; }''')
    st.markdown('<div class="pw-note"><b>Accurate data flow.</b> The app validates the file, stores the original CSV in S3, '
                'and sends the parsed records to Lambda as JSON in the invocation payload. Lambda does not read from S3 in '
                'this version; S3 is the durable input archive. Lambda returns the anomalies, and the dashboard presents '
                'them. In demo mode the same Lambda code runs inside the app and nothing leaves it.</div>', unsafe_allow_html=True)

    st.subheader("Why cloud computing")
    st.markdown("""
**Problem.** Households, hostels and small businesses receive meter data they rarely analyse, so abnormal consumption goes unnoticed until the bill arrives.

- **Amazon S3** gives durable, low-cost storage for raw meter files, with versioning and encryption available.
- **AWS Lambda** runs the detection code on demand with no servers to manage, scaling with the number of requests and billed per execution.
- **Amazon QuickSight** (optional) provides managed, shareable BI dashboards on processed data.

**Scalability and availability.** S3 and Lambda scale automatically and are replicated across availability zones. Larger datasets would be sent in batches or read from S3 by an S3-triggered Lambda.
**Security.** Credentials live in Streamlit secrets or environment variables, never in code. Use a dedicated IAM user with least-privilege permissions, and keep the bucket private with server-side encryption (applied to every upload).
**Cost.** S3 and Lambda usually cost very little at demo scale, but AWS can incur charges; QuickSight has per-user pricing. A public Streamlit app is not itself an AWS deployment.
""")
    st.subheader("Amazon QuickSight integration (optional)")
    st.write("QuickSight can visualise processed results exported from this app (for example the anomaly report CSV in S3). "
             "If a dashboard is configured, it can be embedded using a registered-user embed URL.")
    if ready["quicksight"]:
        if st.button("Load QuickSight dashboard"):
            res = core.quicksight_embed_url(cfg)
            log_op("QuickSight embed URL", res)
            if res.ok:
                st.components.v1.iframe(res.data, height=700, scrolling=True)
            else:
                st.error(res.message)
    else:
        st.info("No QuickSight dashboard is configured. The dashboards in this app are PowerWatch's own Plotly dashboards, "
                "not QuickSight. Set QUICKSIGHT_DASHBOARD_ID, QUICKSIGHT_ACCOUNT_ID and QUICKSIGHT_USER_ARN to enable embedding.")


def page_about():
    st.title("About the Project")
    st.markdown("""
**PowerWatch** is a cloud-based electricity monitoring tool for households, college buildings, hostels and small businesses. It helps users upload meter readings, explore trends, find unusually high consumption and see practical ways to reduce waste.

**Method.** Z-score anomaly detection against a rolling, time-of-day baseline (same hour, same day type, previous N days). It is simple, explainable, and needs no training data. **Limitations:** it flags only high usage; a sustained level shift keeps being flagged until the baseline period passes; the first days of each meter cannot be judged; it cannot identify causes.

**Stack.** Python, Streamlit, Pandas, NumPy, Plotly, Boto3, Amazon S3, AWS Lambda, Amazon QuickSight (optional).

#### 3 to 5 minute demonstration script
1. **(30 s) Problem and design.** Explain unnoticed energy waste and open *AWS Architecture and System Status*. Point out the mode indicator and that nothing is faked.
2. **(60 s) Overview.** Show KPIs, the consumption chart with red markers, and the generated summary. Change the building or date range to show every KPI updating.
3. **(90 s) Demo scenario.** Click *Try a demo scenario*. Show flagged readings, open an explanation, move the sensitivity slider, and download the CSV report.
4. **(45 s) Recommendations.** Change the tariff and the assumed reduction, and note that figures are estimates.
5. **(45 s) Upload.** Download the template, upload a CSV with a bad row, and show the validation report.
6. **(30 s) Cloud.** Explain S3, Lambda and optional QuickSight. If AWS is configured, upload to S3, run *Test AWS connection*, and show the operation log.
""")


{"Overview": page_overview, "Consumption Analytics": page_analytics, "Anomaly Detection": page_anomalies,
 "Recommendations": page_recs, "Data Upload": page_upload,
 "AWS Architecture and System Status": page_arch, "About the Project": page_about}[page]()
footer()
