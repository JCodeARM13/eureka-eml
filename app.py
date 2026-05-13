"""Eureka-EML — Discover equations from data."""

import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Eureka", page_icon="", layout="wide", initial_sidebar_state="collapsed")

# ── Apple-minimal CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg: #ffffff;
    --fg: #1d1d1f;
    --muted: #86868b;
    --accent: #0071e3;
    --accent-light: #e8f0fe;
    --border: #e5e5e7;
    --card-bg: #fbfbfd;
    --success: #34c759;
    --radius: 16px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--fg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--card-bg) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.02em !important; }

/* Hero animation */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.6; }
}

.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    animation: fadeUp 0.8s ease-out;
}
.hero h1 {
    font-size: 3.2rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.04em !important;
    margin-bottom: 0.3rem !important;
    background: linear-gradient(135deg, #1d1d1f 0%, #0071e3 50%, #5e5ce6 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s ease-in-out infinite;
}
.hero p {
    color: var(--muted);
    font-size: 1.15rem;
    font-weight: 300;
    margin-top: 0;
}

/* Cards */
.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 0.5rem 0;
    animation: fadeUp 0.6s ease-out;
    transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.card:hover {
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    transform: translateY(-2px);
}

/* Metric */
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    text-align: center;
    animation: fadeUp 0.7s ease-out;
}
.metric-card .label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--fg);
}

/* Equation display */
.equation-box {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    animation: fadeUp 0.5s ease-out;
}
.equation-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.8rem;
}

/* Success pill */
.success-pill {
    display: inline-block;
    background: #e8fbe8;
    color: #1a7a1a;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 500;
    animation: fadeUp 0.4s ease-out;
}

/* Buttons */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #005bb5 !important;
    transform: scale(1.02) !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stStatusWidget"] { visibility: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-weight: 500;
    padding: 0.8rem 1.5rem;
    border-radius: 0;
}

/* Plotly */
.js-plotly-plot { border-radius: var(--radius) !important; overflow: hidden; }

/* Spinner */
.stSpinner > div { color: var(--accent) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: var(--radius) !important; overflow: hidden; }

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--fg) !important;
    font-weight: 400 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ── State ──────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Hero ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Eureka</h1>
    <p>Discover interpretable equations from your data</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    population = st.slider("Population", 10, 100, 30)
    generations = st.slider("Generations", 10, 200, 50)
    max_depth = st.slider("Max depth", 3, 8, 6)
    timeout = st.slider("Timeout (s)", 30, 300, 120)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:var(--muted);'>"
        "EML uses a single operator — <code>exp(x) - ln(y)</code> — "
        "to build all elementary functions. Eureka evolves and optimizes "
        "EML trees to find the equation hiding in your data."
        "</p>",
        unsafe_allow_html=True,
    )


# ── Helpers ────────────────────────────────────────────────────────
DEMO_1D = {
    "exp(x)": ("exp(x)", lambda x: np.exp(x), (-2, 2)),
    "sin(x)": ("sin(x)", lambda x: np.sin(x), (-3.14, 3.14)),
    "x²": ("x²", lambda x: x**2, (-3, 3)),
    "5 · exp(-0.3x)": ("5·exp(-0.3x)", lambda x: 5*np.exp(-0.3*x), (0, 10)),
}

DEMO_2D = {
    "E = mc² (Einstein)": {
        "label": "E = mc²",
        "vars": ["m", "c"],
        "target": "E",
        "gen": lambda n: _gen_emc2(n),
    },
}

DEMO_ALL = list(DEMO_1D.keys()) + list(DEMO_2D.keys())


def _gen_emc2(n):
    m = np.random.uniform(0.1, 10.0, n).astype(np.float32)
    c = np.random.uniform(1.0, 10.0, n).astype(np.float32)
    E = (m * c**2).astype(np.float32)
    return np.column_stack([m, c]), E


def _predict(result, X_np):
    import torch
    with torch.no_grad():
        return result.tree(torch.from_numpy(X_np).float()).numpy()


def _show_results(result, y_actual, y_pred, target_name,
                  feature_names=None, x_1d=None, true_label=None):
    from eureka.simplify import simplify_tree, to_latex, format_equation

    eq_text = simplify_tree(result.tree)
    eq_latex = to_latex(result.tree)
    eq_named = format_equation(result.tree, target_name, feature_names) if feature_names else f"{target_name} = {eq_text}"
    mse_val = float(np.mean((y_actual - y_pred) ** 2))

    # Success pill
    st.markdown('<div style="text-align:center;"><span class="success-pill">Equation discovered</span></div>', unsafe_allow_html=True)
    st.markdown("")

    # Equation
    st.markdown('<div class="equation-box"><div class="equation-label">Discovered Equation</div>', unsafe_allow_html=True)
    st.latex(f"{target_name} = {eq_latex}")
    st.markdown('</div>', unsafe_allow_html=True)

    if true_label:
        st.markdown(f'<p style="text-align:center; color:var(--muted); font-size:0.85rem;">Ground truth: <code>{true_label}</code></p>', unsafe_allow_html=True)

    # Metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'''<div class="metric-card">
            <div class="label">Mean Squared Error</div>
            <div class="value">{mse_val:.4e}</div>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="metric-card">
            <div class="label">Complexity</div>
            <div class="value">{result.complexity}</div>
        </div>''', unsafe_allow_html=True)
    with c3:
        r2 = 1 - np.sum((y_actual - y_pred)**2) / (np.sum((y_actual - y_actual.mean())**2) + 1e-10)
        st.markdown(f'''<div class="metric-card">
            <div class="label">R² Score</div>
            <div class="value">{r2:.4f}</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("")

    # Charts
    if x_1d is not None:
        sort_idx = np.argsort(x_1d)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_1d, y=y_actual, mode="markers", name="Data",
            marker=dict(size=5, opacity=0.4, color="#86868b"),
        ))
        fig.add_trace(go.Scatter(
            x=x_1d[sort_idx], y=y_pred[sort_idx], mode="lines", name="Discovered",
            line=dict(color="#0071e3", width=2.5),
        ))
        fig.update_layout(
            template="plotly_white", height=380, margin=dict(l=40, r=20, t=30, b=40),
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Actual vs Predicted", "Residuals"],
                            horizontal_spacing=0.08)
        fig.add_trace(go.Scatter(
            x=y_actual, y=y_pred, mode="markers", name="Points",
            marker=dict(size=5, opacity=0.5, color="#0071e3"),
        ), row=1, col=1)
        mn, mx = min(y_actual.min(), y_pred.min()), max(y_actual.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines", name="",
            line=dict(dash="dash", color="#e5e5e7", width=1), showlegend=False,
        ), row=1, col=1)
        residuals = y_actual - y_pred
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers", name="Residuals",
            marker=dict(size=4, opacity=0.4, color="#ff9500"), showlegend=False,
        ), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="#e5e5e7", line_width=1, row=1, col=2)
        fig.update_layout(
            template="plotly_white", height=350, margin=dict(l=40, r=20, t=40, b=40),
            font=dict(family="Inter, sans-serif", size=11), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Export
    col_l, col_r = st.columns([3, 1])
    with col_r:
        export = {"equation": eq_named, "latex": eq_latex, "mse": mse_val,
                  "complexity": result.complexity, "r2": float(r2)}
        st.download_button("Export JSON", json.dumps(export, indent=2),
                           "eureka_result.json", "application/json")

    # History
    st.session_state.history.append({
        "time": time.strftime("%H:%M"),
        "equation": eq_text[:40] + ("..." if len(eq_text) > 40 else ""),
        "mse": f"{mse_val:.2e}",
        "nodes": result.complexity,
        "r2": f"{r2:.3f}",
    })


# ── Tabs ───────────────────────────────────────────────────────────
tab_csv, tab_demo = st.tabs(["Your Data", "Try a Demo"])

with tab_csv:
    st.markdown("")
    uploaded = st.file_uploader("Drop a CSV file here", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        with st.expander("Preview", expanded=True):
            st.dataframe(df.head(8), use_container_width=True, hide_index=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.markdown('<p style="color:#ff3b30; text-align:center;">Need at least 2 numeric columns.</p>', unsafe_allow_html=True)
        else:
            col_sel, col_btn = st.columns([2, 1])
            with col_sel:
                target = st.selectbox("Target column", numeric_cols, label_visibility="collapsed")
            feature_cols = [c for c in numeric_cols if c != target]
            st.markdown(f'<p style="color:var(--muted); font-size:0.8rem;">{len(feature_cols)} features · {len(df)} samples</p>', unsafe_allow_html=True)

            if len(df) > 1000:
                st.markdown('<p style="color:#ff9500; font-size:0.8rem;">Large dataset — consider increasing timeout.</p>', unsafe_allow_html=True)

            if st.button("Discover", type="primary", key="csv_go", use_container_width=True):
                y = df[target].values.astype(np.float32)
                X = df[feature_cols].values.astype(np.float32)
                with st.spinner(""):
                    try:
                        from eureka.regression import discover
                        result = discover(X, y, population=population, generations=generations,
                                          max_depth=max_depth, timeout=timeout, verbose=False)
                        y_pred = _predict(result, X)
                        _show_results(result, y, y_pred, target, feature_cols)
                    except Exception as e:
                        st.markdown(f'<p style="color:#ff3b30; text-align:center;">{e}</p>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:var(--muted);">
            <p style="font-size:2rem; margin-bottom:0.5rem;">↑</p>
            <p>Upload a CSV to get started</p>
        </div>
        """, unsafe_allow_html=True)

with tab_demo:
    st.markdown("")
    col_fn, col_n, col_noise = st.columns(3)
    with col_fn:
        demo_choice = st.selectbox("Function", DEMO_ALL, label_visibility="collapsed")
    with col_n:
        n_samples = st.slider("Samples", 50, 500, 200, label_visibility="collapsed")
    with col_noise:
        noise_level = st.slider("Noise", 0.0, 1.0, 0.1, label_visibility="collapsed")

    is_2d = demo_choice in DEMO_2D

    if st.button("Discover", type="primary", key="demo_go", use_container_width=True):
        with st.spinner(""):
            try:
                from eureka.regression import discover

                if is_2d:
                    info = DEMO_2D[demo_choice]
                    np.random.seed(42)
                    X, y_true = info["gen"](n_samples)
                    y = (y_true + noise_level * np.random.randn(n_samples).astype(np.float32) * max(y_true.std(), 0.1)).astype(np.float32)
                    result = discover(X, y, population=population, generations=generations,
                                      max_depth=max_depth, timeout=timeout, verbose=False)
                    y_pred = _predict(result, X)
                    _show_results(result, y, y_pred, info["target"],
                                  feature_names=info["vars"], true_label=info["label"])
                else:
                    true_name, fn, (lo, hi) = DEMO_1D[demo_choice]
                    x = np.linspace(lo, hi, n_samples).astype(np.float32)
                    y_true = fn(x).astype(np.float32)
                    y = (y_true + noise_level * np.random.randn(n_samples).astype(np.float32) * max(y_true.std(), 0.1)).astype(np.float32)
                    X = x.reshape(-1, 1)
                    result = discover(X, y, population=population, generations=generations,
                                      max_depth=max_depth, timeout=timeout, verbose=False)
                    y_pred = _predict(result, X)
                    _show_results(result, y, y_pred, "y", x_1d=x, true_label=true_name)
            except Exception as e:
                st.markdown(f'<p style="color:#ff3b30; text-align:center;">{e}</p>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="text-align:center; padding:2rem 1rem; color:var(--muted);">
            <p style="font-size:0.9rem;">Select a function and hit <strong>Discover</strong> to see Eureka in action</p>
        </div>
        """, unsafe_allow_html=True)

# ── History ────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown(f'<p style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted);">History · {len(st.session_state.history)} runs</p>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
