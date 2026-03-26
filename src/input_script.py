"""
Credit Card Fraud Detection — Premium Streamlit Dashboard
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudShield AI — Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — glass-morphism cards, animations, premium typography
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Root variables */
:root {
    --accent: #6C63FF;
    --accent-light: #8B83FF;
    --accent-glow: rgba(108,99,255,0.35);
    --success: #00D68F;
    --danger: #FF3D71;
    --warning: #FFAA00;
    --surface: rgba(26,29,41,0.65);
    --surface-border: rgba(108,99,255,0.18);
    --text-primary: #FAFAFA;
    --text-secondary: #A0A3BD;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide default streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Animated gradient hero */
.hero-banner {
    background: linear-gradient(135deg, #0E1117 0%, #1a1040 30%, #2d1b69 60%, #0E1117 100%);
    background-size: 300% 300%;
    animation: gradient-shift 8s ease infinite;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    border: 1px solid var(--surface-border);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 50%);
    animation: pulse-glow 4s ease-in-out infinite;
}
@keyframes gradient-shift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes pulse-glow {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.05); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    position: relative;
    z-index: 2;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    position: relative;
    z-index: 2;
}

/* Glass metric cards */
.metric-card {
    background: var(--surface);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--surface-border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(.25,.8,.25,1);
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px var(--accent-glow);
    border-color: var(--accent);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.3rem 0;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 500;
}
.accent { color: var(--accent-light); }
.success { color: var(--success); }
.danger { color: var(--danger); }
.warning { color: var(--warning); }

/* Prediction result badges */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    margin: 1rem 0;
    animation: fade-in 0.5s ease;
}
.result-fraud {
    background: linear-gradient(135deg, rgba(255,61,113,0.15), rgba(255,61,113,0.05));
    border: 2px solid var(--danger);
    color: var(--danger);
}
.result-legit {
    background: linear-gradient(135deg, rgba(0,214,143,0.15), rgba(0,214,143,0.05));
    border: 2px solid var(--success);
    color: var(--success);
}

/* Confidence bar */
.conf-bar-wrapper {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.3rem;
    margin: 0.5rem 0;
}
.conf-bar {
    height: 10px;
    border-radius: 10px;
    transition: width 1s cubic-bezier(.25,.8,.25,1);
}

/* Section headers */
.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

@keyframes fade-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background: linear-gradient(180deg, #0E1117 0%, #1a1040 100%);
}

/* Feature importance bar tweaks */
.stPlotlyChart {
    border-radius: 16px;
    overflow: hidden;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model & encoder (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    encoder = joblib.load(MODELS_DIR / "ordinal_encoder.pkl")
    return model, encoder


model, encoder = load_artifacts()

CATEGORICAL_COLS = ["type", "nameOrig", "nameDest"]
FEATURE_NAMES = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFlaggedFraud",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    enc = df.copy()
    enc[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])
    return enc


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful derived features for display / analysis."""
    out = df.copy()
    out["balance_change_orig"] = out["newbalanceOrig"] - out["oldbalanceOrg"]
    out["balance_change_dest"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["amount_to_balance_ratio"] = np.where(
        out["oldbalanceOrg"] > 0,
        out["amount"] / out["oldbalanceOrg"],
        0,
    )
    return out


def predict(df: pd.DataFrame):
    enc = preprocess(df)
    probs = model.predict_proba(enc)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs


def confidence_bar(value: float):
    """Return HTML for a gradient confidence bar."""
    pct = value * 100
    if value < 0.3:
        color = "var(--success)"
    elif value < 0.7:
        color = "var(--warning)"
    else:
        color = "var(--danger)"
    return f"""
    <div class='conf-bar-wrapper'>
        <div class='conf-bar' style='width:{pct}%; background: {color};'></div>
    </div>
    """


def metric_card(label: str, value: str, css_class: str = "accent"):
    return f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value {css_class}'>{value}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛡️ FraudShield AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "🔍 Predict", "📊 Model Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    # Hero
    st.markdown("""
    <div class='hero-banner'>
        <p class='hero-title'>🛡️ FraudShield AI</p>
        <p class='hero-subtitle'>Real-time credit card fraud detection powered by XGBoost — 
        analyze transactions, explore model insights, and protect your finances.</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats from dataset
    @st.cache_data
    def load_data_summary():
        try:
            # Try to read just the first rows and the target for speed
            df = pd.read_csv(DATA_DIR / "data.csv")
            total = len(df)
            fraud = int(df["isFraud"].sum())
            legit = total - fraud
            fraud_pct = fraud / total * 100
            avg_amount = df["amount"].mean()
            types = df["type"].value_counts().to_dict()
            # Sample for charts
            sample = df.sample(min(5000, total), random_state=42)
            return {
                "total": total, "fraud": fraud, "legit": legit,
                "fraud_pct": fraud_pct, "avg_amount": avg_amount,
                "types": types, "sample": sample,
            }
        except Exception:
            return None

    stats = load_data_summary()

    if stats:
        # Metric cards row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card("Total Transactions", f"{stats['total']:,}"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Fraud Cases", f"{stats['fraud']:,}", "danger"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Legitimate", f"{stats['legit']:,}", "success"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Fraud Rate", f"{stats['fraud_pct']:.3f}%", "warning"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts row
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("<div class='section-header'>📈 Transaction Type Distribution</div>", unsafe_allow_html=True)
            type_df = pd.DataFrame(
                list(stats["types"].items()), columns=["Type", "Count"]
            ).sort_values("Count", ascending=True)
            fig_bar = px.bar(
                type_df, x="Count", y="Type", orientation="h",
                color="Count",
                color_continuous_scale=["#2d1b69", "#6C63FF", "#8B83FF"],
            )
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA", family="Inter"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=10, b=0),
                height=350,
                yaxis=dict(title=""),
                xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.markdown("<div class='section-header'>🎯 Fraud vs Legitimate</div>", unsafe_allow_html=True)
            fig_donut = go.Figure(data=[
                go.Pie(
                    labels=["Legitimate", "Fraud"],
                    values=[stats["legit"], stats["fraud"]],
                    hole=0.65,
                    marker=dict(colors=["#00D68F", "#FF3D71"]),
                    textinfo="percent+label",
                    textfont=dict(size=14, family="Inter"),
                )
            ])
            fig_donut.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA", family="Inter"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Amount distribution
        st.markdown("<div class='section-header'>💰 Transaction Amount Distribution (log scale)</div>", unsafe_allow_html=True)
        sample = stats["sample"]
        fig_hist = px.histogram(
            sample, x="amount", color=sample["isFraud"].map({0: "Legitimate", 1: "Fraud"}),
            nbins=80, log_y=True, barmode="overlay",
            color_discrete_map={"Legitimate": "#6C63FF", "Fraud": "#FF3D71"},
            opacity=0.75,
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FAFAFA", family="Inter"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
            xaxis=dict(title="Amount", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Count (log)", gridcolor="rgba(255,255,255,0.05)"),
            legend_title_text="",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("📂 Place `data.csv` in the `data/` folder to see dashboard statistics.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.markdown("""
    <div class='hero-banner' style='padding:1.8rem 2rem;'>
        <p class='hero-title' style='font-size:1.8rem;'>🔍 Fraud Prediction</p>
        <p class='hero-subtitle'>Enter transaction details manually or upload a CSV for batch predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_manual, tab_csv = st.tabs(["✏️ Manual Input", "📁 CSV Upload"])

    # --- Manual Input ---
    with tab_manual:
        st.markdown("<div class='section-header'>Enter Transaction Details</div>", unsafe_allow_html=True)

        with st.form("manual_prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                txn_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
                amount = st.number_input("Amount ($)", min_value=0.0, step=100.0, format="%.2f")
                step = st.number_input("Time Step", min_value=0, step=1, help="Hour of simulation (1 step = 1 hour)")
            with col2:
                name_orig = st.text_input("Origin Account", value="C1234567890", help="Customer ID starting with C")
                old_bal_orig = st.number_input("Origin Old Balance", min_value=0.0, step=100.0, format="%.2f")
                new_bal_orig = st.number_input("Origin New Balance", min_value=0.0, step=100.0, format="%.2f")
            with col3:
                name_dest = st.text_input("Destination Account", value="M9876543210", help="Merchant (M) or Customer (C)")
                old_bal_dest = st.number_input("Dest Old Balance", min_value=0.0, step=100.0, format="%.2f")
                new_bal_dest = st.number_input("Dest New Balance", min_value=0.0, step=100.0, format="%.2f")

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Analyze Transaction", use_container_width=True, type="primary")

        if submitted:
            is_flagged = 1 if amount > 200_000 else 0
            input_df = pd.DataFrame({
                "step": [step], "type": [txn_type], "amount": [amount],
                "nameOrig": [name_orig], "oldbalanceOrg": [old_bal_orig],
                "newbalanceOrig": [new_bal_orig], "nameDest": [name_dest],
                "oldbalanceDest": [old_bal_dest], "newbalanceDest": [new_bal_dest],
                "isFlaggedFraud": [is_flagged],
            })

            preds, probs = predict(input_df)
            conf = probs[0]
            is_fraud = preds[0] == 1

            # Store in session state so results persist
            st.session_state["manual_result"] = {
                "is_fraud": is_fraud, "conf": conf, "input_df": input_df,
            }

        # Display results from session state (persists across reruns)
        if "manual_result" in st.session_state:
            res = st.session_state["manual_result"]
            conf = res["conf"]
            is_fraud = res["is_fraud"]
            input_df = res["input_df"]

            if is_fraud:
                st.markdown(
                    "<div class='result-badge result-fraud'>🚨 FRAUDULENT TRANSACTION DETECTED</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-badge result-legit'>✅ TRANSACTION IS LEGITIMATE</div>",
                    unsafe_allow_html=True,
                )

            st.markdown(f"**Fraud Probability:** {conf:.4f} ({conf*100:.2f}%)")
            st.markdown(confidence_bar(conf), unsafe_allow_html=True)

            # Feature analysis
            enriched = add_engineered_features(input_df)
            st.markdown("<div class='section-header'>📋 Transaction Analysis</div>", unsafe_allow_html=True)

            a1, a2, a3 = st.columns(3)
            with a1:
                bal_change = enriched["balance_change_orig"].iloc[0]
                st.markdown(
                    metric_card("Origin Bal. Change", f"${bal_change:,.2f}",
                                "danger" if bal_change < 0 else "success"),
                    unsafe_allow_html=True,
                )
            with a2:
                bal_change_d = enriched["balance_change_dest"].iloc[0]
                st.markdown(
                    metric_card("Dest Bal. Change", f"${bal_change_d:,.2f}",
                                "success" if bal_change_d > 0 else "danger"),
                    unsafe_allow_html=True,
                )
            with a3:
                ratio = enriched["amount_to_balance_ratio"].iloc[0]
                st.markdown(
                    metric_card("Amt / Orig Balance", f"{ratio:.2f}x",
                                "danger" if ratio > 1 else "accent"),
                    unsafe_allow_html=True,
                )

    # --- CSV Upload ---
    with tab_csv:
        st.markdown("<div class='section-header'>Upload Transaction CSV</div>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='color:var(--text-secondary);'>Required columns: "
            f"<code>{'</code>, <code>'.join(FEATURE_NAMES)}</code></p>",
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

        if uploaded is not None:
            try:
                csv_data = pd.read_csv(uploaded)
                missing = [c for c in FEATURE_NAMES if c not in csv_data.columns]

                if missing:
                    st.error(f"❌ Missing columns: {', '.join(missing)}")
                else:
                    st.markdown(f"**Preview** — {len(csv_data):,} transactions loaded")
                    st.dataframe(csv_data.head(10), use_container_width=True)

                    if st.button("🚀 Run Batch Prediction", use_container_width=True, type="primary"):
                        with st.spinner("Analyzing transactions…"):
                            preds, probs = predict(csv_data[FEATURE_NAMES])

                        results = csv_data.copy()
                        results["fraud_prediction"] = preds
                        results["confidence"] = probs
                        results["risk_level"] = pd.cut(
                            probs,
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=["🟢 Low", "🟡 Medium", "🔴 High"],
                        )

                        fraud_count = int(preds.sum())
                        legit_count = len(preds) - fraud_count

                        # Summary cards
                        s1, s2, s3, s4 = st.columns(4)
                        with s1:
                            st.markdown(metric_card("Total", f"{len(preds):,}"), unsafe_allow_html=True)
                        with s2:
                            st.markdown(metric_card("Fraud", f"{fraud_count:,}", "danger"), unsafe_allow_html=True)
                        with s3:
                            st.markdown(metric_card("Legitimate", f"{legit_count:,}", "success"), unsafe_allow_html=True)
                        with s4:
                            avg_conf = probs.mean()
                            st.markdown(metric_card("Avg Confidence", f"{avg_conf:.3f}", "warning"), unsafe_allow_html=True)

                        # Risk distribution chart
                        st.markdown("<div class='section-header'>📊 Risk Distribution</div>", unsafe_allow_html=True)
                        fig_risk = px.histogram(
                            results, x="confidence", nbins=50,
                            color_discrete_sequence=["#6C63FF"],
                        )
                        fig_risk.add_vline(x=0.5, line_dash="dash", line_color="#FF3D71",
                                           annotation_text="Threshold (0.5)")
                        fig_risk.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#FAFAFA", family="Inter"),
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=300,
                            xaxis=dict(title="Fraud Probability", gridcolor="rgba(255,255,255,0.05)"),
                            yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.05)"),
                        )
                        st.plotly_chart(fig_risk, use_container_width=True)

                        # Results table
                        st.markdown("<div class='section-header'>📋 Detailed Results</div>", unsafe_allow_html=True)
                        st.dataframe(
                            results.sort_values("confidence", ascending=False),
                            use_container_width=True,
                            height=400,
                        )

                        # Download
                        csv_out = results.to_csv(index=False)
                        st.download_button(
                            "⬇️ Download Results CSV",
                            data=csv_out,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":
    st.markdown("""
    <div class='hero-banner' style='padding:1.8rem 2rem;'>
        <p class='hero-title' style='font-size:1.8rem;'>📊 Model Insights</p>
        <p class='hero-subtitle'>Explore feature importance, model parameters, and performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_feat, tab_params, tab_perf = st.tabs(["🌟 Feature Importance", "⚙️ Parameters", "📈 Performance"])

    with tab_feat:
        st.markdown("<div class='section-header'>Feature Importance (Gain)</div>", unsafe_allow_html=True)

        importance = model.get_booster().get_score(importance_type="gain")
        if importance:
            imp_df = pd.DataFrame(
                list(importance.items()), columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)

            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature", orientation="h",
                color="Importance",
                color_continuous_scale=["#1a1040", "#6C63FF", "#FF3D71"],
            )
            fig_imp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA", family="Inter"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=20, t=10, b=0),
                height=450,
                yaxis=dict(title=""),
                xaxis=dict(title="Gain", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # Radar chart
            st.markdown("<div class='section-header'>🕸️ Feature Radar</div>", unsafe_allow_html=True)
            radar_df = imp_df.tail(8)  # top 8
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_df["Importance"].tolist() + [radar_df["Importance"].iloc[0]],
                theta=radar_df["Feature"].tolist() + [radar_df["Feature"].iloc[0]],
                fill='toself',
                fillcolor='rgba(108,99,255,0.2)',
                line=dict(color='#6C63FF', width=2),
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, gridcolor="rgba(255,255,255,0.1)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FAFAFA", family="Inter"),
                margin=dict(l=60, r=60, t=30, b=30),
                height=420,
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("No feature importance data available.")

    with tab_params:
        st.markdown("<div class='section-header'>Model Configuration</div>", unsafe_allow_html=True)
        params = model.get_params()
        param_df = pd.DataFrame(
            [(k, str(v)) for k, v in sorted(params.items())],
            columns=["Parameter", "Value"],
        )
        st.dataframe(param_df, use_container_width=True, height=500)

    with tab_perf:
        st.markdown("<div class='section-header'>Run Live Evaluation</div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:var(--text-secondary);'>Click below to evaluate the model on 20% holdout data.</p>",
            unsafe_allow_html=True,
        )

        if st.button("▶️ Evaluate Model", type="primary", use_container_width=True):
            with st.spinner("Loading data and evaluating…"):
                try:
                    from sklearn.metrics import (
                        accuracy_score,
                        classification_report,
                        confusion_matrix,
                        f1_score,
                        precision_score,
                        recall_score,
                        roc_auc_score,
                        roc_curve,
                    )
                    from sklearn.model_selection import train_test_split
                    from sklearn.preprocessing import OrdinalEncoder

                    df = pd.read_csv(DATA_DIR / "data.csv")
                    X = df.drop("isFraud", axis=1)
                    Y = df["isFraud"]
                    cat_cols = X.select_dtypes(include="object").columns.tolist()

                    X_train, X_test, Y_train, Y_test = train_test_split(
                        X, Y, test_size=0.2, random_state=42
                    )
                    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                    enc.fit(X_train[cat_cols])
                    X_test_enc = X_test.copy()
                    X_test_enc[cat_cols] = enc.transform(X_test[cat_cols])

                    y_probs = model.predict_proba(X_test_enc)[:, 1]
                    y_pred = (y_probs >= 0.5).astype(int)

                    acc = accuracy_score(Y_test, y_pred)
                    prec = precision_score(Y_test, y_pred)
                    rec = recall_score(Y_test, y_pred)
                    f1 = f1_score(Y_test, y_pred)
                    auc = roc_auc_score(Y_test, y_probs)

                    # Metric cards
                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        st.markdown(metric_card("Accuracy", f"{acc:.4f}", "accent"), unsafe_allow_html=True)
                    with m2:
                        st.markdown(metric_card("Precision", f"{prec:.4f}", "success"), unsafe_allow_html=True)
                    with m3:
                        st.markdown(metric_card("Recall", f"{rec:.4f}", "warning"), unsafe_allow_html=True)
                    with m4:
                        st.markdown(metric_card("F1 Score", f"{f1:.4f}", "accent"), unsafe_allow_html=True)
                    with m5:
                        st.markdown(metric_card("AUC-ROC", f"{auc:.4f}", "danger"), unsafe_allow_html=True)

                    # ROC Curve
                    st.markdown("<div class='section-header'>📉 ROC Curve</div>", unsafe_allow_html=True)
                    fpr, tpr, _ = roc_curve(Y_test, y_probs)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"AUC = {auc:.4f}",
                        line=dict(color="#6C63FF", width=3),
                        fill="tozeroy",
                        fillcolor="rgba(108,99,255,0.15)",
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], mode="lines",
                        name="Random", line=dict(dash="dash", color="#A0A3BD", width=1),
                    ))
                    fig_roc.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FAFAFA", family="Inter"),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=400,
                        xaxis=dict(title="False Positive Rate", gridcolor="rgba(255,255,255,0.05)"),
                        yaxis=dict(title="True Positive Rate", gridcolor="rgba(255,255,255,0.05)"),
                        legend=dict(x=0.6, y=0.1),
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                    # Confusion Matrix
                    st.markdown("<div class='section-header'>🔢 Confusion Matrix</div>", unsafe_allow_html=True)
                    cm = confusion_matrix(Y_test, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Legitimate", "Fraud"],
                        y=["Legitimate", "Fraud"],
                        color_continuous_scale=["#0E1117", "#6C63FF", "#FF3D71"],
                        text_auto=True,
                    )
                    fig_cm.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#FAFAFA", family="Inter", size=14),
                        margin=dict(l=0, r=0, t=10, b=0),
                        height=380,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

                except Exception as e:
                    st.error(f"Evaluation error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ═══════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown("""
    <div class='hero-banner' style='padding:1.8rem 2rem;'>
        <p class='hero-title' style='font-size:1.8rem;'>ℹ️ About FraudShield AI</p>
        <p class='hero-subtitle'>Learn about the model, data, and methodology behind the system.</p>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("""
        ### 🧠 Model Architecture
        
        **Algorithm:** XGBoost (Extreme Gradient Boosting)
        
        - Handles class imbalance via `scale_pos_weight`
        - Ordinal encoding for categorical features
        - Optimized probability threshold for fraud detection
        
        ### 📊 Dataset
        
        - **Source:** PaySim synthetic financial dataset
        - **Features:** 10 transaction attributes
        - **Target:** `isFraud` (binary)
        - Highly imbalanced — fraud < 0.2% of transactions
        """)

    with col_r:
        st.markdown("""
        ### 🔧 Feature Engineering
        
        | Feature | Description |
        |---------|------------|
        | `step` | Hour of the simulation |
        | `type` | CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER |
        | `amount` | Transaction amount |
        | `oldbalanceOrg` | Origin balance before txn |
        | `newbalanceOrig` | Origin balance after txn |
        | `oldbalanceDest` | Destination balance before txn |
        | `newbalanceDest` | Destination balance after txn |
        | `isFlaggedFraud` | System flag for amounts > $200K |
        
        ### 🏗️ Tech Stack
        
        `Python` · `Streamlit` · `XGBoost` · `scikit-learn` · `Plotly` · `Pandas`
        """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#A0A3BD;'>Built with ❤️ using Streamlit &bull; "
        "FraudShield AI v2.0</p>",
        unsafe_allow_html=True,
    )