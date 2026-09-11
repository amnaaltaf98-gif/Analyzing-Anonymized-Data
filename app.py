"""
Anonymized Sensor Dataset — Exploratory Data Analysis Dashboard
Built from a raw EDA notebook. Single consistent color scheme throughout.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sensor Data EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# COLOR SCHEME — one palette, used everywhere (charts, badges, accents)
# ----------------------------------------------------------------------------
PRIMARY = "#1F4E79"      # deep steel blue   -> main / default series
SECONDARY = "#2A9D8F"    # teal              -> secondary series / positive
ACCENT = "#E76F51"       # warm coral        -> outliers, warnings, "CPU"
NEUTRAL = "#8C97A5"      # slate gray        -> de-emphasized elements
LIGHT = "#F2F6F8"        # pale blue-gray    -> backgrounds / cards
DARK_TEXT = "#1A1A1A"

# Sequential shades of the primary palette for the 5 machine groups (F1)
MACHINE_COLORS = {
    "M1": "#1F4E79",
    "M2": "#2E6F95",
    "M3": "#2A9D8F",
    "M4": "#6FBFAE",
    "M5": "#B7DED2",
}
# Binary category palette for F16 (kept consistent everywhere it appears)
F16_COLORS = {"MOB": PRIMARY, "CPU": ACCENT}

PLOTLY_TEMPLATE = "plotly_white"
FONT = dict(family="Helvetica, Arial, sans-serif", color=DARK_TEXT)

def style_fig(fig, height=None, legend_title=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=FONT,
        title_font=dict(size=17, color=PRIMARY),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=30, t=60, b=40),
    )
    if height:
        fig.update_layout(height=height)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig

# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    h1, h2, h3 {{ color: {PRIMARY}; }}
    .insight-box {{
        background-color: {LIGHT};
        border-left: 5px solid {SECONDARY};
        padding: 14px 18px;
        border-radius: 6px;
        margin: 12px 0px;
        font-size: 0.95rem;
    }}
    .why-box {{
        background-color: #FFFFFF;
        border: 1px dashed {NEUTRAL};
        padding: 12px 16px;
        border-radius: 6px;
        margin: 10px 0px 18px 0px;
        font-size: 0.9rem;
        color: #444;
    }}
    .metric-card {{
        background-color: {LIGHT};
        padding: 14px;
        border-radius: 8px;
        text-align: center;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {LIGHT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

def why_box(text):
    st.markdown(f'<div class="why-box">🎯 <b>Why this subset:</b> {text}</div>', unsafe_allow_html=True)

def insight_box(text):
    st.markdown(f'<div class="insight-box">💡 <b>What this could mean:</b> {text}</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df["F2_seq"] = df["F2"].str.extract(r"(\d+)").astype(int)
    return df

DATA_PATH = "data/data.csv"

st.sidebar.title("📊 EDA Dashboard")
uploaded = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])

try:
    if uploaded is not None:
        df = load_data(uploaded)
    else:
        df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("No dataset found. Please upload the CSV using the sidebar uploader.")
    st.stop()

num_cols = [c for c in df.columns if c not in ["F1", "F2", "F16", "F2_seq"]]
cat_cols = ["F1", "F16"]

BIMODAL_COLS = ["F3", "F4", "F5", "F6", "F7", "F19"]
RIGHT_SKEW_COLS = ["F8", "F9", "F10", "F11", "F12", "F13", "F17"]
LEFT_SKEW_COLS = ["F15"]
UNIFORM_COLS = ["F14"]
NORMAL_COLS = ["F18"]
ALL_SHAPE_COLS = BIMODAL_COLS + RIGHT_SKEW_COLS + LEFT_SKEW_COLS + UNIFORM_COLS + NORMAL_COLS

OUTLIER_COLS = ["F8", "F9", "F10", "F11", "F12", "F13"]

REDUNDANT_PAIRS = [
    ("F3", "F4"), ("F3", "F6"), ("F3", "F7"),
    ("F12", "F13"),
    ("F8", "F9"), ("F8", "F10"), ("F9", "F10"),
    ("F14", "F15"),
    ("F18", "F19"),
]

# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
PAGES = [
    "🏠 Overview",
    "🧹 Data Quality",
    "🗂️ Categorical Breakdown",
    "📈 Numeric Distributions",
    "🚨 Outliers by Machine",
    "🔗 Feature Redundancy",
    "🔀 Relationship Shapes",
    "🧭 PCA — All Features at Once",
    "⏱️ F2 as Time",
]
page = st.sidebar.radio("Sections", PAGES)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div class="metric-card">
    <b>{df.shape[0]}</b> rows &nbsp;·&nbsp; <b>{df.shape[1]-1}</b> columns<br>
    <span style="color:{NEUTRAL}; font-size:0.8rem;">F1 groups: M1–M5 &nbsp;|&nbsp; F16: MOB / CPU</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# PAGE 1 — OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.title("📊 Anonymized Sensor Dataset — Exploratory Analysis")
    st.markdown(
        "This dashboard walks through an end-to-end EDA of an anonymized dataset "
        "with 19 features (F1–F19) collected across **5 machines (M1–M5)** and split "
        "into two device categories, **MOB** and **CPU**. The goal was to understand "
        "the *shape* of each feature, find *redundant* or duplicated signals, and see "
        "whether the anonymized columns hide a real underlying story."
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Rows", "Numeric features", "Categorical features", "Missing values"],
        [df.shape[0], len(num_cols), len(cat_cols), int(df.isna().sum().sum())],
    ):
        col.markdown(
            f'<div class="metric-card"><h2 style="margin:0;color:{PRIMARY}">{value}</h2>'
            f'<span style="color:{NEUTRAL}">{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### How to read this dashboard")
    st.markdown(
        """
        Every section follows the same pattern:
        - **🎯 Why this subset** — which features are shown and the reasoning for grouping them together.
        - **The chart(s)** — all using the same color scheme, so a color always means the same thing across the whole dashboard.
        - **💡 What this could mean** — a plain-language interpretation you can use when presenting.

        **Color key used throughout:**
        """
    )
    legend_cols = st.columns(7)
    swatches = [("M1", MACHINE_COLORS["M1"]), ("M2", MACHINE_COLORS["M2"]), ("M3", MACHINE_COLORS["M3"]),
                ("M4", MACHINE_COLORS["M4"]), ("M5", MACHINE_COLORS["M5"]),
                ("MOB", F16_COLORS["MOB"]), ("CPU", F16_COLORS["CPU"])]
    for col, (label, color) in zip(legend_cols, swatches):
        col.markdown(
            f'<div style="background:{color};color:white;border-radius:6px;'
            f'padding:8px;text-align:center;font-size:0.85rem;">{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Raw data preview")
    st.dataframe(df.drop(columns=["F2_seq"]).head(10), use_container_width=True)

# ============================================================================
# PAGE 2 — DATA QUALITY
# ============================================================================
elif page == "🧹 Data Quality":
    st.title("🧹 Data Quality Check")
    why_box(
        "Before trusting any pattern in the data, we confirm there are no missing values "
        "or duplicate rows that could distort distributions, correlations, or the PCA later on."
    )

    missing = df.drop(columns=["F2_seq"]).isna().sum()
    dup_count = df.drop(columns=["F2_seq"]).duplicated().sum()

    c1, c2 = st.columns([2, 1])
    with c1:
        fig = px.bar(
            x=missing.index, y=missing.values,
            labels={"x": "Column", "y": "Missing values"},
            title="Missing values per column",
        )
        fig.update_traces(marker_color=PRIMARY)
        fig = style_fig(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(
            f"""
            <div class="metric-card" style="height:100%;display:flex;flex-direction:column;justify-content:center;">
            <h1 style="color:{SECONDARY};margin:0;">0</h1>
            <span>total missing values</span>
            <hr>
            <h1 style="color:{SECONDARY};margin:0;">{dup_count}</h1>
            <span>duplicate rows</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    insight_box(
        "The dataset is already clean — no imputation, deduplication, or row-dropping is "
        "needed before analysis. Any patterns we find later (bimodal distributions, outlier "
        "clusters, correlations) reflect the real underlying process, not data-entry artifacts."
    )

# ============================================================================
# PAGE 3 — CATEGORICAL BREAKDOWN
# ============================================================================
elif page == "🗂️ Categorical Breakdown":
    st.title("🗂️ Categorical Breakdown — F1 and F16")
    why_box(
        "F1 and F16 are the only two categorical columns in the dataset. They matter because "
        "every downstream grouping (outlier analysis, stratified scatter plots, PCA coloring) "
        "hinges on these two labels, so we check their balance first."
    )

    c1, c2 = st.columns(2)
    with c1:
        f1_counts = df["F1"].value_counts().sort_index()
        fig = px.bar(
            x=f1_counts.index, y=f1_counts.values,
            labels={"x": "F1 (machine group)", "y": "Row count"},
            title="Rows per F1 group",
            color=f1_counts.index, color_discrete_map=MACHINE_COLORS,
        )
        fig.update_layout(showlegend=False)
        fig = style_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        f16_counts = df["F16"].value_counts()
        fig = px.bar(
            x=f16_counts.index, y=f16_counts.values,
            labels={"x": "F16 (device category)", "y": "Row count"},
            title="Rows per F16 category",
            color=f16_counts.index, color_discrete_map=F16_COLORS,
        )
        fig.update_layout(showlegend=False)
        fig = style_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "F1 is perfectly balanced — 26 rows for each of the 5 machines (M1–M5) — which looks "
        "like a controlled sampling design rather than organic/random logging. F16 is "
        "imbalanced (105 MOB vs 25 CPU rows), meaning MOB is the dominant device category. "
        "Any comparison between MOB and CPU later on should keep this imbalance in mind — "
        "CPU patterns are based on a much smaller sample."
    )

# ============================================================================
# PAGE 4 — NUMERIC DISTRIBUTIONS
# ============================================================================
elif page == "📈 Numeric Distributions":
    st.title("📈 Numeric Distributions")
    why_box(
        "The 16 numeric columns (F3–F19, excluding F16) are grouped by the *shape* of their "
        "distribution rather than shown in column order. Features that share a shape often "
        "share a cause, so grouping them this way makes the patterns easier to explain."
    )

    tabs = st.tabs([
        "Bimodal (F3,F4,F5,F6,F7,F19)",
        "Right-skewed (F8,F9,F10,F11,F12,F13,F17)",
        "Left-skewed (F15)",
        "Uniform (F14)",
        "Normal (F18)",
    ])
    groups = [BIMODAL_COLS, RIGHT_SKEW_COLS, LEFT_SKEW_COLS, UNIFORM_COLS, NORMAL_COLS]
    shape_msgs = [
        "Bimodal distributions usually mean the sensor is capturing **two distinct operating states** "
        "(e.g. healthy vs. degraded, or idle vs. active) rather than one continuous process. "
        "Since F3, F4, F6, F7 are also near-perfectly correlated (see Feature Redundancy), "
        "this is likely **one real signal** measured on different scales, not six.",
        "Right-skewed distributions (most values low, a long tail of large values) are typical of "
        "**event-driven or fault-related measurements** — normal operation clusters near zero, "
        "and rare spikes pull the tail out. These are the columns most worth an outlier deep-dive.",
        "F15 being left-skewed (most values high, a tail toward low values) suggests it measures "
        "something that is **usually near a ceiling** (e.g. a health or utilization score) and "
        "occasionally drops — the opposite pattern of the right-skewed group.",
        "F14 being roughly uniform across its range suggests it isn't a natural sensor reading at "
        "all — it looks more like a **cycling index, batch counter, or scheduled workload step**.",
        "F18's roughly bell-shaped distribution suggests it is a **stable, calibrated baseline "
        "measurement** with natural random variation around a machine-specific setpoint.",
    ]

    for tab, cols, msg in zip(tabs, groups, shape_msgs):
        with tab:
            n = len(cols)
            fig = go.Figure()
            rows_n = 1
            cols_n = n
            from plotly.subplots import make_subplots
            sp = make_subplots(rows=1, cols=n, subplot_titles=cols)
            for i, c in enumerate(cols):
                sp.add_trace(
                    go.Histogram(x=df[c], marker_color=PRIMARY, nbinsx=15, name=c, showlegend=False),
                    row=1, col=i + 1,
                )
            sp = style_fig(sp, height=300)
            sp.update_layout(bargap=0.05)
            st.plotly_chart(sp, use_container_width=True)
            insight_box(msg)

    st.markdown("### Boxplots — spotting outliers at a glance")
    why_box(
        "Same 16 columns, same order — boxplots make it immediate which features have points "
        "sitting far outside the whisker range, which is what motivates the dedicated outlier "
        "section next."
    )
    fig = make_subplots_box = None
    from plotly.subplots import make_subplots as _ms
    sp = _ms(rows=4, cols=4, subplot_titles=ALL_SHAPE_COLS)
    for i, c in enumerate(ALL_SHAPE_COLS):
        r, cc = divmod(i, 4)
        sp.add_trace(go.Box(y=df[c], marker_color=SECONDARY, name=c, showlegend=False), row=r + 1, col=cc + 1)
    sp = style_fig(sp, height=800)
    st.plotly_chart(sp, use_container_width=True)
    insight_box(
        "F8, F9, F10, F11, F12, F13 and F17 all show points well above the box — these are the "
        "right-skewed columns confirming outliers. F15 shows a few low outliers, consistent with "
        "its left skew. F3, F4, F5, F6, F7, F14, F18 and F19 show no extreme outliers, meaning "
        "their spread is well-behaved even though some of them are bimodal."
    )

# ============================================================================
# PAGE 5 — OUTLIERS BY MACHINE
# ============================================================================
elif page == "🚨 Outliers by Machine":
    st.title("🚨 Outlier Deep-Dive, Grouped by Machine (F1)")
    why_box(
        "We narrow down to the 6 columns that showed the strongest outliers in the previous "
        "section (F8, F9, F10, F11, F12, F13) and break each one down by F1 (machine group). "
        "This tells us **whether outliers are random noise or tied to a specific machine** — "
        "a much more actionable finding than 'this column has outliers'."
    )

    for c in OUTLIER_COLS:
        fig = go.Figure()
        for m in ["M1", "M2", "M3", "M4", "M5"]:
            sub = df[df["F1"] == m]
            fig.add_trace(go.Box(
                y=sub[c], name=m, marker_color=MACHINE_COLORS[m],
                boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(size=4, opacity=0.6),
            ))
        fig.update_layout(title=f"{c} by machine group", showlegend=False)
        fig = style_fig(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "**F8, F9, F10:** M2 produces the most extreme spikes in all three (up to ~250,000 in F8, "
        "~6,500 in F9, ~670 in F10), while M4 stays compressed near zero. This points to M2 as a "
        "machine with occasional severe events — worth flagging for maintenance or fault review. "
        "<br><br>"
        "**F11:** the pattern flips — M4 is the one with a huge spread (~160 to ~630) while the "
        "other four machines stay tightly bounded below ~180, suggesting F11 captures something "
        "M4-specific (e.g. a different load profile or a sensor calibration difference). "
        "<br><br>"
        "**F12 & F13:** identical relative shapes across all groups, both dominated by the same "
        "M2 elevation — strong evidence F12 and F13 are the **same underlying quantity on two "
        "scales**, both driven by the same M2 events seen in F8–F10."
    )

# ============================================================================
# PAGE 6 — FEATURE REDUNDANCY
# ============================================================================
elif page == "🔗 Feature Redundancy":
    st.title("🔗 Feature Redundancy — Correlation & Mutual Information")
    why_box(
        "All 16 numeric columns are compared against each other, in full, because redundancy "
        "detection only works if nothing is excluded — a hidden duplicate could be anywhere. "
        "We use **two** methods on purpose: correlation only catches straight-line relationships, "
        "while mutual information also catches curved / non-linear ones, so pairs that agree "
        "across both methods are the most confidently redundant."
    )

    corr = df[num_cols].corr()

    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale=[NEUTRAL, LIGHT, PRIMARY],
            zmin=-1, zmax=1, title="Correlation matrix (linear relationships)",
        )
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        with st.spinner("Computing mutual information matrix..."):
            mi_matrix = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
            for c in num_cols:
                scores = mutual_info_regression(df[num_cols], df[c], random_state=42)
                mi_matrix[c] = scores
        fig = px.imshow(
            mi_matrix, text_auto=".2f", color_continuous_scale=[LIGHT, SECONDARY, PRIMARY],
            title="Mutual information matrix (linear + non-linear)",
        )
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "F3, F4, F6, F7 form a near-perfect correlation block (r ≈ 1.00) confirmed by a high "
        "mutual-information block too — almost certainly the same signal on different scales. "
        "F12/F13 are essentially identical (r = 1.00, MI ≈ 3.55, the highest in the matrix). "
        "F8, F9, F10 move together tightly, and F11 shares meaningful MI with that group without "
        "being perfectly correlated — suggesting it's *related* but not a pure duplicate. "
        "F14–F15 and F18–F19 are strongly related in both matrices but not identical, which is "
        "exactly the signature of a **non-linear or stratified relationship** rather than a "
        "simple rescaling — confirmed visually in the next section. "
        "F17 stands out as the most **isolated** feature, sharing very little information with "
        "everything else — it likely captures something genuinely independent of the rest of the "
        "dataset."
    )

    st.markdown("### Quantifying the strongest pairs")
    why_box(
        "This table restates the same handful of pairs the heatmaps flagged, but as exact "
        "numbers — useful for backing up a claim like 'these are duplicates' with a precise R²."
    )
    rows = []
    for x, y in REDUNDANT_PAIRS:
        slope, intercept = np.polyfit(df[x], df[y], 1)
        r = df[x].corr(df[y])
        rows.append({"Feature X": x, "Feature Y": y, "r": round(r, 4), "R²": round(r**2, 4)})
    pair_df = pd.DataFrame(rows).sort_values("R²", ascending=False)
    st.dataframe(pair_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 7 — RELATIONSHIP SHAPES
# ============================================================================
elif page == "🔀 Relationship Shapes":
    st.title("🔀 Relationship Shapes — Scatter Plots")
    why_box(
        "We visualize exactly the pairs flagged as redundant in the previous section — seeing "
        "the *shape* of a relationship (straight line vs. curve vs. split lines) tells us "
        "**what kind** of redundancy we're dealing with, which a correlation number alone can't."
    )

    from plotly.subplots import make_subplots as _ms2
    sp = _ms2(rows=3, cols=3, subplot_titles=[f"{x} vs {y}" for x, y in REDUNDANT_PAIRS])
    for i, (x, y) in enumerate(REDUNDANT_PAIRS):
        r, c = divmod(i, 3)
        sp.add_trace(
            go.Scatter(x=df[x], y=df[y], mode="markers",
                       marker=dict(color=PRIMARY, opacity=0.6, size=6), showlegend=False),
            row=r + 1, col=c + 1,
        )
    sp = style_fig(sp, height=850)
    st.plotly_chart(sp, use_container_width=True)

    insight_box(
        "F3–F4, F3–F6, F3–F7, F8–F9, F8–F10, F9–F10 and F12–F13 all fall almost exactly on a "
        "straight line — the visual signature of a correlation near 1.00, confirming those are "
        "scaled duplicates. F14 vs F15 traces a clear **curve** rather than a line — a genuine "
        "non-linear relationship, not a duplicate. F18 vs F19 splits into **two parallel lines** "
        "— a strong sign that a hidden category is separating the data into two regimes."
    )

    st.markdown("### What explains the F18/F19 split and the F3 bimodal peaks?")
    why_box(
        "F16 (device category) is the only other categorical column, so it's the natural "
        "candidate to test as the hidden driver behind the F18/F19 split and the F3 bimodal "
        "shape — we color by F16 to check."
    )
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            df, x="F18", y="F19", color="F16", color_discrete_map=F16_COLORS,
            title="F18 vs F19, colored by F16",
        )
        fig = style_fig(fig, height=420, legend_title="F16")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.violin(
            df, x="F16", y="F3", color="F16", color_discrete_map=F16_COLORS,
            box=True, points=False, title="F3 distribution, split by F16",
        )
        fig = style_fig(fig, height=420, legend_title="F16")
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "F16 **does** explain the F18/F19 split: CPU rows sit strictly on the upper line, while "
        "MOB rows split across both the upper and a lower offset line — meaning F16 (or a factor "
        "correlated with it) drives part of that relationship. F16 does **not** explain F3's "
        "bimodal shape — MOB and CPU show the exact same two peaks (~95 and ~15), so whatever "
        "creates that split in F3 is a **different, still-hidden factor** — a good candidate for "
        "further investigation beyond this dataset."
    )

# ============================================================================
# PAGE 8 — PCA
# ============================================================================
elif page == "🧭 PCA — All Features at Once":
    st.title("🧭 Multivariate View — PCA")
    why_box(
        "Rather than looking at features two at a time, PCA compresses **all 16 numeric "
        "columns at once** into 2 new axes (PC1, PC2) that capture as much of the original "
        "spread as possible. This is the fastest way to check for redundancy and hidden "
        "grouping across the entire dataset in a single plot."
    )

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    df_pca["F16"] = df["F16"].values
    df_pca["F1"] = df["F1"].values

    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><h2 style="color:{PRIMARY}">{var1:.1f}%</h2>PC1 variance explained</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h2 style="color:{SECONDARY}">{var2:.1f}%</h2>PC2 variance explained</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h2 style="color:{ACCENT}">{var1+var2:.1f}%</h2>combined</div>', unsafe_allow_html=True)

    color_by = st.radio("Color points by:", ["F16", "F1"], horizontal=True)
    palette = F16_COLORS if color_by == "F16" else MACHINE_COLORS
    fig = px.scatter(
        df_pca, x="PC1", y="PC2", color=color_by, color_discrete_map=palette,
        title=f"PCA — all features compressed to 2D, colored by {color_by}",
        opacity=0.75,
    )
    fig = style_fig(fig, height=500, legend_title=color_by)
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        f"PC1 and PC2 together already explain **{var1+var2:.1f}%** of the variation across all "
        "16 numeric columns — with only 16% of the original dimensionality (2 out of 16 axes). "
        "That confirms what the correlation and MI matrices already suggested: a large chunk of "
        "these columns are redundant, scaled copies of a smaller set of true underlying signals. "
        "When colored by F16, CPU and MOB points separate into visibly different regions — F16 "
        "is a **real, structural difference** in the data, not an arbitrary label. Try switching "
        "the color to F1 to see whether individual machines separate the same way."
    )

# ============================================================================
# PAGE 9 — F2 AS TIME
# ============================================================================
elif page == "⏱️ F2 as Time":
    st.title("⏱️ Treating F2 as a Time / Sequence Index")
    why_box(
        "F2 looks like a step label (T1, T2, T3…) rather than a category. We test the "
        "hypothesis that it encodes **sequence/time** by extracting its numeric part and "
        "tracking three representative features — one from each shape family found earlier "
        "(F3: bimodal, F14: uniform, F18: normal) — across that sequence, per machine."
    )

    features_to_track = ["F3", "F14", "F18"]
    df_sorted = df.sort_values(by=["F1", "F2_seq"])

    for feat in features_to_track:
        fig = go.Figure()
        for m in ["M1", "M2", "M3", "M4", "M5"]:
            sub = df_sorted[df_sorted["F1"] == m]
            fig.add_trace(go.Scatter(
                x=sub["F2_seq"], y=sub[feat], mode="lines+markers",
                name=m, line=dict(color=MACHINE_COLORS[m], width=2),
                marker=dict(size=5),
            ))
        fig.update_layout(title=f"{feat} across sequence steps (F2), by machine",
                          xaxis_title="Sequence index (F2)", yaxis_title=feat)
        fig = style_fig(fig, height=380, legend_title="F1")
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        "**F3 (system status):** M1 and M2 stay consistently healthy around ~95 throughout the "
        "sequence. M3 and M5 drop sharply and permanently around step 11 — consistent with a "
        "**permanent failure or shutdown event**. M4 oscillates — periodic dips and recoveries, "
        "more consistent with **intermittent faults** than a single failure. "
        "<br><br>"
        "**F14 (batch workload):** every machine follows the *same* 4–5 step wave, in sync. "
        "Since this pattern is identical across independent machines, it's most likely driven by "
        "a **shared external process or scheduling cycle** rather than anything machine-specific. "
        "<br><br>"
        "**F18 (machine baseline):** each machine sits at its own fixed offset (M5 highest, M3 "
        "lowest) and all of them spike in sync with F14's workload peaks — suggesting F18 is a "
        "**per-machine calibration/load baseline** that responds to the same shared workload "
        "driving F14. "
        "<br><br>"
        "Put together, this section is the strongest evidence that F2 genuinely encodes time, "
        "and that the machines share an external workload cycle while still failing or degrading "
        "independently."
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built from raw EDA notebook · consistent color scheme applied across all charts.")