"""
Anonymized Sensor Dataset - Exploratory Data Analysis Dashboard
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
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sensor Data EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"

# ----------------------------------------------------------------------------
# CHART COLOR SCHEME - fixed, does not change with theme. This is what
# keeps every chart in the dashboard reading the same way.
# ----------------------------------------------------------------------------
PRIMARY = "#1F4E79"      # deep steel blue   -> main / default series
SECONDARY = "#2A9D8F"    # teal              -> secondary series / positive
ACCENT = "#E76F51"       # warm coral        -> outliers, warnings, "CPU"
NEUTRAL = "#8C97A5"      # slate gray        -> de-emphasized elements
LIGHT = "#F2F6F8"        # pale blue-gray    -> heatmap midpoint

MACHINE_COLORS = {
    "M1": "#1F4E79",
    "M2": "#2E6F95",
    "M3": "#2A9D8F",
    "M4": "#6FBFAE",
    "M5": "#B7DED2",
}
F16_COLORS = {"MOB": PRIMARY, "CPU": ACCENT}

# ----------------------------------------------------------------------------
# UI THEME - controls page chrome only (background, panels, nav, fonts).
# Two palettes, switched at runtime by the toggle button in the sidebar.
# ----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "ink": "#EAF2F1",
        "muted": "#9FB3AE",
        "glass": "rgba(255, 255, 255, 0.05)",
        "glass_strong": "rgba(255, 255, 255, 0.09)",
        "glass_hover": "rgba(255, 255, 255, 0.15)",
        "border": "rgba(255, 255, 255, 0.14)",
        "accent": "#3FB6A8",
        "accent_warm": "#F0895F",
        "accent2": "#8FD3C7",
        "base": "#0D1417",
        "plot_grid": "rgba(255, 255, 255, 0.08)",
        "toggle_label": "Switch to light theme",
    },
    "light": {
        "ink": "#1A1A1A",
        "muted": "#6B7686",
        "glass": "rgba(255, 255, 255, 0.55)",
        "glass_strong": "rgba(255, 255, 255, 0.75)",
        "glass_hover": "rgba(255, 255, 255, 0.92)",
        "border": "rgba(31, 78, 121, 0.16)",
        "accent": "#2A9D8F",
        "accent_warm": "#E76F51",
        "accent2": "#6FBFAE",
        "base": "#F2F6F8",
        "plot_grid": "rgba(31, 78, 121, 0.12)",
        "toggle_label": "Switch to dark theme",
    },
}
theme = THEMES[st.session_state.theme]

# ----------------------------------------------------------------------------
# GLOBAL CSS
# ----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    :root {{
        --ink: {theme['ink']};
        --muted: {theme['muted']};
        --glass: {theme['glass']};
        --glass-strong: {theme['glass_strong']};
        --glass-hover: {theme['glass_hover']};
        --border: {theme['border']};
        --accent: {theme['accent']};
        --accent-warm: {theme['accent_warm']};
        --base: {theme['base']};
    }}

    header[data-testid="stHeader"] {{ background: transparent !important; }}

    .stApp {{
        color: var(--ink);
        background: var(--base);
        background-image:
            radial-gradient(ellipse 55% 40% at 80% -5%, color-mix(in srgb, var(--accent) 22%, transparent), transparent 60%),
            radial-gradient(ellipse 45% 35% at 95% 8%, color-mix(in srgb, var(--accent-warm) 18%, transparent), transparent 65%),
            radial-gradient(circle at 8% 90%, color-mix(in srgb, var(--accent) 12%, transparent), transparent 35%);
        transition: background 0.3s ease;
    }}

    .block-container {{ position: relative; z-index: 1; padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }}
    h1, h2, h3, p, label, [data-testid="stCaptionContainer"], [data-testid="stMarkdownContainer"] li {{ color: var(--ink); }}
    h1 {{ letter-spacing: 0.01em; font-weight: 800; }}

    section[data-testid="stSidebar"] {{
        background: color-mix(in srgb, var(--base) 96%, black 4%) !important;
        border-right: 1px solid var(--border);
        width: 17rem !important;
        min-width: 17rem !important;
    }}
    [data-testid="stSidebarUserContent"] {{ padding: 0.6rem 0.7rem !important; }}

    .side-brand {{
        font-weight: 800;
        font-size: 1.05rem;
        color: var(--ink);
        padding: 0.3rem 0.35rem 1rem 0.35rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 0.8rem;
    }}

    [data-testid="stSidebar"] .stButton button {{
        width: 100% !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--muted) !important;
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.8rem !important;
        margin-bottom: 0.25rem;
        font-weight: 600;
        font-size: 0.92rem;
        transition: all 0.2s ease;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background: var(--glass-hover) !important;
        color: var(--ink) !important;
        border-color: var(--border) !important;
    }}
    [data-testid="stSidebar"] .stButton button[kind="primary"] {{
        background: var(--accent-warm) !important;
        color: #1E2029 !important;
        border-color: transparent !important;
        box-shadow: 0 0 16px color-mix(in srgb, var(--accent-warm) 35%, transparent);
    }}
    .nav-spacer {{ height: 0.7rem; }}

    .metric-card, .why-box, .insight-box {{
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 14px;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    .metric-card {{ padding: 14px; text-align: center; }}
    .metric-card:hover {{ transform: translateY(-2px); border-color: var(--accent); }}
    .insight-box {{ border-left: 4px solid var(--accent); padding: 14px 18px; margin: 12px 0; font-size: 0.95rem; }}
    .why-box {{ border-style: dashed; padding: 12px 16px; margin: 10px 0 18px 0; font-size: 0.9rem; color: var(--muted); }}
    .box-label {{ color: var(--accent); font-size: 0.76rem; letter-spacing: 0.05em; text-transform: uppercase; }}
    .why-box .box-label {{ color: var(--muted); }}

    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {{
        background: var(--glass);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.4rem;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.16);
        backdrop-filter: blur(10px);
        transition: border-color 0.25s ease;
    }}
    [data-testid="stPlotlyChart"]:hover {{ border-color: var(--accent); }}

    [data-testid="stTabs"] button[role="tab"] {{ color: var(--muted) !important; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: var(--ink) !important;
        border-bottom-color: var(--accent) !important;
    }}

    [data-testid="stRadio"] label {{ color: var(--ink) !important; }}
    </style>
    """,
    unsafe_allow_html=True,
)

def render_box(label, points, style_class):
    if isinstance(points, str):
        points = [points]
    items = "".join(f"<li style='margin-bottom:4px;'>{p}</li>" for p in points)
    st.markdown(
        f'<div class="{style_class}"><div class="box-label">{label}</div>'
        f'<ul style="margin:6px 0 0 18px;padding:0;">{items}</ul></div>',
        unsafe_allow_html=True,
    )

def why_box(points):
    render_box("Why this subset", points, "why-box")

def insight_box(points):
    render_box("What this could mean", points, "insight-box")

def style_fig(fig, height=None, legend_title=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Helvetica, Arial, sans-serif", color=theme["ink"]),
        title_font=dict(size=17, color=theme["accent"]),
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(font=dict(color=theme["muted"])),
    )
    fig.update_xaxes(
        gridcolor=theme["plot_grid"], zerolinecolor=theme["plot_grid"],
        tickfont=dict(color=theme["muted"]), title_font=dict(color=theme["ink"]),
    )
    fig.update_yaxes(
        gridcolor=theme["plot_grid"], zerolinecolor=theme["plot_grid"],
        tickfont=dict(color=theme["muted"]), title_font=dict(color=theme["ink"]),
    )
    if height:
        fig.update_layout(height=height)
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["F2_seq"] = df["F2"].str.extract(r"(\d+)").astype(int)
    return df

DATA_PATH = "data/data.csv"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("Could not find data/data.csv. Make sure the dataset is included in the app folder.")
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
    "Overview",
    "Data Quality",
    "Categorical Breakdown",
    "Numeric Distributions",
    "Outliers by Machine",
    "Feature Redundancy",
    "Relationship Shapes",
    "PCA - All Features at Once",
    "Overall Hypothesis",
]

with st.sidebar:
    if st.button(theme["toggle_label"], key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown('<div class="side-brand">EDA Dashboard</div>', unsafe_allow_html=True)

    for label in PAGES:
        is_active = st.session_state.active_page == label
        if st.button(label, key=f"nav_{label}",
                     type="primary" if is_active else "secondary",
                     use_container_width=True):
            st.session_state.active_page = label
            st.rerun()

    st.markdown('<div class="nav-spacer"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-card">
        <b>{df.shape[0]}</b> rows &nbsp;&middot;&nbsp; <b>{df.shape[1]-1}</b> columns<br>
        <span style="color:{theme['muted']}; font-size:0.8rem;">F1 groups: M1-M5 &nbsp;|&nbsp; F16: MOB / CPU</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

page = st.session_state.active_page

# ============================================================================
# PAGE 1 - OVERVIEW
# ============================================================================
if page == "Overview":
    st.title("Anonymized Sensor Dataset - Exploratory Analysis")
    st.markdown(
        "This dashboard walks through an EDA of an anonymized dataset with 19 features "
        "(F1-F19), collected across 5 machines (M1-M5) and split into two device categories, "
        "MOB and CPU. The goal was to understand the shape of each feature, find redundant "
        "or duplicated signals, and see whether the anonymized columns hide a real "
        "underlying story."
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Rows", "Numeric features", "Categorical features", "Missing values"],
        [df.shape[0], len(num_cols), len(cat_cols), int(df.isna().sum().sum())],
    ):
        col.markdown(
            f'<div class="metric-card"><h2 style="margin:0;color:{theme["accent"]}">{value}</h2>'
            f'<span style="color:{theme["muted"]}">{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### How to read this dashboard")
    st.markdown(
        "- Each section opens with a short note on why that subset of features is being looked at\n"
        "- The charts all follow the same color scheme, so a color means the same thing everywhere\n"
        "- Each section closes with a plain-language read on what the pattern could mean"
    )
    st.markdown("Color key used throughout:")
    legend_cols = st.columns(7)
    swatches = [("M1", MACHINE_COLORS["M1"]), ("M2", MACHINE_COLORS["M2"]), ("M3", MACHINE_COLORS["M3"]),
                ("M4", MACHINE_COLORS["M4"]), ("M5", MACHINE_COLORS["M5"]),
                ("MOB", F16_COLORS["MOB"]), ("CPU", F16_COLORS["CPU"])]
    for col, (label, color) in zip(legend_cols, swatches):
        col.markdown(
            f'<div style="background:{color};color:white;border-radius:8px;'
            f'padding:8px;text-align:center;font-size:0.85rem;">{label}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Raw data preview")
    st.dataframe(df.drop(columns=["F2_seq"]).head(10), use_container_width=True)

# ============================================================================
# PAGE 2 - DATA QUALITY
# ============================================================================
elif page == "Data Quality":
    st.title("Data Quality Check")
    why_box([
        "Before trusting any pattern in the data, it's worth confirming there are no missing "
        "values or duplicate rows that could distort distributions, correlations, or the PCA later on."
    ])

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
            <h1 style="color:{theme['accent']};margin:0;">0</h1>
            <span>total missing values</span>
            <hr style="border-color:{theme['border']};">
            <h1 style="color:{theme['accent']};margin:0;">{dup_count}</h1>
            <span>duplicate rows</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    insight_box([
        "The data is clean, nothing missing and nothing duplicated",
        "That means the patterns found later on are real, not mistakes in the data",
    ])

# ============================================================================
# PAGE 3 - CATEGORICAL BREAKDOWN
# ============================================================================
elif page == "Categorical Breakdown":
    st.title("Categorical Breakdown - F1 and F16")
    why_box([
        "F1 and F16 are the only two categorical columns in the dataset, and every grouping "
        "used later (outlier analysis, stratified scatter plots, PCA coloring) depends on them, "
        "so it makes sense to check their balance first."
    ])

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

    insight_box([
        "F1 is split evenly, 26 rows for each of the 5 machines, which looks planned rather than random",
        "F16 is not even, 105 MOB rows against 25 CPU rows",
        "Worth remembering later on, since any MOB vs CPU comparison is resting on a much smaller CPU sample",
    ])

# ============================================================================
# PAGE 4 - NUMERIC DISTRIBUTIONS
# ============================================================================
elif page == "Numeric Distributions":
    st.title("Numeric Distributions")
    why_box([
        "The 16 numeric columns are grouped by the shape of their distribution rather than "
        "shown in column order, since features that share a shape often share a cause."
    ])

    tabs = st.tabs([
        "Bimodal (F3,F4,F5,F6,F7,F19)",
        "Right-skewed (F8,F9,F10,F11,F12,F13,F17)",
        "Left-skewed (F15)",
        "Uniform (F14)",
        "Normal (F18)",
    ])
    groups = [BIMODAL_COLS, RIGHT_SKEW_COLS, LEFT_SKEW_COLS, UNIFORM_COLS, NORMAL_COLS]
    shape_msgs = [
        [
            "Two humps usually means two situations rather than one steady process, like "
            "healthy vs broken, or on vs off",
            "F3, F4, F6 and F7 all move together, so they are probably one real signal "
            "repeated on different scales, not six different things",
        ],
        [
            "Most values sit low, with a few high spikes",
            "This is what you would expect from rare events or faults, normal most of the "
            "time, then something goes wrong and the number jumps",
            "These are the columns worth checking for outliers",
        ],
        [
            "F15 is the opposite, usually high with occasional low dips",
            "This looks like a health or usage score that is normally near its max and "
            "drops when something is off",
        ],
        [
            "F14 is spread out evenly with no real peak",
            "That is unusual for a sensor reading, it looks more like a counter or a "
            "repeating schedule than something being measured",
        ],
        [
            "F18 forms a clean bell curve",
            "That usually means it is a stable baseline reading with normal small ups and "
            "downs, not something reacting to events",
        ],
    ]

    for tab, cols, msg in zip(tabs, groups, shape_msgs):
        with tab:
            n = len(cols)
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

    st.markdown("### Boxplots - spotting outliers at a glance")
    why_box([
        "Same 16 columns, same order. Boxplots make it easy to see which features have "
        "points sitting far outside the normal range, which is what the outlier section "
        "that follows digs into."
    ])
    sp = make_subplots(rows=4, cols=4, subplot_titles=ALL_SHAPE_COLS)
    for i, c in enumerate(ALL_SHAPE_COLS):
        r, cc = divmod(i, 4)
        sp.add_trace(go.Box(y=df[c], marker_color=SECONDARY, name=c, showlegend=False), row=r + 1, col=cc + 1)
    sp = style_fig(sp, height=800)
    st.plotly_chart(sp, use_container_width=True)
    insight_box([
        "F8, F9, F10, F11, F12, F13 and F17 all have points shooting up above the box, "
        "these are the columns with real outliers",
        "F15 has a few low outliers, which matches its shape",
        "F3, F4, F5, F6, F7, F14, F18 and F19 have no extreme values, even the ones with two humps",
    ])

# ============================================================================
# PAGE 5 - OUTLIERS BY MACHINE
# ============================================================================
elif page == "Outliers by Machine":
    st.title("Outlier Deep-Dive, Grouped by Machine (F1)")
    why_box([
        "This narrows down to the 6 columns that showed the strongest outliers in the "
        "previous section, and breaks each one down by machine, to see whether the "
        "outliers are random noise or tied to a specific machine."
    ])

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

    insight_box([
        "F8, F9 and F10: machine M2 has the biggest spikes in all three, while M4 stays low "
        "and steady the whole time, pointing to M2 having occasional serious problems",
        "F11: the pattern flips, M4 is the one that swings wildly while the other four "
        "machines barely move, so F11 seems to be picking up something specific to M4",
        "F12 and F13: these two look almost identical, and both spike with M2, which is "
        "strong evidence they are the same measurement written down twice on different scales",
    ])

# ============================================================================
# PAGE 6 - FEATURE REDUNDANCY
# ============================================================================
elif page == "Feature Redundancy":
    st.title("Feature Redundancy - Correlation and Mutual Information")
    why_box([
        "All 16 numeric columns are compared against each other in full, since a hidden "
        "duplicate could be anywhere. Two methods are used on purpose: correlation only "
        "catches straight-line relationships, mutual information also catches curved ones, "
        "so pairs that agree across both are the most confidently redundant."
    ])

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
            title="Mutual information matrix (linear and non-linear)",
        )
        fig = style_fig(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

    insight_box([
        "F3, F4, F6 and F7 move together almost perfectly, likely the same signal recorded differently",
        "F12 and F13 are basically identical twins, the strongest match in the whole matrix",
        "F8, F9 and F10 also travel together, and F11 is related to them but not a duplicate",
        "F14/F15 and F18/F19 are connected but not copies of each other, something a bit "
        "more complex links them, shown clearly in the next section",
        "F17 stands alone, it barely relates to anything else, so it is probably measuring "
        "something genuinely different from the rest of the dataset",
    ])

    st.markdown("### Quantifying the strongest pairs")
    why_box([
        "This table restates the same handful of pairs the heatmaps flagged, but as exact "
        "numbers, useful for backing up a claim like these are duplicates with a precise figure."
    ])
    rows = []
    for x, y in REDUNDANT_PAIRS:
        slope, intercept = np.polyfit(df[x], df[y], 1)
        r = df[x].corr(df[y])
        rows.append({"Feature X": x, "Feature Y": y, "r": round(r, 4), "R2": round(r**2, 4)})
    pair_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
    st.dataframe(pair_df, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE 7 - RELATIONSHIP SHAPES
# ============================================================================
elif page == "Relationship Shapes":
    st.title("Relationship Shapes - Scatter Plots")
    why_box([
        "This visualizes the pairs flagged as redundant in the previous section, since the "
        "shape of a relationship, straight line vs curve vs split lines, tells us what kind "
        "of redundancy is at play, which a correlation number alone can't."
    ])

    sp = make_subplots(rows=3, cols=3, subplot_titles=[f"{x} vs {y}" for x, y in REDUNDANT_PAIRS])
    for i, (x, y) in enumerate(REDUNDANT_PAIRS):
        r, c = divmod(i, 3)
        sp.add_trace(
            go.Scatter(x=df[x], y=df[y], mode="markers",
                       marker=dict(color=PRIMARY, opacity=0.6, size=6), showlegend=False),
            row=r + 1, col=c + 1,
        )
    sp = style_fig(sp, height=850)
    st.plotly_chart(sp, use_container_width=True)

    insight_box([
        "F3-F4, F3-F6, F3-F7, F8-F9, F8-F10, F9-F10 and F12-F13 all form clean straight "
        "lines, confirming they are duplicates of each other",
        "F14 vs F15 forms a curve, not a straight line, a real relationship but not a simple copy",
        "F18 vs F19 splits into two separate parallel lines, usually a sign that a hidden "
        "category is splitting the data into two groups",
    ])

    st.markdown("### What explains the F18/F19 split")
    why_box([
        "F16 is the only other categorical column, so it's the natural candidate to test as "
        "the hidden driver behind the F18/F19 split."
    ])
    fig = px.scatter(
        df, x="F18", y="F19", color="F16", color_discrete_map=F16_COLORS,
        title="F18 vs F19, colored by F16",
    )
    fig = style_fig(fig, height=420, legend_title="F16")
    st.plotly_chart(fig, use_container_width=True)

    insight_box([
        "F16 only partly explains the split, all CPU rows sit on one line, but MOB rows "
        "appear on both lines, so F16 alone isn't the full story",
    ])

# ============================================================================
# PAGE 8 - PCA
# ============================================================================
elif page == "PCA - All Features at Once":
    st.title("Multivariate View - PCA")
    why_box([
        "PCA combines all 16 columns into 2 new axes that capture most of the spread, a "
        "fast way to check for redundancy across everything at once, not just two columns at a time."
    ])

    X_scaled = StandardScaler().fit_transform(df[num_cols])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    df_pca["F16"] = df["F16"].values
    df_pca["F1"] = df["F1"].values

    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><h2 style="color:{theme["accent"]}">{var1:.1f}%</h2>PC1 variance explained</div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h2 style="color:{theme["accent2"]}">{var2:.1f}%</h2>PC2 variance explained</div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h2 style="color:{theme["accent_warm"]}">{var1+var2:.1f}%</h2>combined</div>', unsafe_allow_html=True)

    color_by = st.radio("Color points by:", ["F16", "F1"], horizontal=True)
    palette = F16_COLORS if color_by == "F16" else MACHINE_COLORS
    fig = px.scatter(
        df_pca, x="PC1", y="PC2", color=color_by, color_discrete_map=palette,
        title=f"PCA, all features compressed to 2D, colored by {color_by}",
        opacity=0.75,
    )
    fig = style_fig(fig, height=500, legend_title=color_by)
    st.plotly_chart(fig, use_container_width=True)

    loadings = pd.DataFrame(pca.components_.T, index=num_cols, columns=["PC1", "PC2"])

    def top_features(col, n=4):
        s = loadings[col]
        top_idx = s.abs().sort_values(ascending=False).head(n).index
        ordered = [c for c in num_cols if c in top_idx]
        pos = [f for f in ordered if s[f] > 0]
        neg = [f for f in ordered if s[f] < 0]
        return pos, neg

    pc1_pos, pc1_neg = top_features("PC1", n=5)
    pc2_pos, pc2_neg = top_features("PC2", n=6)

    def fmt(features):
        return ", ".join(features)

    pc1_line = f"PC1 ({var1:.1f}% of the spread): almost entirely {fmt(pc1_pos)}"
    if pc1_neg:
        pc1_line += f", pulled the other way by {fmt(pc1_neg)}"
    pc2_line = f"PC2 ({var2:.1f}% of the spread): {fmt(pc2_pos)} move one way"
    if pc2_neg:
        pc2_line += f", while {fmt(pc2_neg)} move the opposite way"

    st.markdown("### What is actually driving PC1 and PC2")
    why_box([
        "Correlation heatmaps compare two columns at a time. This shows which columns "
        "combine into each axis using all 16 at once."
    ])
    st.markdown(f"- {pc1_line}\n- {pc2_line}")

    insight_box([
        f"Two axes capture {var1+var2:.1f}% of everything in the data",
        "That means a lot of these columns repeat the same information on different scales",
        "PC2 mixes F14 and F15 against F12, F13, F18 and F19, a pattern the pairwise heatmaps don't show",
        "CPU tends to sit lower on PC2 than MOB, but the two overlap a lot, they don't split into separate clusters",
        "Switch to F1 to see if machines separate instead",
    ])

# ============================================================================
# PAGE 9 - OVERALL HYPOTHESIS
# ============================================================================
elif page == "Overall Hypothesis":
    st.title("Putting It All Together - What Is This Data Actually Tracking")
    why_box([
        "This section lines up every earlier finding to guess what the dataset actually "
        "measures. It only uses what the graphs in this dashboard actually show, and it's "
        "a hypothesis, not a confirmed fact."
    ])

    st.markdown("### Which variables are related")
    st.markdown(
        "- F3, F4, F6, F7: nearly identical (r ~ 1.00), same signal on different scales\n"
        "- F8, F9, F10: nearly identical (r ~ 1.00), same signal on different scales\n"
        "- F12, F13: identical (r = 1.00), same signal on different scales\n"
        "- F11: weakly tied to F8/F9/F10 (r ~ -0.28), related but not a duplicate\n"
        "- F14, F15: related but non-linear (r = 0.92, a curve rather than a line)\n"
        "- F18, F19: related but not identical (r = 0.93)\n"
        "- F17: weak relation to everything else, the closest is F14 at r = -0.35, the "
        "most independent column in the dataset"
    )

    st.markdown("### F1, F2 and F16")
    st.markdown(
        "- F1: machine ID, 5 machines, 26 rows each\n"
        "- F2: looks like a step or reading counter (labels T1 to T26), not a measurement\n"
        "- F16: a device or mode label (MOB / CPU). It only partly explains the F18 vs F19 "
        "split, CPU rows sit on one line, but MOB rows appear on both lines, so F16 alone "
        "doesn't fully explain it"
    )

    st.markdown("### Short version")
    st.markdown(
        f"Of the {len(num_cols)} numeric columns, most collapse into the handful of related "
        "groups listed above rather than being independent measurements. F1 and F2 look like "
        "identifiers, not sensor readings. Machine M2 stands out with the largest fault "
        "spikes (F8, F9, F10, F12, F13), and machine M4 stands out with the widest spread in "
        "F11. This is consistent with a machine monitoring log, though the real column names "
        "and what they represent stay unconfirmed."
    )

    st.markdown("### Where this is uncertain")
    st.markdown(
        "- The real column names aren't known, so labels like fault or status are best-fit guesses\n"
        "- F16 explains only part of the F18/F19 split, the rest is still unexplained\n"
        "- F15 and F17 don't fit into any of the related groups above\n"
        "- Only 130 rows across 5 machines, a small sample to generalize from"
    )

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Built from raw EDA notebook. Consistent color scheme applied across all charts.")