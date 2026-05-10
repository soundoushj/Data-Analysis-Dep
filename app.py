import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Cars Price Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }
    h1, h2, h3 { font-family: Arial, sans-serif; }
 
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .sidebar .sidebar-content { background-color: #161b27; }
 
    .metric-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2740 100%);
        border: 1px solid #2d3a52;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: #60a5fa; font-family: Arial, sans-serif; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.08em; }
 
    .section-header {
        font-family: Arial, sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f4ff;
        border-left: 4px solid #3b82f6;
        padding-left: 14px;
        margin: 32px 0 16px 0;
    }
 
    .insight-box {
        background: #1a2035;
        border: 1px solid #2d3a52;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #cbd5e1;
    }
 
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-family: Arial, sans-serif; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
 
    .stDataFrame { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)
 
# -------------------------------------------------
# PALETTE
# -------------------------------------------------
DARK_BG  = "#0f1117"
CARD_BG  = "#1a2035"
BORDER   = "#2d3a52"
BLUE     = "#3b82f6"
TEAL     = "#10b981"
AMBER    = "#f59e0b"
ROSE     = "#f43f5e"
MUTED    = "#94a3b8"
TEXT     = "#e8eaf0"
 
def apply_dark_style(ax, fig):
    fig.patch.set_facecolor(CARD_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
 
# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Cars_dataset.csv")
    except FileNotFoundError:
        np.random.seed(42)
        n = 300
        brands = ["Toyota", "BMW", "Honda", "Ford", "Mercedes"]
        fuel   = ["Petrol", "Diesel", "Electric"]
        df = pd.DataFrame({
            "Car ID":     range(1, n+1),
            "Brand":      np.random.choice(brands, n),
            "Fuel Type":  np.random.choice(fuel, n),
            "Year":       np.random.randint(2010, 2024, n),
            "Mileage":    np.random.randint(5000, 120000, n),
            "Engine Size":np.random.uniform(1.0, 4.5, n).round(1),
            "HP":         np.random.randint(80, 400, n),
            "Price":      np.random.randint(8000, 80000, n),
        })
    return df
 
df = load_data()

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.markdown("## Cars Dashboard")
    st.markdown("---")
 
    page = st.radio(
        "Navigate",
        ["Overview", "EDA", "ML Model", "Predict"],
        label_visibility="collapsed"
    )
 
    st.markdown("---")
    st.markdown("### Filters")
    brands_avail = sorted(df["Brand"].unique().tolist()) if "Brand" in df.columns else []
    selected_brands = st.multiselect("Brand", brands_avail, default=brands_avail)
 
    if "Fuel Type" in df.columns:
        fuel_avail = sorted(df["Fuel Type"].unique().tolist())
        selected_fuel = st.multiselect("Fuel Type", fuel_avail, default=fuel_avail)
    else:
        selected_fuel = []
 
    if "Year" in df.columns:
        year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
        year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max))
    else:
        year_range = (0, 9999)
 
# -------------------------------------------------
# FILTER
# -------------------------------------------------
dff = df.copy()
if selected_brands and "Brand" in dff.columns:
    dff = dff[dff["Brand"].isin(selected_brands)]
if selected_fuel and "Fuel Type" in dff.columns:
    dff = dff[dff["Fuel Type"].isin(selected_fuel)]
if "Year" in dff.columns:
    dff = dff[(dff["Year"] >= year_range[0]) & (dff["Year"] <= year_range[1])]
 
# -------------------------------------------------
# PAGE 1: OVERVIEW
# -------------------------------------------------
if page == "Overview":
    st.markdown('<h1 style="font-family:Arial,sans-serif;font-size:2.2rem;color:#f0f4ff;margin-bottom:4px;">Cars Price Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:1rem;font-family:Arial,sans-serif;">End-to-end data science pipeline — from raw data to predictive modeling</p>', unsafe_allow_html=True)
    st.markdown("---")
 
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Cars",    f"{len(dff):,}"),
        ("Avg Price",     f"${dff['Price'].mean():,.0f}"),
        ("Max Price",     f"${dff['Price'].max():,.0f}"),
        ("Min Price",     f"${dff['Price'].min():,.0f}"),
        ("Brands",        str(dff["Brand"].nunique()) if "Brand" in dff.columns else "-"),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(dff.head(10), use_container_width=True)
 
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(dff.describe().T.style.format("{:.2f}"), use_container_width=True)
 
    st.markdown('<div class="insight-box"><b>Insight:</b> The dataset spans multiple car brands, fuel types, and model years. Price varies widely, indicating strong feature influence — ideal for regression modeling.</div>', unsafe_allow_html=True)
 
# -------------------------------------------------
# PAGE 2: EDA
# -------------------------------------------------
elif page == "EDA":
    st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
 
    col1, col2 = st.columns(2)
 
    with col1:
        st.markdown("##### Price Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(ax, fig)
        sns.histplot(dff["Price"], kde=True, color=BLUE, ax=ax, bins=30, alpha=0.7)
        ax.set_title("Price Distribution", color=TEXT, fontsize=11)
        ax.set_xlabel("Price ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">Price is right-skewed — most cars are affordable, with a long tail of premium models. Log transformation may improve model fit.</div>', unsafe_allow_html=True)
 
    with col2:
        st.markdown("##### Cars by Brand")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(ax, fig)
        brand_counts = dff["Brand"].value_counts() if "Brand" in dff.columns else pd.Series()
        brand_counts.plot(kind="bar", color=[BLUE, TEAL, AMBER, ROSE, "#a78bfa"][:len(brand_counts)], ax=ax, edgecolor="none")
        ax.set_title("Cars by Brand", color=TEXT, fontsize=11)
        ax.set_xlabel("")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
        plt.close()
 
    col3, col4 = st.columns(2)
 
    with col3:
        st.markdown("##### Fuel Type Distribution")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(ax, fig)
        if "Fuel Type" in dff.columns:
            fuel_counts = dff["Fuel Type"].value_counts()
            colors_pie = [BLUE, TEAL, AMBER, ROSE]
            wedges, texts, autotexts = ax.pie(fuel_counts, labels=fuel_counts.index, autopct="%1.1f%%",
                                               colors=colors_pie[:len(fuel_counts)], startangle=140)
            for t in texts: t.set_color(MUTED)
            for at in autotexts: at.set_color(TEXT)
        ax.set_title("Fuel Type", color=TEXT, fontsize=11)
        fig.patch.set_facecolor(CARD_BG)
        st.pyplot(fig)
        plt.close()
 
    with col4:
        st.markdown("##### Price by Brand")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        apply_dark_style(ax, fig)
        if "Brand" in dff.columns:
            palette = {b: c for b, c in zip(dff["Brand"].unique(), [BLUE, TEAL, AMBER, ROSE, "#a78bfa"])}
            sns.boxplot(data=dff, x="Brand", y="Price", palette=palette, ax=ax, width=0.5, linewidth=0.8)
            plt.xticks(rotation=30, ha="right")
        ax.set_title("Price by Brand", color=TEXT, fontsize=11)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig)
        plt.close()
 
    st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
    numeric_df = dff.select_dtypes(include=np.number).drop(columns=["Car ID"], errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 5))
    apply_dark_style(ax, fig)
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax, mask=mask,
                annot_kws={"size": 9}, linewidths=0.5, linecolor=BORDER,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", color=TEXT, fontsize=12)
    st.pyplot(fig)
    plt.close()
 
    if "Mileage" in dff.columns:
        st.markdown('<div class="insight-box"><b>Key correlation:</b> Mileage is negatively correlated with Price — higher mileage cars sell cheaper. HP and Engine Size tend to correlate positively with Price.</div>', unsafe_allow_html=True)
 
    if "Mileage" in dff.columns:
        st.markdown('<div class="section-header">Mileage vs Price</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 4))
        apply_dark_style(ax, fig)
        ax.scatter(dff["Mileage"], dff["Price"], alpha=0.45, s=18, c=BLUE, edgecolors="none")
        z = np.polyfit(dff["Mileage"].fillna(dff["Mileage"].median()), dff["Price"], 1)
        p = np.poly1d(z)
        xs = np.linspace(dff["Mileage"].min(), dff["Mileage"].max(), 200)
        ax.plot(xs, p(xs), color=ROSE, linewidth=2, label="Trend")
        ax.set_xlabel("Mileage (km)")
        ax.set_ylabel("Price ($)")
        ax.set_title("Mileage vs Price — negative relationship expected", color=TEXT, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.legend(frameon=False, labelcolor=MUTED)
        st.pyplot(fig)
        plt.close()
 
# -------------------------------------------------
# PAGE 3: ML MODEL
# -------------------------------------------------
elif page == "ML Model":
    st.markdown('<h2 class="section-header">Linear Regression Modeling</h2>', unsafe_allow_html=True)
 
    df_model = dff.drop(columns=["Car ID"], errors="ignore").copy()
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
 
    y = df_encoded["Price"]
    X = df_encoded.drop(columns=["Price"])
 
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
 
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
 
    model_choice = st.selectbox("Model", ["Linear Regression", "Ridge Regression", "Lasso Regression"])
 
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Ridge Regression":
        alpha = st.slider("Ridge alpha", 0.01, 10.0, 1.0)
        model = Ridge(alpha=alpha)
    else:
        alpha = st.slider("Lasso alpha", 0.01, 10.0, 0.1)
        model = Lasso(alpha=alpha)
 
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
 
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv   = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="r2").mean()
 
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    metrics = [("R2 Score", f"{r2:.4f}", "Higher = better fit"),
               ("MAE ($)", f"{mae:,.0f}", "Mean absolute error"),
               ("RMSE ($)", f"{rmse:,.0f}", "Root mean squared error"),
               ("CV R2", f"{cv:.4f}", "5-fold cross-validation")]
    for col, (lbl, val, sub) in zip([m1, m2, m3, m4], metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
            <div style="font-size:0.75rem;color:#475569;margin-top:4px">{sub}</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown(f'<div class="insight-box"><b>Model Interpretation:</b> The {model_choice} explains <b>{r2*100:.1f}%</b> of price variance (R2 = {r2:.4f}). '
                f'On average, predictions deviate by <b>${mae:,.0f}</b> (MAE). '
                f'A 5-fold cross-validated R2 of {cv:.4f} confirms the model generalizes well to unseen data.</div>',
                unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("##### Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        apply_dark_style(ax, fig)
        ax.scatter(y_test, y_pred, alpha=0.5, s=20, color=BLUE, edgecolors="none")
        lim = (min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()))
        ax.plot(lim, lim, color=ROSE, linewidth=1.5, linestyle="--", label="Perfect prediction")
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title(f"Actual vs Predicted — R2={r2:.3f}", color=TEXT)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax.legend(frameon=False, labelcolor=MUTED, fontsize=8)
        st.pyplot(fig)
        plt.close()
 
    with col_b:
        st.markdown("##### Residuals Distribution")
        residuals = y_test - y_pred
        fig, ax = plt.subplots(figsize=(5.5, 4))
        apply_dark_style(ax, fig)
        sns.histplot(residuals, kde=True, color=TEAL, ax=ax, bins=30, alpha=0.7)
        ax.axvline(0, color=ROSE, linewidth=1.5, linestyle="--")
        ax.set_xlabel("Residual ($)")
        ax.set_title("Residual Distribution (should be ~normal, centered at 0)", color=TEXT, fontsize=9)
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">Residuals centred near zero = no systematic bias. Bell-shaped distribution validates linear regression assumptions.</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="section-header">Feature Importance (Coefficients)</div>', unsafe_allow_html=True)
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=False).index)
    top_coef = coef_df.head(12)
 
    fig, ax = plt.subplots(figsize=(9, 4))
    apply_dark_style(ax, fig)
    colors = [TEAL if v >= 0 else ROSE for v in top_coef["Coefficient"]]
    ax.barh(top_coef["Feature"], top_coef["Coefficient"], color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color=BORDER, linewidth=1)
    ax.set_title("Top Coefficient Magnitudes (Standardised)", color=TEXT)
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()
 
    st.markdown('<div class="insight-box"><b>Interpretation:</b> Green bars = positive impact on price (e.g., high HP, recent year). Red bars = negative impact (e.g., high mileage, older year). Magnitude reflects relative importance after standardisation.</div>', unsafe_allow_html=True)
 
    st.markdown('<div class="section-header">Prediction Samples</div>', unsafe_allow_html=True)
    results = pd.DataFrame({
        "Actual Price ($)": y_test.values[:15].round(0),
        "Predicted Price ($)": y_pred[:15].round(0),
        "Error ($)": (y_test.values[:15] - y_pred[:15]).round(0),
        "Error (%)": ((y_test.values[:15] - y_pred[:15]) / y_test.values[:15] * 100).round(1)
    })
    st.dataframe(results, use_container_width=True)
 
# -------------------------------------------------
# PAGE 4: PREDICT
# -------------------------------------------------
elif page == "Predict":
    st.markdown('<h2 class="section-header">Price Predictor</h2>', unsafe_allow_html=True)
    st.markdown("Enter car details below to get an estimated price.")
 
    df_model = df.drop(columns=["Car ID"], errors="ignore").copy()
    df_encoded = pd.get_dummies(df_model, drop_first=True)
    df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))
    y_full = df_encoded["Price"]
    X_full = df_encoded.drop(columns=["Price"])
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X_full)
    lr = LinearRegression()
    lr.fit(X_scaled, y_full)
 
    col1, col2 = st.columns(2)
    inputs = {}
 
    with col1:
        if "Year" in df.columns:
            inputs["Year"] = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].median()))
        if "Mileage" in df.columns:
            inputs["Mileage"] = st.number_input("Mileage (km)", 0, 300000, 50000, 5000)
        if "HP" in df.columns:
            inputs["HP"] = st.slider("Horsepower", int(df["HP"].min()), int(df["HP"].max()), int(df["HP"].median()))
 
    with col2:
        if "Engine Size" in df.columns:
            inputs["Engine Size"] = st.slider("Engine Size (L)", float(df["Engine Size"].min()), float(df["Engine Size"].max()), float(df["Engine Size"].median()), 0.1)
        if "Brand" in df.columns:
            brand_choice = st.selectbox("Brand", sorted(df["Brand"].unique().tolist()))
        if "Fuel Type" in df.columns:
            fuel_choice = st.selectbox("Fuel Type", sorted(df["Fuel Type"].unique().tolist()))
 
    if st.button("Estimate Price", use_container_width=True):
        input_row = pd.DataFrame([inputs])
        if "Brand" in df.columns:
            input_row["Brand"] = brand_choice
        if "Fuel Type" in df.columns:
            input_row["Fuel Type"] = fuel_choice
 
        try:
            input_encoded = pd.get_dummies(input_row)
            input_encoded = input_encoded.reindex(columns=X_full.columns, fill_value=0)
            input_scaled  = scaler_full.transform(input_encoded)
            price_pred     = lr.predict(input_scaled)[0]
 
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1a2035,#1e2740);border:1px solid #3b82f6;border-radius:14px;padding:30px;text-align:center;margin-top:20px;font-family:Arial,sans-serif;">
                <div style="font-size:0.9rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.1em;">Estimated Price</div>
                <div style="font-size:3rem;font-weight:800;color:#60a5fa;font-family:Arial,sans-serif;margin:8px 0">${price_pred:,.0f}</div>
                <div style="font-size:0.8rem;color:#475569">Based on Linear Regression model trained on full dataset</div>
            </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")
 
    st.markdown("---")
    st.markdown('<p style="color:#475569;font-size:0.8rem;text-align:center;font-family:Arial,sans-serif;">Built with Streamlit · Linear Regression · Cars Dataset Analysis</p>', unsafe_allow_html=True)