import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
from pathlib import Path

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Uber Demand Prediction Dashboard",
    layout="wide",
)

ROOT = Path(__file__).parent

# ------------------------------------------------------------
# NYC MAP BOUNDS
# ------------------------------------------------------------
MIN_LAT, MAX_LAT = 40.60, 40.85
MIN_LON, MAX_LON = -74.05, -73.70

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df_test = pd.read_csv(ROOT/"data/test.csv", parse_dates=["tpep_pickup_datetime"])
df_train = pd.read_csv(ROOT/"data/train.csv", parse_dates=["tpep_pickup_datetime"])

scaler = joblib.load(ROOT/"models/scaler.joblib")
kmeans = joblib.load(ROOT/"models/mb_kmeans.joblib")
MODEL_DIR = ROOT / "models"

# ------------------------------------------------------------
# CLUSTER CENTERS → REAL COORDINATES
# ------------------------------------------------------------
cluster_centers_real = scaler.inverse_transform(kmeans.cluster_centers_)
region_map_df = pd.DataFrame(cluster_centers_real, columns=["pickup_longitude", "pickup_latitude"])
region_map_df["region"] = region_map_df.index

# ------------------------------------------------------------
# PROPHET TREND FUNCTION
# ------------------------------------------------------------
def compute_prophet_trend(region_id):
    """Recomputes Prophet trend exactly like training step."""
    train_region = df_train[df_train["region"] == region_id].copy()
    test_region  = df_test[df_test["region"] == region_id].copy()

    prophet_model = joblib.load(MODEL_DIR / f"prophet_region_{region_id}.joblib")

    full_data = pd.concat([train_region, test_region])
    full_data["hour"] = full_data["tpep_pickup_datetime"].dt.floor("H")

    unique_hours = full_data["hour"].drop_duplicates().reset_index(drop=True)
    future = pd.DataFrame({"ds": unique_hours})
    forecast = prophet_model.predict(future)[["ds", "yhat"]]

    full_data = full_data.merge(forecast, left_on="hour", right_on="ds", how="left")
    full_data.rename(columns={"yhat": "prophet_trend"}, inplace=True)
    return full_data

# ------------------------------------------------------------
# XGBOOST REGION PREDICTION
# ------------------------------------------------------------
def predict_row(row, region_id):
    """Runs region-specific XGBoost model for one row."""
    model = joblib.load(MODEL_DIR / f"xgb_region_{region_id}.joblib")

    feat_cols = [
        "lag_1", "lag_2", "lag_3", "lag_4",
        "avg_pickups", "day_of_week",
        "rolling_mean_3", "rolling_mean_6",
        "rolling_std_3", "rolling_std_6",
        "prophet_trend"
    ]

    X = pd.DataFrame([row[feat_cols]])
    return int(round(model.predict(X)[0]))  # integer result

# ------------------------------------------------------------
# INTRO
# ------------------------------------------------------------
st.title("🚕 Uber Demand Prediction Dashboard")

st.markdown("""
Welcome to the **NYC Uber Demand Forecaster**, powered by:

- 🧠 **MiniBatch KMeans** → divides NYC into demand regions  
- 🔮 **Prophet** → learns hourly trends  
- ⚡ **XGBoost** → forecasts 15-minute demand  

You will:
1. Pick any date & time  
2. A random NYC coordinate will be generated  
3. We detect its region  
4. We predict demand for **all 30 regions**  
5. Show results beautifully + on NYC heatmap  
---
""")

# ------------------------------------------------------------
# STEP 1 — DATE & TIME PICKING
# ------------------------------------------------------------
st.header("📅 Step 1 — Select When You Want to Forecast")

min_date = df_test["tpep_pickup_datetime"].dt.date.min()
max_date = df_test["tpep_pickup_datetime"].dt.date.max()

date_selected = st.date_input(
    "Select a Date:",
    value=min_date,
    min_value=min_date,
    max_value=max_date,
)

available_times = sorted(
    df_test[df_test["tpep_pickup_datetime"].dt.date == date_selected]
           ["tpep_pickup_datetime"].dt.time.unique()
)

time_selected = st.selectbox("Select 15-Minute Time Slot:", available_times)

selected_dt = pd.Timestamp.combine(date_selected, time_selected)
next_dt = selected_dt + pd.Timedelta(minutes=15)

st.info(
    f"⏳ You selected **{selected_dt}**.\n"
    f"📌 Prediction will be for **the next window: {next_dt}**."
)

# ------------------------------------------------------------
# STEP 2 — RUN PREDICTION
# ------------------------------------------------------------
if st.button("🔮 Predict Demand for Next 15 Minutes"):

    # PROGRESS
    progress_text = st.empty()
    progress_bar = st.progress(0)

    # ------------------------------------------------------------
    # RANDOM COORDINATE
    # ------------------------------------------------------------
    progress_text.write("🎲 Generating a random coordinate inside NYC...")
    rand_lat = np.random.uniform(MIN_LAT, MAX_LAT)
    rand_lon = np.random.uniform(MIN_LON, MAX_LON)
    progress_bar.progress(15)

    # ------------------------------------------------------------
    # DETECT REGION
    # ------------------------------------------------------------
    progress_text.write("📍 Detecting nearest region cluster...")
    scaled = scaler.transform([[rand_lon, rand_lat]])
    detected_region = int(kmeans.predict(scaled)[0])
    progress_bar.progress(30)

    st.success(
        f"""
        ### 🎯 Region Identified: **Region {detected_region}**

        Your generated coordinate:
        - 📍 **Latitude:** `{rand_lat:.5f}`
        - 📍 **Longitude:** `{rand_lon:.5f}`  

        belongs to **Region {detected_region}**, based on distance to KMeans cluster centers.
        """
    )

    # ------------------------------------------------------------
    # PREDICT FOR ALL REGIONS
    # ------------------------------------------------------------
    progress_text.write("🔮 Predicting demand for all NYC regions...")

    predictions = []
    regions = sorted(df_test["region"].unique())
    total = len(regions)

    for i, rid in enumerate(regions):
        pct = 30 + int((i+1)/total * 65)
        progress_bar.progress(pct)
        progress_text.write(f"📡 Computing prediction for Region {rid}...")

        merged_df = compute_prophet_trend(rid)
        row = merged_df[merged_df["tpep_pickup_datetime"] == next_dt].iloc[0]
        pred = predict_row(row, rid)
        predictions.append([rid, pred])

    progress_bar.progress(100)
    progress_text.write("✅ Prediction complete!")

    pred_df = pd.DataFrame(predictions, columns=["region", "predicted_demand"])
    pred_df_sorted = pred_df.sort_values("predicted_demand", ascending=False)

    # ------------------------------------------------------------
    # TOP 5 REGIONS — BEAUTIFUL CARDS
    st.header("🏆 Top 5 Highest Demand Regions")
    st.caption("These regions are expected to be busiest in the next 15-minute interval.")

    top5 = pred_df_sorted.head(5)

    st.markdown(
        """
        <style>
        .card-scroll-container {
            display: flex;
            flex-direction: row;
            gap: 16px;
            margin-top: 20px;
            overflow-x: auto;
            padding-bottom: 10px;
            white-space: nowrap;
        }
        .demand-card {
            flex: 0 0 auto;
            width: 180px;
            background: linear-gradient(135deg, #ff9a00, #ff3d00);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }
        .demand-card h3 { margin: 0; }
        </style>
        """,
        unsafe_allow_html=True
    )

    cards = ["<div class='card-scroll-container'>"]

    for _, row in top5.iterrows():
        cards.append(
            f"<div class='demand-card'>"
            f"<h3>Region {int(row['region'])}</h3>"
            f"<p style='font-size:28px; margin:4px 0;'>{int(row['predicted_demand'])}</p>"
            f"<small>Predicted Demand</small>"
            f"</div>"
        )

    cards.append("</div>")

    st.markdown("".join(cards), unsafe_allow_html=True)
        # ------------------------------------------------------------
    # 📊 BAR GRAPH — DEMAND ACROSS ALL 30 REGIONS
    # ------------------------------------------------------------
    st.header("📊 Demand Comparison Across All 30 Regions")
    st.caption("Bar chart showing predicted demand distribution across every NYC region.")

    bar_df = pred_df_sorted.copy()
    bar_df["Region"] = bar_df["region"].astype(str)

    st.bar_chart(
        bar_df.set_index("Region")["predicted_demand"],
        height=380
    )



    # ------------------------------------------------------------
    # TOP 10 CLOSEST REGIONS TO USER COORDINATE
    # ------------------------------------------------------------
    st.header("📍 Top 10 Closest Regions to Your Location")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (np.sin(dlat/2)**2
             + np.cos(np.radians(lat1))
             * np.cos(np.radians(lat2))
             * np.sin(dlon/2)**2)
        return 2 * R * np.arcsin(np.sqrt(a))

    region_map_df["distance_km"] = region_map_df.apply(
        lambda r: haversine(rand_lat, rand_lon, r["pickup_latitude"], r["pickup_longitude"]),
        axis=1
    )

    closest_df = (
        region_map_df.merge(pred_df, on="region")
                     .sort_values("distance_km")
                     .head(10)
                     .reset_index(drop=True)
    )

    colset = st.columns(5)
    for i, row in closest_df.iterrows():
        col = colset[i % 5]
        with col:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #009FFD, #2A2A72);
                            padding: 16px; border-radius: 14px; margin-bottom: 12px;
                            text-align: center; color: white;
                            box-shadow: 0 4px 10px rgba(0,0,0,0.25);">
                    <h3 style="margin:0;">Region {int(row['region'])}</h3>
                    <p style="font-size:26px;">Demand: {int(row['predicted_demand'])}</p>
                    <small>{row['distance_km']:.2f} km away</small>
                </div>
                """,
                unsafe_allow_html=True
            )
    # ------------------------------------------------------------
    # CLEAN DEMAND MAP — No dropdown, no labels, 3-level color coding
    # ------------------------------------------------------------

    st.header("🗺️ NYC Demand Heatmap (High / Medium / Low)")

    # Merge region coordinates + predictions
    map_df = region_map_df.merge(pred_df, on="region")

    # 1️⃣ Split all regions into 3 equal demand tiers
    map_df = map_df.sort_values("predicted_demand", ascending=True).reset_index(drop=True)

    n = len(map_df)
    low_cut = n // 3
    medium_cut = 2 * (n // 3)

    map_df.loc[:low_cut, "demand_class"] = "Low"
    map_df.loc[low_cut:medium_cut, "demand_class"] = "Medium"
    map_df.loc[medium_cut:, "demand_class"] = "High"

    # 2️⃣ Assign fixed colors for the 3 tiers
    COLOR_MAP = {
        "Low":    [0, 180, 255, 220],   # Light Blue
        "Medium": [255, 165, 0, 220],   # Orange
        "High":   [255, 0, 0, 220],     # Red
    }

    map_df["color"] = map_df["demand_class"].apply(lambda x: COLOR_MAP[x])

    # 3️⃣ PyDeck Map — Clean layer
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[pickup_longitude, pickup_latitude]",
        get_fill_color="color",
        get_radius=650,
        pickable=True,
    )

    tooltip = {
        "html": "<b>Region:</b> {region}<br/>"
                "<b>Demand Tier:</b> {demand_class}<br/>"
                "<b>Predicted Demand:</b> {predicted_demand}",
        "style": {"backgroundColor": "black", "color": "white"},
    }

    view_state = pdk.ViewState(
        latitude=(MIN_LAT + MAX_LAT) / 2,
        longitude=(MIN_LON + MAX_LON) / 2,
        zoom=11,
    )

    deck = pdk.Deck(
        layers=[scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None,  # black background
    )

    st.pydeck_chart(deck)

    # 4️⃣ SIMPLE LEGEND AS INFO BOX
    st.info(
        "🎨 **Color Coding**\n"
        "- 🔴 **High Demand** (bright red points)\n"
        "- 🟠 **Medium Demand** (orange points)\n"
        "- 🔵 **Low Demand** (light blue points)"
    )


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def plot_prophet_trend(region_id):
    # Load Prophet model
    prophet_model = joblib.load(MODEL_DIR / f"prophet_region_{region_id}.joblib")

    # Prepare data
    region_data = df_train[df_train["region"] == region_id].copy()
    region_data = region_data.sort_values("tpep_pickup_datetime")

    df_input = pd.DataFrame({
        "ds": region_data["tpep_pickup_datetime"].dt.floor("H"),
        "y": region_data["total_pickups"]
    }).drop_duplicates(subset="ds")

    # Prophet forecast
    forecast = prophet_model.predict(df_input[["ds"]])

    # Beautiful trend plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Actual demand (noisy)
    ax.plot(
        df_input["ds"], df_input["y"],
        color="lightgray", alpha=0.4, linewidth=1.3, label="Actual Demand"
    )

    # Prophet trend
    ax.plot(
        forecast["ds"], forecast["trend"],
        color="#FF3B3B", linewidth=3.5, label="Prophet Trend"
    )

    ax.set_title("Long-Term Trend Extraction", fontsize=20, weight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Demand", fontsize=12)
    ax.legend()

    # Format x-axis
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()

    st.pyplot(fig)
    st.caption("""
    ⭐ Prophet removes high-frequency noise and extracts the true long-term direction of demand.
    ⭐ XGBoost uses this trend to avoid reacting to random fluctuations.
    ⭐ This is one of the main reasons the Hybrid model outperforms pure ML.
    """)
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hourly_heatmap(region_id):
    region_data = df_train[df_train["region"] == region_id].copy()

    region_data["hour"] = region_data["tpep_pickup_datetime"].dt.hour
    region_data["day"] = region_data["tpep_pickup_datetime"].dt.dayofweek  # 0 = Monday

    pivot = region_data.pivot_table(
        index="hour",
        columns="day",
        values="total_pickups",
        aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        cmap="viridis",
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Avg Pickups"}
    )

    ax.set_title("Hourly Demand Seasonality Heatmap", fontsize=18, weight="bold")
    ax.set_xlabel("Day of Week (0=Mon, 6=Sun)")
    ax.set_ylabel("Hour of Day")
    st.pyplot(fig)

    st.caption("""
    🔍 This heatmap reveals strong **daily & weekly seasonality** in NYC demand.
    • Clear morning & evening peaks  
    • Weekend patterns differ from weekdays  
    • Prophet captures these seasonal cycles automatically  
    """)

region_select = st.selectbox("Select region for trend analysis:", sorted(df_train["region"].unique()), key="trend_viz")
plot_prophet_trend(region_select)
region_select2 = st.selectbox("Select region for seasonality heatmap:", sorted(df_train["region"].unique()), key="heat_viz")
plot_hourly_heatmap(region_select2)
