import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from geopy.distance import geodesic
from datetime import datetime
from streamlit_folium import folium_static
import requests
import time
import io


# 🔹 **Fetch Earthquake Data from USGS API**
def fetch_earthquake_data(start_date, end_date, min_magnitude=4.0):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "csv",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "maxmagnitude": 10.0,
        "orderby": "time",
    }

    st.info(f"Fetching earthquake data from {start_date} to {end_date}...")

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            return df
        else:
            st.error(f"❌ Failed to fetch data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None


# 🔹 **Load Population Data (Local CSV)**
@st.cache_data
def load_population_data():
    df_pop_mm = pd.read_csv("MyanmarPop_2014_Township.csv")
    df_pop_mm = df_pop_mm.loc[:, ~df_pop_mm.columns.str.startswith('Unnamed')]
    return df_pop_mm


df_pop_mm = load_population_data()


# 🔹 **Estimate Affected Population**
def estimate_affected_population(eq_lat, eq_lon, eq_mag):
    affected_pop = 0
    for _, row in df_pop_mm.iterrows():
        region_lat = row["lat"]
        region_lon = row["lon"]
        total_pop = row["hh_total"]

        if pd.isna(region_lat) or pd.isna(region_lon) or pd.isna(total_pop):
            continue

        distance_km = geodesic((eq_lat, eq_lon), (region_lat, region_lon)).km
        max_radius = eq_mag * 10

        if distance_km <= max_radius:
            weight = np.exp(-distance_km / (max_radius / 2))
            magnitude_factor = (eq_mag / 7.0) ** 2
            affected_pop += total_pop * weight * magnitude_factor

    return int(affected_pop)


# 🔹 **Plot Earthquake Map**
def plot_earthquake_map(df_eq):
    if df_eq is None or df_eq.empty:
        st.warning("⚠️ No earthquakes found in the selected time range.")
        return None, None

    df_eq["Estimated_Affected_Pop"] = df_eq.apply(
        lambda row: estimate_affected_population(row["latitude"], row["longitude"], row["mag"]), axis=1
    )

    myanmar_map = folium.Map(location=[21.0, 96.0], zoom_start=6)

    # Add Population Heatmap
    heat_data = [[row["lat"], row["lon"], row["hh_total"]] for _, row in df_pop_mm.iterrows()]
    if heat_data:
        HeatMap(heat_data, min_opacity=0.1, radius=10, blur=5, max_zoom=1).add_to(myanmar_map)


    # Add Earthquake Markers
    for _, row in df_eq.iterrows():
        weighted_population = row["Estimated_Affected_Pop"]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(5, np.log1p(weighted_population) / 2),
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=min(0.9, weighted_population / 1e6) if weighted_population > 0 else 0.3,
            popup=(
                f"<b>Date:</b> {row['time']}<br>"
                f"<b>Magnitude:</b> {row['mag']}<br>"
                f"<b>Estimated Affected Population:</b> {int(weighted_population):,}"
            ),
        ).add_to(myanmar_map)

    folium_static(myanmar_map)

    return df_eq, myanmar_map  # Return both filtered data and the map


# 🔹 **Streamlit UI**
st.title("🌍 Earthquake-Affected Population(MM)")
st.sidebar.header("📌 Filter Earthquake Data")

# **Set default date range to today**
today_date = datetime.today().date()
start_date = st.sidebar.date_input("📅 Start Date", today_date)
end_date = st.sidebar.date_input("📅 End Date", today_date)

# Validate date selection
if start_date > end_date:
    st.error("❌ Start date cannot be after end date!")
else:
    df_eq = fetch_earthquake_data(start_date, end_date)

    if df_eq is not None and not df_eq.empty:
        # Convert 'time' column to datetime
        df_eq["time"] = pd.to_datetime(df_eq["time"], errors="coerce")
        df_eq["latitude"] = pd.to_numeric(df_eq["latitude"], errors="coerce")
        df_eq["longitude"] = pd.to_numeric(df_eq["longitude"], errors="coerce")
        df_eq["mag"] = pd.to_numeric(df_eq["mag"], errors="coerce")

        # Generate map based on fetched data
        filtered_data, _ = plot_earthquake_map(df_eq)

        # CSV Export Button
        if filtered_data is not None and not filtered_data.empty:
            csv = filtered_data.to_csv(index=False).encode("utf-8")
            filename = f"earthquake_data_{start_date}_to_{end_date}.csv"

            st.download_button(
                label="📥 Download Earthquake Data as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
            )
# Display earthquake table only if filtered_data exists and is not empty
if 'filtered_data' in locals() and filtered_data is not None and not filtered_data.empty:
    st.subheader("📊 Earthquake Data Table")
    st.dataframe(filtered_data[['time', 'latitude', 'longitude', 'mag', 'Estimated_Affected_Pop', 'place', 'updated']])
else:
    st.warning("⚠️ No earthquake data available for the selected date range.")
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 📌 **Load Data** (df_eq is fetched from the USGS API in your existing code)
st.title("📊 Earthquake Data Visualization")

if "df_eq" in locals() and df_eq is not None and not df_eq.empty:
    # Convert 'time' column to datetime if not already
    df_eq["time"] = pd.to_datetime(df_eq["time"], errors="coerce")

    # Sidebar: Select Visualization
    st.sidebar.header("Visualization Options")
    plot_type = st.sidebar.radio(
        "Choose a plot:",
        ["Earthquake Occurrences Over Time", "Magnitude Distribution", "Geographic Distribution"]
    )

    sns.set_style("whitegrid")

    # Plot 1: Earthquake occurrences over time
    if plot_type == "Earthquake Occurrences Over Time":
        st.subheader("📅 Earthquake Occurrences Over Time")
        fig, ax = plt.subplots(figsize=(12, 5))
        df_eq["time"].hist(bins=100, color="orange", alpha=0.7, ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Earthquakes")
        ax.set_title("Earthquake Occurrences Over Time")
        st.pyplot(fig)

    # Plot 2: Magnitude distribution
    elif plot_type == "Magnitude Distribution":
        st.subheader("📏 Distribution of Earthquake Magnitudes")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_eq["mag"], bins=30, kde=True, color="red", alpha=0.6, ax=ax)
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Earthquake Magnitudes")
        st.pyplot(fig)

    # Plot 3: Geographic distribution of earthquakes
    elif plot_type == "Geographic Distribution":
        st.subheader("🌍 Geographic Distribution of Earthquakes")
        fig, ax = plt.subplots(figsize=(10, 5))
        scatter = ax.scatter(df_eq["longitude"], df_eq["latitude"], c=df_eq["mag"], cmap="coolwarm", alpha=0.6)
        plt.colorbar(scatter, label="Magnitude")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Geographic Distribution of Earthquakes")
        st.pyplot(fig)
else:
    st.warning("⚠️ No earthquake data available. Please adjust the date range or try again.")

########################

import streamlit as st
import pandas as pd

# Function to prepare data for forecasting
def prepare_forecasting_data(df):
    st.subheader("🛠️ Data Preparation for Forecasting")

    # Select necessary columns
    df_clean = df[["time", "latitude", "longitude", "depth", "mag"]].copy()

    # Handle missing values
    df_clean = df_clean.dropna()

    # Feature Engineering: Extract time-based features
    df_clean["year"] = df_clean["time"].dt.year
    df_clean["month"] = df_clean["time"].dt.month
    df_clean["day"] = df_clean["time"].dt.day
    df_clean["hour"] = df_clean["time"].dt.hour

    # Feature Engineering: Create lag features
    df_clean["mag_lag_1"] = df_clean["mag"].shift(1)
    df_clean["mag_lag_2"] = df_clean["mag"].shift(2)

    # Drop NaN values resulting from lag creation
    df_clean = df_clean.dropna()

    # Display processed dataset
    st.write("📌 Processed Dataset Preview:")
    st.dataframe(df_clean.head())

    return df_clean

@st.cache_data
# 🔹 **Fetch Earthquake Data from USGS API**
def fetch_earthquake_data_forecast(start_date, end_date, min_magnitude=2.0):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "csv",
        "starttime": start_date.strftime("%Y-%m-%d"),
        "endtime": end_date.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "maxmagnitude": 10.0,
        "orderby": "time",
    }

    st.info(f"Fetching earthquake data from {start_date} to {end_date}...")

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            return df
        else:
            st.error(f"❌ Failed to fetch data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
        return None

def load_data():
    # Fetch data from the USGS API (assuming it's already implemented)
    df = fetch_earthquake_data_forecast(pd.to_datetime("2024-01-01"), pd.to_datetime("2025-12-31"))
    if df is not None:
        df["time"] = pd.to_datetime(df["time"])
    return df

# Streamlit App
st.title("📊 Earthquake Forecasting Data Preparation - Random Forest & LSTM")
df = load_data()
if df is not None:
    df_clean = prepare_forecasting_data(df)
else:
    st.error("Failed to load earthquake data.")


##########################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data (Assuming it's already fetched from USGS API)
@st.cache_data
def load_data():
    df = fetch_earthquake_data(pd.to_datetime("2025-01-01"), pd.to_datetime("2025-12-31"))
    if df is not None:
        df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()
if df is None:
    st.error("Failed to load earthquake data.")
    st.stop()

df["time"] = pd.to_datetime(df["time"], errors="coerce")
df_myanmar = df[(df["latitude"] >= 9.5) & (df["latitude"] <= 28.5) &
                (df["longitude"] >= 92.2) & (df["longitude"] <= 101.2)]

# Feature engineering
df_myanmar["year"] = df_myanmar["time"].dt.year
df_myanmar["month"] = df_myanmar["time"].dt.month
df_myanmar["day"] = df_myanmar["time"].dt.day
df_myanmar["hour"] = df_myanmar["time"].dt.hour

# Train-test split
features = ["year", "month", "day", "hour", "depth"]
target = ["latitude", "longitude", "mag"]
X_train, X_test, y_train, y_test = train_test_split(
    df_myanmar[features], df_myanmar[target], test_size=0.2, random_state=42)

# Train Random Forest models
rf_lat = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_long = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_mag = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
rf_lat.fit(X_train, y_train["latitude"])
rf_long.fit(X_train, y_train["longitude"])
rf_mag.fit(X_train, y_train["mag"])

# Predictions
rf_pred_lat = rf_lat.predict(X_test)
rf_pred_long = rf_long.predict(X_test)
rf_pred_mag = rf_mag.predict(X_test)

# Scatter plot of predictions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x=y_test["latitude"], y=rf_pred_lat, ax=axes[0])
axes[0].set_xlabel("Actual Latitude")
axes[0].set_ylabel("Predicted Latitude")
sns.scatterplot(x=y_test["longitude"], y=rf_pred_long, ax=axes[1])
axes[1].set_xlabel("Actual Longitude")
sns.scatterplot(x=y_test["mag"], y=rf_pred_mag, ax=axes[2])
axes[2].set_xlabel("Actual Magnitude")

st.pyplot(fig)


#########################
#actual Vs. predicted points map
rf_predictions = pd.DataFrame({
    "Actual Latitude": y_test["latitude"].values,
    "Predicted Latitude": rf_pred_lat,
    "Actual Longitude": y_test["longitude"].values,
    "Predicted Longitude": rf_pred_long,
    "Actual Magnitude": y_test["mag"].values,
    "Predicted Magnitude": rf_pred_mag
})

import streamlit as st
#st.set_page_config(layout="wide")  # Should always be at the top

import pandas as pd
import folium
from geopy.distance import geodesic
from streamlit_folium import folium_static

# 🔹 App Title
st.title("📍 Earthquake Prediction vs Actual Locations - Random Forest")

# 🔹 Calculate geodesic error
rf_predictions["Error (km)"] = rf_predictions.apply(lambda row: geodesic(
    (row["Actual Latitude"], row["Actual Longitude"]),
    (row["Predicted Latitude"], row["Predicted Longitude"])
).km, axis=1)

# 🔹 Show data table
st.subheader("📊 Prediction Results with Geodesic Error - Random Forest")
st.dataframe(rf_predictions)

# 🔹 Initialize map
map_pred = folium.Map(location=[21.0, 96.0], zoom_start=6, tiles="CartoDB positron")

# 🔹 Add markers and lines
for idx, row in rf_predictions.iterrows():
    actual = (row["Actual Latitude"], row["Actual Longitude"])
    predicted = (row["Predicted Latitude"], row["Predicted Longitude"])
    error_km = row["Error (km)"]

    folium.Marker(actual, tooltip=f"Actual #{idx+1}", icon=folium.Icon(color="green")).add_to(map_pred)
    folium.Marker(predicted, tooltip=f"Predicted #{idx+1} (Error: {error_km:.2f} km)", icon=folium.Icon(color="red")).add_to(map_pred)
    folium.PolyLine([actual, predicted], color="orange", weight=2, tooltip=f"{error_km:.2f} km").add_to(map_pred)

# 🔹 Display map
st.subheader("🗺️ Prediction vs Actual Location Map")
folium_static(map_pred)

#########################

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Title of the Streamlit app
st.title("Earthquake Forecasting in Myanmar using LSTM")

def load_data():
    # Assuming you have a function fetch_earthquake_data() that retrieves the dataset
    df = fetch_earthquake_data(pd.to_datetime("2024-01-01"), pd.to_datetime("2025-12-31"))
    if df is not None:
        df["time"] = pd.to_datetime(df["time"])
    return df

df = load_data()

if df is None:
    st.error("Failed to load earthquake data.")
    st.stop()

# Convert time column to datetime and create a timestamp column
df["time"] = pd.to_datetime(df["time"], errors="coerce")

# Filter for Myanmar's region
df_myanmar_lstm = df[(df["latitude"] >= 9.5) & (df["latitude"] <= 28.5) &
                     (df["longitude"] >= 92.2) & (df["longitude"] <= 101.2)]

# Ensure we have data
if df_myanmar_lstm.empty:
    st.error("No earthquake data available for Myanmar.")
    st.stop()

# Convert time to Unix timestamp
df_myanmar_lstm["timestamp"] = df_myanmar_lstm["time"].view(np.int64) // 10**9

# Define features and target
features = ["timestamp", "depth", "mag"]
target = ["latitude", "longitude", "mag"]

# Normalize features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(df_myanmar_lstm[features])
y_scaled = scaler_y.fit_transform(df_myanmar_lstm[target])

# Create sequences for LSTM
seq_length = 5
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i: i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_length)

# Split into train and test sets
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, X_train.shape[2])),
    LSTM(50, return_sequences=False),
    Dense(32, activation="relu"),
    Dense(3)  # Output: latitude, longitude, magnitude
])

# Compile model
model.compile(optimizer="adam", loss="mse")

# Train model with a button click
if st.button("Train LSTM Model"):
    with st.spinner("Training model... This may take a while."):
        model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_test, y_test), verbose=1)
    st.success("Model training completed!")

    # Make predictions
    y_pred = model.predict(X_test)

    # Inverse transform predictions
    y_pred_actual = scaler_y.inverse_transform(y_pred)
    y_test_actual = scaler_y.inverse_transform(y_test)

    # Convert results to DataFrame
    predictions_df = pd.DataFrame({
        "Actual Latitude": y_test_actual[:, 0], "Predicted Latitude": y_pred_actual[:, 0],
        "Actual Longitude": y_test_actual[:, 1], "Predicted Longitude": y_pred_actual[:, 1],
        "Actual Magnitude": y_test_actual[:, 2], "Predicted Magnitude": y_pred_actual[:, 2],
    })

    st.write("### Predictions vs Actual Data")
    st.write(predictions_df.head())

    # Plot Actual vs Predicted Values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(predictions_df["Actual Latitude"], label="Actual Latitude")
    axes[0].plot(predictions_df["Predicted Latitude"], label="Predicted Latitude")
    axes[0].set_title("Actual vs. Predicted Latitude")
    axes[0].legend()

    axes[1].plot(predictions_df["Actual Longitude"], label="Actual Longitude")
    axes[1].plot(predictions_df["Predicted Longitude"], label="Predicted Longitude")
    axes[1].set_title("Actual vs. Predicted Longitude")
    axes[1].legend()

    axes[2].plot(predictions_df["Actual Magnitude"], label="Actual Magnitude")
    axes[2].plot(predictions_df["Predicted Magnitude"], label="Predicted Magnitude")
    axes[2].set_title("Actual vs. Predicted Magnitude")
    axes[2].legend()

    st.pyplot(fig)
