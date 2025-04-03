import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from geopy.distance import geodesic
from datetime import datetime
from streamlit_folium import folium_static
import io  # For CSV downloads


# Load datasets
def load_data():
    df_eq = pd.read_csv("earthquake_data_myanmar.csv")
    df_eq["time"] = pd.to_datetime(df_eq["time"], errors="coerce")
    df_eq["time"] = df_eq["time"].dt.tz_localize(None)

    df_pop_mm = pd.read_csv("MyanmarPop_2014_Township.csv")
    df_pop_mm = df_pop_mm.loc[:, ~df_pop_mm.columns.str.startswith('Unnamed')]
    return df_eq, df_pop_mm


df_eq, df_pop_mm = load_data()


# Function to estimate affected population
def estimate_affected_population(eq_lat, eq_lon, eq_mag):
    affected_pop = 0
    for _, row in df_pop_mm.iterrows():
        region_lat = row["lat"]
        region_lon = row["lon"]
        total_pop = row["hh_total"]

        if pd.isna(region_lat) or pd.isna(region_lon) or pd.isna(total_pop):
            continue

        distance_km = geodesic((eq_lat, eq_lon), (region_lat, region_lon)).km
        max_radius = eq_mag * 10  # Dynamic buffer based on magnitude

        if distance_km <= max_radius:
            weight = np.exp(-distance_km / (max_radius / 2))
            magnitude_factor = (eq_mag / 7.0) ** 2
            affected_pop += total_pop * weight * magnitude_factor

    return int(affected_pop)


# Function to plot earthquake data on a map
def plot_earthquake_map(start_date, end_date):
    filtered_eq = df_eq[(df_eq["time"] >= start_date) & (df_eq["time"] <= end_date)].copy()

    if filtered_eq.empty:
        st.warning("⚠️ No earthquakes found in the selected time range.")
        return None, None  # Ensure two values are returned

    # Compute estimated affected population
    filtered_eq["Estimated_Affected_Pop"] = filtered_eq.apply(
        lambda row: estimate_affected_population(row["latitude"], row["longitude"], row["mag"]), axis=1
    )

    # Create a map centered on Myanmar
    myanmar_map = folium.Map(location=[21.0, 96.0], zoom_start=6)

    # Add population heatmap
    heat_data = [[row["lat"], row["lon"], row["hh_total"]] for _, row in df_pop_mm.iterrows()]
    if heat_data:
        HeatMap(heat_data, min_opacity=0.4, radius=20, blur=10, max_zoom=1).add_to(myanmar_map)

    # Add earthquake markers
    for _, row in filtered_eq.iterrows():
        weighted_population = row["Estimated_Affected_Pop"]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=max(5, np.log1p(weighted_population) / 2),
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=min(0.9, weighted_population / 1e6) if weighted_population > 0 else 0.3,
            popup=(
                f"<b>Date:</b> {row['time'].date()}<br>"
                f"<b>Magnitude:</b> {row['mag']}<br>"
                f"<b>Estimated Affected Population:</b> {int(weighted_population):,}"
            ),
        ).add_to(myanmar_map)

    folium_static(myanmar_map)

    return filtered_eq, myanmar_map  # Return both filtered data and the map


# Streamlit UI
st.title("🌍 Earthquake-Affected Population")
st.sidebar.header("📌 Filter Earthquake Data")

# 🔹 Set today's date as default for both start and end date
today_date = datetime.today().date()

start_date = st.sidebar.date_input("📅 Start Date", today_date)
end_date = st.sidebar.date_input("📅 End Date", today_date)

# Validate date selection
if start_date > end_date:
    st.error("❌ Start date cannot be after end date!")
else:
    # Generate map based on selected date range
    filtered_data, _ = plot_earthquake_map(pd.to_datetime(start_date), pd.to_datetime(end_date))

    # CSV Export Button
    if filtered_data is not None and isinstance(filtered_data, pd.DataFrame) and not filtered_data.empty:
        csv = filtered_data.to_csv(index=False).encode("utf-8")
        filename = f"earthquake_data_{start_date}_to_{end_date}.csv"

        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=filename,
            mime="text/csv",
        )
