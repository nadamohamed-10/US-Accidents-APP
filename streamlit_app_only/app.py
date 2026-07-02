import requests
import streamlit as st
from datetime import datetime, date, time, timedelta

st.set_page_config(page_title="US Accidents Predictor", layout="centered")

API_URL = st.sidebar.text_input(
    "API URL", value="https://us-accidents-app-3.onrender.com"
)

st.title(" US Accidents — Severity & Duration Predictor")
st.caption("Enter accident conditions and get a model prediction for severity and expected traffic blockage time.")

with st.sidebar:
    st.subheader("API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        if health.get("status") == "ok":
            st.success("Connected — models ready")
        else:
            st.warning(f"Connected, but: {health.get('status')}")
    except Exception:
        st.error("Can't reach the API. Is it running?")

STATE_CENTROIDS = {
    "CA": (36.78, -119.42), "TX": (31.97, -99.90), "FL": (27.66, -81.52),
    "NY": (42.17, -74.95), "PA": (41.20, -77.19), "IL": (40.35, -88.99),
    "OH": (40.42, -82.91), "GA": (32.16, -82.90), "NC": (35.76, -79.02),
    "MI": (44.31, -85.60), "AZ": (34.05, -111.09), "WA": (47.75, -120.74),
    "CO": (39.06, -105.31), "VA": (37.77, -78.17), "TN": (35.75, -86.69),
    "SC": (33.84, -81.16), "OR": (43.80, -120.55), "MN": (46.73, -94.69),
}
DEFAULT_CENTROID = (39.83, -98.58)  # geographic center of contiguous US

WEATHER_DEFAULTS = dict(
    Temperature_F=60.0, Wind_Chill_F=55.0, Humidity=65.0, Pressure_in=29.9,
    Visibility_mi=9.0, Wind_Speed_mph=7.0, Precipitation_in=0.0,
    Weather_Condition="Clear", Wind_Direction="CALM",
)
DEFAULT_POI = {k: False for k in [
    "Amenity", "Bump", "Crossing", "Give_Way", "No_Exit",
    "Railway", "Roundabout", "Station", "Traffic_Calming", "Stop",
]}


def build_payload(*, acc_date, start_time_input, duration_min, state,
                   distance, junction, traffic_signal, lighting,
                   start_lat=None, start_lng=None, weather_overrides=None):
    start_dt = datetime.combine(acc_date, start_time_input)
    end_dt = start_dt + timedelta(minutes=duration_min)

    if start_lat is None or start_lng is None:
        start_lat, start_lng = STATE_CENTROIDS.get(state, DEFAULT_CENTROID)

    if lighting == "Day":
        sunrise_sunset, civil_twilight = "Day", "Day"
    elif lighting == "Night":
        sunrise_sunset, civil_twilight = "Night", "Night"
    else:  
        sunrise_sunset, civil_twilight = "Night", "Day"

    weather = {**WEATHER_DEFAULTS, **(weather_overrides or {})}

    payload = {
        "Start_Time": start_dt.isoformat(),
        "End_Time": end_dt.isoformat(),
        "Start_Lat": start_lat,
        "Start_Lng": start_lng,
        "Distance(mi)": distance,
        "Temperature(F)": weather["Temperature_F"],
        "Wind_Chill(F)": weather["Wind_Chill_F"],
        "Humidity(%)": weather["Humidity"],
        "Pressure(in)": weather["Pressure_in"],
        "Visibility(mi)": weather["Visibility_mi"],
        "Wind_Speed(mph)": weather["Wind_Speed_mph"],
        "Precipitation(in)": weather["Precipitation_in"],
        "Weather_Condition": weather["Weather_Condition"],
        "Wind_Direction": weather["Wind_Direction"],
        "State": state,
        "Sunrise_Sunset": sunrise_sunset,
        "Civil_Twilight": civil_twilight,
        "Junction": junction,
        "Traffic_Signal": traffic_signal,
        **DEFAULT_POI,
    }
    return payload


def show_results(payload):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Severity")
        try:
            r = requests.post(f"{API_URL}/predict/severity", json=payload, timeout=10)
            r.raise_for_status()
            result = r.json()
            label, prob = result["predicted_class"], result["high_severity_probability"]
            (st.error if label == "High Severity" else st.success)(f"**{label}**")
            st.metric("High-Severity Probability", f"{prob * 100:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))
        except Exception as e:
            st.error(f"Request failed: {e}")

    with col_b:
        st.markdown("### Expected Duration")
        try:
            dur_payload = {k: v for k, v in payload.items() if k != "End_Time"}
            r = requests.post(f"{API_URL}/predict/duration", json=dur_payload, timeout=10)
            r.raise_for_status()
            mins = r.json()["predicted_duration_minutes"]
            st.metric("Predicted traffic blockage", f"{mins:.0f} min")
            if mins > 120:
                st.warning("Long expected blockage — consider rerouting traffic.")
        except Exception as e:
            st.error(f"Request failed: {e}")

with tab_advanced:
    with st.form("advanced_form"):
        st.subheader(" Location & Time")
        col1, col2 = st.columns(2)
        with col1:
            acc_date = st.date_input("Date", value=date(2026, 1, 15), key="a_date")
            start_lat = st.number_input("Latitude", value=34.05, format="%.4f", key="a_lat")
            state = st.text_input("State (2-letter code)", value="CA", max_chars=2, key="a_state").upper()
        with col2:
            start_time_input = st.time_input("Start time", value=time(8, 30), key="a_time")
            start_lng = st.number_input("Longitude", value=-118.25, format="%.4f", key="a_lng")

        duration_min = st.slider(
            "Estimated duration (minutes) — only used for severity prediction",
            min_value=1, max_value=600, value=60, key="a_dur"
        )

        st.subheader(" Weather")
        col3, col4 = st.columns(2)
        with col3:
            temperature = st.number_input("Temperature (°F)", value=45.0, key="a_temp")
            humidity = st.number_input("Humidity (%)", value=80.0, min_value=0.0, max_value=100.0, key="a_hum")
            pressure = st.number_input("Pressure (in)", value=29.8, key="a_pres")
            visibility = st.number_input("Visibility (mi)", value=6.0, key="a_vis")
        with col4:
            wind_chill = st.number_input("Wind Chill (°F)", value=40.0, key="a_wc")
            wind_speed = st.number_input("Wind Speed (mph)", value=12.0, key="a_ws")
            precipitation = st.number_input("Precipitation (in)", value=0.0, key="a_precip")
            distance = st.number_input("Distance impacted (mi)", value=1.2, key="a_dist")

        weather_condition = st.selectbox(
            "Weather Condition",
            ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm", "Fair", "Overcast"], key="a_wcond"
        )
        wind_direction = st.selectbox(
            "Wind Direction", ["CALM", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "VAR"], key="a_wdir"
        )
        sunrise_sunset = st.radio("Sunrise/Sunset", ["Day", "Night"], horizontal=True, key="a_ss")
        civil_twilight = st.radio("Civil Twilight", ["Day", "Night"], horizontal=True, key="a_ct")

        st.subheader(" Road Features Present")
        poi_cols = st.columns(4)
        poi_labels = ["Junction", "Traffic_Signal", "Crossing", "Stop",
                      "Amenity", "Bump", "Give_Way", "No_Exit",
                      "Railway", "Roundabout", "Station", "Traffic_Calming"]
        poi_values = {}
        for i, label in enumerate(poi_labels):
            with poi_cols[i % 4]:
                poi_values[label] = st.checkbox(label.replace("_", " "), key=f"a_{label}")

        adv_submit = st.form_submit_button(" Predict", use_container_width=True)

    if adv_submit:
        start_dt = datetime.combine(acc_date, start_time_input)
        end_dt = start_dt + timedelta(minutes=duration_min)
        payload = {
            "Start_Time": start_dt.isoformat(),
            "End_Time": end_dt.isoformat(),
            "Start_Lat": start_lat,
            "Start_Lng": start_lng,
            "Distance(mi)": distance,
            "Temperature(F)": temperature,
            "Wind_Chill(F)": wind_chill,
            "Humidity(%)": humidity,
            "Pressure(in)": pressure,
            "Visibility(mi)": visibility,
            "Wind_Speed(mph)": wind_speed,
            "Precipitation(in)": precipitation,
            "Weather_Condition": weather_condition,
            "Wind_Direction": wind_direction,
            "State": state,
            "Sunrise_Sunset": sunrise_sunset,
            "Civil_Twilight": civil_twilight,
            **poi_values,
        }
        show_results(payload)

st.divider()
