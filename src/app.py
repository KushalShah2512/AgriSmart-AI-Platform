import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import requests
from datetime import datetime, timedelta
from streamlit_geolocation import streamlit_geolocation
from deep_translator import GoogleTranslator

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AgriSmart Pro Platform", page_icon="🌾", layout="wide")

# --- 🌐 LIVE API TRANSLATION SETUP ---
api_lang_codes = {
    'English': 'en',
    'हिन्दी (Hindi)': 'hi',
    'मराठी (Marathi)': 'mr',
    'ગુજરાતી (Gujarati)': 'gu'
}

@st.cache_data(show_spinner=False)
def _t(text, target_lang):
    if target_lang == 'English' or not text: return text
    try:
        lang_code = api_lang_codes.get(target_lang, 'en')
        return GoogleTranslator(source='auto', target=lang_code).translate(text)
    except Exception:
        return text 

def _t_num(num_str, target_lang):
    num_str = str(num_str)
    if target_lang == 'English': return num_str
    devnagari = str.maketrans('0123456789', '०१२३४५६७८९')
    gujarati = str.maketrans('0123456789', '૦૧૨૩૪૫૬૭૮૯')
    
    if target_lang in ['हिन्दी (Hindi)', 'मराठी (Marathi)']: return num_str.translate(devnagari)
    elif target_lang == 'ગુજરાતી (Gujarati)': return num_str.translate(gujarati)
    return num_str

# --- DATA DICTIONARIES ---
crop_emojis = {
    'rice': '🌾', 'maize': '🌽', 'chickpea': '🥜', 'kidneybeans': '🫘',
    'pigeonpeas': '🫛', 'mothbeans': '🌱', 'mungbean': '🌱', 'blackgram': '☕',
    'lentil': '🍲', 'pomegranate': '🍎', 'banana': '🍌', 'mango': '🥭',
    'grapes': '🍇', 'watermelon': '🍉', 'muskmelon': '🍈', 'apple': '🍏',
    'orange': '🍊', 'papaya': '🍈', 'coconut': '🥥', 'cotton': '☁️',
    'jute': '🌿', 'coffee': '☕'
}

# Kept solely as a fallback if the Government API is down or missing a key
fallback_mandi_prices = {
    'rice': '₹ 2,200 - ₹ 2,400', 'maize': '₹ 2,000 - ₹ 2,100', 'chickpea': '₹ 5,300 - ₹ 5,500',
    'kidneybeans': '₹ 6,000 - ₹ 6,500', 'pigeonpeas': '₹ 6,800 - ₹ 7,200', 'mothbeans': '₹ 5,000 - ₹ 5,200',
    'mungbean': '₹ 7,000 - ₹ 7,500', 'blackgram': '₹ 6,500 - ₹ 7,000', 'lentil': '₹ 5,800 - ₹ 6,000',
    'pomegranate': '₹ 8,000 - ₹ 10,000', 'banana': '₹ 1,500 - ₹ 2,000', 'mango': '₹ 4,000 - ₹ 8,000',
    'grapes': '₹ 4,500 - ₹ 6,000', 'watermelon': '₹ 1,000 - ₹ 1,500', 'muskmelon': '₹ 1,500 - ₹ 2,000',
    'apple': '₹ 6,000 - ₹ 12,000', 'orange': '₹ 3,000 - ₹ 5,000', 'papaya': '₹ 1,500 - ₹ 2,500',
    'coconut': '₹ 2,500 - ₹ 3,000', 'cotton': '₹ 7,000 - ₹ 7,500', 'jute': '₹ 4,500 - ₹ 5,000',
    'coffee': '₹ 25,000 - ₹ 30,000'
}

crop_wiki = {
    'rice': "Requires heavily waterlogged soil. Best sown in June-July.",
    'apple': "Thrives in cooler, hilly regions. Requires well-drained, loamy soil.",
    'chickpea': "A hardy, drought-tolerant crop. Excellent for nitrogen fixation in soil.",
    'cotton': "Requires a long frost-free period and plenty of sunshine.",
    'jute': "Requires hot and humid climates with abundant rainfall.",
    'maize': "Versatile crop. Highly responsive to nitrogen fertilizers."
}

# --- LOAD THE MODEL ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'crop_recommendation_model.pkl')

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Please run train.py first!")
    st.stop()

# --- REAL-TIME API FUNCTIONS ---
def fetch_weather_by_coords(lat, lon):
    try:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m"
        weather_data = requests.get(weather_url).json()
        return weather_data["current"]["temperature_2m"], weather_data["current"]["relative_humidity_2m"]
    except:
        return None

def fetch_7_day_forecast(lat, lon):
    try:
        forecast_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
        data = requests.get(forecast_url).json()
        return pd.DataFrame({
            "Date": data["daily"]["time"],
            "Max Temp": data["daily"]["temperature_2m_max"],
            "Min Temp": data["daily"]["temperature_2m_min"],
            "Rainfall": data["daily"]["precipitation_sum"]
        })
    except:
        return None

def fetch_recent_rainfall(lat, lon):
    """Fetches the sum of actual rainfall over the last 30 days using the live past_days API."""
    try:
        # Use the live forecast API with past_days=30 to avoid archive delay errors
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&past_days=30&forecast_days=1&timezone=auto"
        data = requests.get(url).json()
        
        # Sum all precipitation data from the past 30 days
        total_rain = sum(p for p in data['daily']['precipitation_sum'] if p is not None)
        return round(total_rain, 1)
    except:
        return 100.0 # Safe fallback

def fetch_live_mandi_price(crop_name):
    """Fetches live prices from the Indian Gov API with a dictionary fallback."""
    API_KEY = "579b464db66ec23bdd0000017676a56ae0a94c254cc568ba5d60d277" 
    
    try:
        url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={API_KEY}&format=json&filters[commodity]={crop_name.capitalize()}"
        response = requests.get(url).json()
        
        # Ensure we actually got records back from the government server
        if response.get('records') and len(response['records']) > 0:
            min_price = response['records'][0]['min_price']
            max_price = response['records'][0]['max_price']
            return f"₹ {min_price} - ₹ {max_price}"
            
        return fallback_mandi_prices.get(crop_name.lower(), 'Data Unavailable')
    except Exception:
        return fallback_mandi_prices.get(crop_name.lower(), 'Data Unavailable')

def check_weather_alerts(forecast_df, lang):
    try:
        alerts = []
        if any(temp > 40.0 for temp in forecast_df["Max Temp"] if temp is not None):
            alerts.append(_t("🔥 **HEATWAVE ALERT:** Temperatures exceeding 40°C expected in the next 7 days.", lang))
        if any(rain > 50.0 for rain in forecast_df["Rainfall"] if rain is not None):
            alerts.append(_t("🌧️ **FLOOD WARNING:** Heavy rainfall (>50mm) expected in a single day.", lang))
        return alerts
    except Exception:
        return []

# --- LANGUAGE SELECTION ---
lang_choice = st.sidebar.selectbox("🌐 Select Language / भाषा चुनें / भाषा निवडा / ભાષા પસંદ કરો", 
                                   list(api_lang_codes.keys()))

# --- APP HEADER ---
st.title(_t("🌾 AgriSmart Pro: AI Farming Assistant", lang_choice))
st.markdown(_t("Precision agriculture powered by Machine Learning, Live GPS Weather, and Expert Analytics.", lang_choice))
st.divider()

# --- SIDEBAR (INPUTS) ---
with st.sidebar:
    st.header(_t("🌍 Location & Weather", lang_choice))

    if 'temp' not in st.session_state: st.session_state['temp'] = 25.0
    if 'hum' not in st.session_state: st.session_state['hum'] = 60.0
    if 'rain' not in st.session_state: st.session_state['rain'] = 100.0
    if 'location_name' not in st.session_state: st.session_state['location_name'] = "Manual Entry"

    st.markdown(_t("**📍 Option 1: Use Live GPS**", lang_choice))
    st.info(_t("Click the small target icon below to lock your exact farm coordinates.", lang_choice), icon="🛰️")    
    location = streamlit_geolocation()
    
    if location and location.get('latitude') is not None:
        st.session_state['locked_lat'] = location['latitude']
        st.session_state['locked_lon'] = location['longitude']

    if 'locked_lat' in st.session_state:
        lat = st.session_state['locked_lat']
        lon = st.session_state['locked_lon']
        
        t_lat = _t_num(f"{lat:.2f}", lang_choice)
        t_lon = _t_num(f"{lon:.2f}", lang_choice)
        st.success(_t("GPS Locked:", lang_choice) + f" {t_lat}, {t_lon}")
        
        if st.button(_t("Fetch Weather for My GPS", lang_choice), type="primary", use_container_width=True):
            weather = fetch_weather_by_coords(lat, lon)
            if weather:
                st.session_state['temp'], st.session_state['hum'] = weather[0], weather[1]
                st.session_state['location_name'] = _t("GPS", lang_choice) + f" ({t_lat}, {t_lon})"
                
                # --- DYNAMIC RAINFALL TRIGGER ---
                st.session_state['rain'] = fetch_recent_rainfall(lat, lon)
                
                st.session_state['forecast_df'] = fetch_7_day_forecast(lat, lon)
                if st.session_state['forecast_df'] is not None:
                    st.session_state['alerts'] = check_weather_alerts(st.session_state['forecast_df'], lang_choice) 

    st.divider()
    
    st.markdown(_t("**🏙️ Option 2: Search by City**", lang_choice))
    city = st.text_input(_t("City Name", lang_choice), value="Pune")
    if st.button(_t("Fetch Weather by City ☁️", lang_choice), use_container_width=True):
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
            geo_data = requests.get(geo_url).json()
            if "results" in geo_data:
                c_lat, c_lon = geo_data["results"][0]["latitude"], geo_data["results"][0]["longitude"]
                weather = fetch_weather_by_coords(c_lat, c_lon)
                if weather:
                    st.session_state['temp'], st.session_state['hum'] = weather[0], weather[1]
                    st.session_state['location_name'] = _t(city, lang_choice)
                    
                    # --- DYNAMIC RAINFALL TRIGGER ---
                    st.session_state['rain'] = fetch_recent_rainfall(c_lat, c_lon)
                    
                    st.session_state['forecast_df'] = fetch_7_day_forecast(c_lat, c_lon)
                    if st.session_state['forecast_df'] is not None:
                        st.session_state['alerts'] = check_weather_alerts(st.session_state['forecast_df'], lang_choice)
        except:
            pass

    st.divider()
    
    st.subheader(_t("⛅ Climate Parameters", lang_choice))
    temperature = st.slider(_t("Temperature (°C)", lang_choice), 0.0, 50.0, float(st.session_state['temp']), 0.1)
    humidity = st.slider(_t("Humidity (%)", lang_choice), 0.0, 100.0, float(st.session_state['hum']), 0.1)
    rainfall = st.slider(_t("Seasonal Rainfall (mm)", lang_choice), 0.0, 300.0, float(st.session_state['rain']), 0.1)

    st.subheader(_t("🌱 Soil Nutrients", lang_choice))
    N = st.slider(_t("Nitrogen (N)", lang_choice), 0, 150, 50)
    P = st.slider(_t("Phosphorous (P)", lang_choice), 0, 150, 50)
    K = st.slider(_t("Potassium (K)", lang_choice), 0, 205, 50)
    ph = st.slider(_t("Soil pH", lang_choice), 0.0, 14.0, 6.5, 0.1)

# --- MAIN DASHBOARD AREA ---
if 'alerts' in st.session_state and st.session_state['alerts']:
    for alert in st.session_state['alerts']:
        st.error(alert, icon="🚨")

t_tab1 = _t("🎯 AI Recommendation", lang_choice)
t_tab2 = _t("📊 Data & Weather Profile", lang_choice)
t_tab3 = _t("🆘 Disaster Recovery", lang_choice)
tab1, tab2, tab3 = st.tabs([t_tab1, t_tab2, t_tab3])

with tab1:
    if st.button(_t("Generate Crop Recommendation 🚀", lang_choice), type="primary", use_container_width=True):
        user_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        probabilities = model.predict_proba(user_data)[0]
        
        top_3_indices = np.argsort(probabilities)[::-1][:3]
        top_3_crops = model.classes_[top_3_indices]
        top_3_probs = probabilities[top_3_indices]
        
        st.session_state['top_crop'] = top_3_crops[0]
        st.session_state['top_crops'] = top_3_crops
        st.session_state['top_probs'] = top_3_probs
        
        main_crop = top_3_crops[0]
        emoji = crop_emojis.get(main_crop.lower(), '🌱')
        t_main_crop = _t(main_crop.capitalize(), lang_choice)
        
        st.success("## 🏆 " + _t("Primary Recommendation:", lang_choice) + f" {t_main_crop} {emoji}")
        
        # --- DYNAMIC MARKET PRICE TRIGGER ---
        raw_market_rate = fetch_live_mandi_price(main_crop)
        t_market_rate = _t_num(raw_market_rate, lang_choice) if "₹" in raw_market_rate else _t(raw_market_rate, lang_choice)
        
        st.metric(label="📈 " + _t("Live Market Rate", lang_choice) + f" - {t_main_crop} " + _t("(per Quintal)", lang_choice), value=t_market_rate)
        
        raw_note = crop_wiki.get(main_crop.lower(), "Standard care required.")
        st.info("**📖 " + _t("Expert Note:", lang_choice) + "**\n\n" + _t(raw_note, lang_choice))
        
        st.markdown("---")
        for crop, prob in zip(top_3_crops, top_3_probs):
            prob_percent = round(prob * 100, 2)
            if prob_percent > 0:
                c_icon = crop_emojis.get(crop.lower(), '🌱')
                t_crop = _t(crop.capitalize(), lang_choice)
                t_prob = _t_num(prob_percent, lang_choice)
                st.write(f"**{t_crop} {c_icon}** - {t_prob}% " + _t("Match", lang_choice))
                st.progress(int(prob_percent))

with tab2:
    st.markdown(_t("### 🌍 Environmental Profile", lang_choice))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_t("Nitrogen", lang_choice), _t_num(N, lang_choice))
    c2.metric(_t("Phosphorus", lang_choice), _t_num(P, lang_choice))
    c3.metric(_t("Potassium", lang_choice), _t_num(K, lang_choice))
    c4.metric(_t("Soil pH", lang_choice), _t_num(ph, lang_choice))
    
    c5, c6, c7 = st.columns(3)
    c5.metric(_t("Current Temp", lang_choice), _t_num(temperature, lang_choice) + " °C")
    c6.metric(_t("Humidity", lang_choice), _t_num(humidity, lang_choice) + " %")
    c7.metric(_t("Seasonal Rain", lang_choice), _t_num(rainfall, lang_choice) + " mm")
    
    st.divider()
    
    st.markdown(_t("### 📅 7-Day Forecast for", lang_choice) + f" {st.session_state['location_name']}")
    
    if 'forecast_df' in st.session_state and st.session_state['forecast_df'] is not None:
        days = st.columns(7)
        for i, col in enumerate(days):
            day_data = st.session_state['forecast_df'].iloc[i]
            date_obj = datetime.strptime(day_data['Date'], "%Y-%m-%d")
            day_str = _t(date_obj.strftime("%a, %b %d"), lang_choice)
            
            weather_icon = "☀️"
            if day_data['Rainfall'] > 10: weather_icon = "🌧️"
            elif day_data['Rainfall'] > 0: weather_icon = "🌦️"
            elif day_data['Max Temp'] > 35: weather_icon = "🔥"
            
            t_max = _t_num(day_data['Max Temp'], lang_choice)
            t_min = _t_num(day_data['Min Temp'], lang_choice)
            t_rain = _t_num(day_data['Rainfall'], lang_choice)
            
            with col:
                st.markdown(f"**{day_str}**")
                st.markdown(f"<h2 style='text-align: center; margin: 0;'>{weather_icon}</h2>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center;'>🔺 {t_max}°C<br>🔻 {t_min}°C<br>💧 {t_rain}mm</div>", unsafe_allow_html=True)
    else:
        st.info(_t("👈 Please use the sidebar to fetch weather data for your location to see the forecast.", lang_choice))
    
    st.divider()
    st.markdown(_t("### 📄 Enterprise Report Generation", lang_choice))
    
    if 'top_crop' in st.session_state and 'top_crops' in st.session_state and 'top_probs' in st.session_state:
        t_report_crop = _t(st.session_state['top_crops'][0].capitalize(), lang_choice)
        t_report_prob = round(st.session_state['top_probs'][0]*100, 2)
        
        report_text = f"""=========================================
AGRISMART PRO - OFFICIAL FARM REPORT
=========================================
Date Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Location Context: {st.session_state['location_name']}

1. ENVIRONMENTAL DATA:
----------------------
- Nitrogen (N): {N}
- Phosphorous (P): {P}
- Potassium (K): {K}
- Soil pH: {ph}
- Temperature: {temperature} °C
- Humidity: {humidity} %
- Seasonal Rainfall: {rainfall} mm

2. AI RECOMMENDATIONS:
----------------------
Primary Crop: {t_report_crop} ({t_report_prob}% Confidence)
========================================="""
        st.download_button(label="💾 " + _t("Download Official Farm Report (.txt)", lang_choice), data=report_text, file_name="Farm_Report.txt", mime="text/plain", type="primary")
    else:
        st.warning(_t("⚠️ Run the AI Analysis first to unlock report generation.", lang_choice))

with tab3:
    st.markdown(_t("### 🆘 Disaster Mitigation & Crop Salvage", lang_choice))
    
    disaster_options = ["Flood / Waterlogging", "Severe Drought", "Pest Swarm", "Frost / Extreme Cold"]
    translated_disasters = [_t(d, lang_choice) for d in disaster_options]
    
    selected_translated_disaster = st.selectbox(_t("Select Disaster Event:", lang_choice), translated_disasters)
    disaster_type = disaster_options[translated_disasters.index(selected_translated_disaster)]
    
    crop_list = list(crop_emojis.keys())
    default_index = 0
    if 'top_crop' in st.session_state and st.session_state['top_crop'] in crop_list:
        default_index = crop_list.index(st.session_state['top_crop'])
            
    affected_crop_eng = st.selectbox(_t("Select Affected Crop:", lang_choice), crop_list, index=default_index, format_func=lambda x: _t(x.capitalize(), lang_choice))
    t_affected_crop = _t(affected_crop_eng.capitalize(), lang_choice)
    
    if st.button(_t("Get Mitigation Protocol 🚑", lang_choice), type="primary"):
        st.warning("**" + _t("Emergency Protocol for", lang_choice) + f" {t_affected_crop} " + _t("during", lang_choice) + f" {_t(disaster_type, lang_choice)}:**")
        if disaster_type == "Flood / Waterlogging":
            st.write(_t("1. **Drainage:** Immediately dig lateral trenches to drain standing water within 48 hours.", lang_choice))
            st.write(_t("2. **Disease Control:** Apply a broad-spectrum copper-based fungicide to prevent root rot.", lang_choice))
        elif disaster_type == "Severe Drought":
            st.write(_t("1. **Micro-Irrigation:** Deploy drip irrigation focusing strictly on the root zone.", lang_choice))
            st.write(_t("2. **Mulching:** Apply organic mulch thickly to retain remaining soil moisture.", lang_choice))
        elif disaster_type == "Pest Swarm":
            st.write(_t("1. **Identification:** Isolate the affected area. Do not move equipment.", lang_choice))
            st.write(_t("2. **Treatment:** Deploy appropriate localized bio-pesticides immediately.", lang_choice))
        elif disaster_type == "Frost / Extreme Cold":
            st.write(_t("1. **Irrigation:** Lightly irrigate the field; wet soil retains more heat than dry soil.", lang_choice))
            st.write(_t("2. **Covering:** Deploy frost blankets or plastic high-tunnels.", lang_choice))

    st.divider()
    st.markdown(_t("### 💸 Financial Yield Loss & Insurance Estimator", lang_choice))
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        farm_size = st.number_input(_t("Farm Size (Hectares)", lang_choice), min_value=0.1, value=2.0, step=0.1)
        investment = st.number_input(_t("Total Investment per Hectare (₹)", lang_choice), min_value=1000, value=50000, step=1000)
    with col_f2:
        damage_percent = st.slider(_t("Estimated Crop Damage (%)", lang_choice), 0, 100, 50)
        market_price = st.number_input(_t("Expected Market Revenue per Hectare (₹)", lang_choice), min_value=1000, value=80000, step=1000)

    if st.button(_t("Calculate Financial Impact 📉", lang_choice)):
        total_investment = farm_size * investment
        expected_revenue = farm_size * market_price
        financial_loss = expected_revenue * (damage_percent / 100.0)
        net_impact = expected_revenue - financial_loss - total_investment

        t_loss = _t_num(f"{financial_loss:,.2f}", lang_choice)
        t_inv = _t_num(f"{total_investment:,.2f}", lang_choice)
        t_rev = _t_num(f"{expected_revenue:,.2f}", lang_choice)
        t_net = _t_num(f"{net_impact:,.2f}", lang_choice)

        st.error("#### " + _t("Estimated Financial Loss:", lang_choice) + f" ₹ {t_loss}")

        c_inv, c_rev, c_net = st.columns(3)
        c_inv.metric(_t("Total Investment", lang_choice), f"₹ {t_inv}")
        c_rev.metric(_t("Expected Revenue", lang_choice), f"₹ {t_rev}")
        c_net.metric(_t("Net After Damage", lang_choice), f"₹ {t_net}", delta=f"-₹ {t_loss}", delta_color="inverse")

        st.info("**" + _t("Insurance Next Steps:", lang_choice) + "** " + _t("Export this data along with timestamped photos of the field to your local agriculture officer or crop insurance portal within 72 hours of the disaster event.", lang_choice))