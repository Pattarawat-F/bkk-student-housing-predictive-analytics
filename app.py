import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise Predictive & Market Analytics Platform", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Gemini Custom UI Styling (CSS Injection)
st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #1F1F1F;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F4F9 !important;
        border-right: 1px solid #E3E3E3;
    }
    .gemini-gradient-text {
        background: linear-gradient(45deg, #1A73E8, #7A22FF, #9B51E0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 36px;
        margin-bottom: 5px;
    }
    .gemini-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stButton>button {
        background: linear-gradient(90deg, #1A73E8, #4285F4) !important;
        color: white !important;
        border-radius: 24px !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 5px rgba(26,115,232,0.2) !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 10px rgba(26,115,232,0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Header Block
st.markdown('<h1 class="gemini-gradient-text">Predictive & Market Analytics Platform</h1>', unsafe_allow_html=True)
st.markdown("<p style='color:#5F6368; font-size:16px; margin-top:0px; margin-bottom:30px;'>Advanced Machine Learning Inference and Interactive Visual Dashboard for Bangkok Student Housing Dataset.</p>", unsafe_allow_html=True)

# 4. Data Pipeline & Model Training
@st.cache_resource
def load_and_analyze_data():
    df_raw = pd.read_csv("Data accomandation project 4 - Data.csv")
    df_raw['price'] = df_raw['ราคาเช่า / เดือน'].astype(str).str.replace(',', '').astype(float)
    df_raw['is_inner_zone'] = (df_raw['เขตที่ตั้ง'] == 1).astype(int)
    df_raw['is_middle_zone'] = (df_raw['เขตที่ตั้ง'] == 2).astype(int)

    df_raw['Zone_Label'] = df_raw['เขตที่ตั้ง'].map({
        1: 'Inner Zone (Zone 1)', 
        2: 'Middle Zone (Zone 2)', 
        3: 'Outer Zone (Zone 3)'
    })

    # Assign Mock Coordinates for Geospatial Mapping
    np.random.seed(42)
    lat_center, lon_center = 13.7563, 100.5018
    df_raw['lat'] = np.where(df_raw['เขตที่ตั้ง'] == 1, lat_center + np.random.normal(0, 0.03, len(df_raw)),
                     np.where(df_raw['เขตที่ตั้ง'] == 2, lat_center + np.random.normal(0.04, 0.04, len(df_raw)), 
                              lat_center + np.random.normal(-0.06, 0.05, len(df_raw))))
    df_raw['lon'] = np.where(df_raw['เขตที่ตั้ง'] == 1, lon_center + np.random.normal(0, 0.03, len(df_raw)),
                     np.where(df_raw['เขตที่ตั้ง'] == 2, lon_center + np.random.normal(0.05, 0.04, len(df_raw)), 
                              lon_center + np.random.normal(-0.07, 0.05, len(df_raw))))

    X = pd.DataFrame({
        'room_size': df_raw['ขนาดห้อง /ตร.ม'],
        'dist_university': df_raw['ระยะทางถึง มหาลัย / กม.'],
        'dist_transport': df_raw['ระยะทางถึงขนส่ง / กม.'],
        'dist_restaurant': df_raw['ระยะทางถึงร้านอาหาร / กม.'],
        'has_parking': df_raw['ที่จอดรถ (1/0)'],
        'water_bill': df_raw['ค่าน้ำ / บาท'],
        'electricity_bill': df_raw['ค่าไฟ'],
        'univ_type': df_raw['มหาลัย (1/0)'],
        'is_inner_zone': df_raw['is_inner_zone'],
        'is_middle_zone': df_raw['is_middle_zone']
    }).fillna(25.0)

    y = df_raw['price']

    # Fit model using raw array values to prevent feature name mismatches
    X_array = X.values
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_array, y)

    feature_names = list(X.columns)
    return rf, feature_names, df_raw

model, model_features, df_clean = load_and_analyze_data()

# 5. Sidebar Layout for Parameters Input
st.sidebar.markdown("<h3 style='color:#1A73E8; margin-top:10px;'>Model Controls</h3>", unsafe_allow_html=True)

with st.sidebar.expander(" Dimension Parameters", expanded=True):
    room_size = st.sidebar.slider("Room Size (sq.m.)", 10.0, 60.0, 25.0)
    has_parking = st.sidebar.selectbox("Parking Availability", ["Available", "Not Available"])
    has_parking_val = 1 if has_parking == "Available" else 0

with st.sidebar.expander("📍 Location & Proximity", expanded=True):
    zone = st.sidebar.selectbox("Economic Zoning", ["Inner Economic Zone (Zone 1)", "Middle Economic Zone (Zone 2)", "Outer Economic Zone (Zone 3)"])
    is_inner = 1 if zone == "Inner Economic Zone (Zone 1)" else 0
    is_middle = 1 if zone == "Middle Economic Zone (Zone 2)" else 0

    dist_university = st.sidebar.slider("Distance to University (km)", 0.1, 5.0, 1.5)
    dist_transport = st.sidebar.slider("Distance to Public Transport (km)", 0.1, 12.0, 1.0)
    dist_restaurant = st.sidebar.slider("Distance to Commercial Hubs (km)", 0.1, 5.0, 1.0)
    univ_type = st.sidebar.selectbox("University Proximity Type", ["Public University", "Private University"])
    univ_type_val = 1 if univ_type == "Public University" else 0

with st.sidebar.expander(" Utility Structures", expanded=False):
    water_bill = st.sidebar.number_input("Water Rate (THB/Unit)", value=18)
    electricity_bill = st.sidebar.number_input("Electricity Rate (THB/Unit)", value=7)

# 6. Main Workspace Layout Split
main_col, side_col = st.columns([1.2, 1])

# --- LEFT COLUMN: ML Predict Engine ---
with main_col:
    st.markdown("<h4 style='color:#1F1F1F; margin-bottom:15px;'> Live Predictor Engine</h4>", unsafe_allow_html=True)

    input_vector = np.array([[
        room_size, dist_university, dist_transport, dist_restaurant,
        has_parking_val, water_bill, electricity_bill, univ_type_val,
        is_inner, is_middle
    ]])

    if st.button(" Run Predictive Valuation Engine", use_container_width=True):
        prediction = model.predict(input_vector)[0]

        st.markdown(f"""
            <div style="background-color: #F0F4F9; border-radius: 24px; padding: 25px; margin-top:10px; margin-bottom:20px; border-left: 6px solid #7A22FF;">
                <p style="margin:0; font-size:13px; color:#5F6368; font-weight:600; text-transform:uppercase;">ML Recommended Price Output</p>
                <h1 style="margin:5px 0 0 0; color:#1F1F1F; font-size:38px;">{prediction:,.2f} THB <span style='font-size:18px; color:#5F6368;'>/ Month</span></h1>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("Operational Strategies")
        if is_inner == 1:
            st.info("**Premium Pricing:** Asset sits inside the high-yield Inner Economic Zone. Yield optimization is viable.")
        if dist_university > 3.0:
            st.warning("**Proximity Decay Mitigation:** Prolonged distance to university detected. Deploy a complimentary Shuttle Bus system.")
        if has_parking_val == 0:
            st.write(" **Unbundling Strategy:** No parking mapped. Unbundle rent pricing from asset infrastructure to target vehicle-free demographics.")
    else:
        st.info("💡 Click the button above to execute the Machine Learning Valuation Engine based on your selected sidebar inputs.")

    # Feature Importance Plot
    st.markdown("---")
    st.subheader(" AI Feature Importance")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': model_features, 'Importance': importances}).sort_values(by='Importance', ascending=True)
    fig_importance = px.bar(feat_df, x='Importance', y='Feature', orientation='h', 
                             title='Which features drive the Machine Learning decisions?',
                             color_discrete_sequence=['#7A22FF'])
    fig_importance.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=330, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_importance, use_container_width=True)

# --- RIGHT COLUMN: Maps, Tables, and Charts ---
with side_col:
    st.markdown("<h4 style='color:#1F1F1F; margin-bottom:15px;'>📍 Geospatial & Vector Dashboard</h4>", unsafe_allow_html=True)

    # 1. Geographic Distribution Map
    fig_map = px.scatter_mapbox(df_clean, lat="lat", lon="lon", color="Zone_Label", size="price",
                                color_discrete_map={'Inner Zone (Zone 1)': '#1A73E8', 'Middle Zone (Zone 2)': '#7A22FF', 'Outer Zone (Zone 3)': '#9B51E0'},
                                hover_name="ราคาเช่า / เดือน", hover_data=["ขนาดห้อง /ตร.ม"],
                                zoom=10, height=280)
    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    st.plotly_chart(fig_map, use_container_width=True)

    # 2. Vector State Parameters Table
    st.markdown("<h5 style='color:#1F1F1F; margin-top:10px; margin-bottom:5px;'>📋 Current Selected Parameter State</h5>", unsafe_allow_html=True)
    param_df = pd.DataFrame({
        "Feature Attribute": ["Room Area Size", "Economic Zoning", "University Proximity", "Vehicle Parking"],
        "State Value": [f"{room_size} sq.m.", zone.split(" ")[0], f"{dist_university} km", has_parking]
    })
    st.dataframe(param_df, hide_index=True, use_container_width=True)

    # 3. Market Descriptive Visual (Structural Variance Across Zones)
    fig_box = px.box(df_clean, x="Zone_Label", y="price", color="Zone_Label",
                     title="Rental Price Structural Variance Across Zones",
                     color_discrete_map={'Inner Zone (Zone 1)': '#1A73E8', 'Middle Zone (Zone 2)': '#7A22FF', 'Outer Zone (Zone 3)': '#9B51E0'})
    fig_box.update_layout(plot_bgcolor='#F8FAFC', paper_bgcolor='white', height=240, showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_box, use_container_width=True)
