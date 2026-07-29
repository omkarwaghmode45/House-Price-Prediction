import streamlit as st
import joblib  
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #edf4fb 100%);
            color: #10233f;
        }
        .main-header {
            font-size: 2.9rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
            color: #0b1f3a;
            letter-spacing: -0.03em;
        }
        .subheader {
            font-size: 1.08rem;
            color: #31475f;
            margin-bottom: 1.25rem;
        }
        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0b1f3a;
            margin-bottom: 0.35rem;
        }
        .section-subtitle {
            font-size: 0.96rem;
            color: #52657c;
            margin-bottom: 0.75rem;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(16, 35, 63, 0.08);
            border-radius: 18px;
            padding: 1.25rem 1.25rem 0.5rem 1.25rem;
            box-shadow: 0 8px 30px rgba(16, 35, 63, 0.06);
        }
        .metric-card {
            background: linear-gradient(135deg, #0b1f3a 0%, #1f5da8 100%);
            color: white;
            border-radius: 18px;
            padding: 1.35rem 1.4rem;
            box-shadow: 0 16px 35px rgba(16, 35, 63, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }
        .metric-label {
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.82;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 2.35rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.1;
        }
        .metric-caption {
            font-size: 0.94rem;
            opacity: 0.85;
            margin-top: 0.25rem;
        }
        .prediction-shell {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 20px;
            padding: 1rem;
            border: 1px solid rgba(16, 35, 63, 0.08);
            box-shadow: 0 10px 30px rgba(16, 35, 63, 0.06);
        }
        .stButton button {
            width: 100%;
            border-radius: 999px;
            padding: 0.9rem 1rem;
            font-weight: 700;
            border: none;
            background: linear-gradient(135deg, #10233f 0%, #1f5da8 100%);
            color: white;
            box-shadow: 0 10px 24px rgba(31, 93, 168, 0.22);
        }
        .stButton button:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }
        .stButton button:focus {
            outline: none;
        }
        .input-note {
            color: #5c6c80;
            font-size: 0.92rem;
            margin-top: -0.25rem;
        }
        div[data-testid="stNumberInput"] label,
        div[data-testid="stSelectbox"] label {
            font-size: 1.02rem !important;
            font-weight: 800 !important;
            color: #0b1f3a !important;
            letter-spacing: 0.01em;
            margin-bottom: 0.35rem;
        }
        div[data-testid="stNumberInput"] label p,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stNumberInput"] label span,
        div[data-testid="stSelectbox"] label span {
            color: inherit !important;
            font-weight: inherit !important;
        }
        div[data-testid="stNumberInput"] [data-baseweb="input"],
        div[data-testid="stSelectbox"] [data-baseweb="select"] {
            border-radius: 12px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid rgba(31, 93, 168, 0.18);
            border-radius: 14px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.98);
        }
        div[data-testid="stExpander"] details {
            border: none;
            background: transparent;
        }
        div[data-testid="stExpander"] summary {
            background: linear-gradient(135deg, #0b1f3a 0%, #1f5da8 100%);
            color: #ffffff !important;
            padding: 0.8rem 1rem;
            font-weight: 800;
            letter-spacing: 0.01em;
        }
        div[data-testid="stExpander"] summary:hover {
            background: linear-gradient(135deg, #10284a 0%, #2569bc 100%);
            color: #ffffff !important;
        }
        div[data-testid="stExpander"] summary p,
        div[data-testid="stExpander"] summary span,
        div[data-testid="stExpander"] summary svg {
            color: inherit !important;
            fill: currentColor !important;
        }
        div[data-testid="stExpander"] details[open] summary {
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }
        .footer-wrap {
            margin-top: 2rem;
            padding: 1rem 0 0.25rem 0;
            color: #4b5d72;
            text-align: center;
            font-size: 0.95rem;
            border-top: 1px solid rgba(16, 35, 63, 0.12);
        }
        .footer-strong {
            font-weight: 700;
            color: #0b1f3a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🏡 California House Price Predictor")
    st.caption("Portfolio-ready machine learning dashboard")
    st.divider()
    st.markdown("**Project Name**\n\nCalifornia House Price Prediction")
    st.markdown("**Model Name**\n\nRandom Forest Regressor")
    st.markdown("**Dataset**\n\nCalifornia Housing Dataset")
    st.markdown("**R² Score**\n\n0.817")
    st.markdown(
        "**Project Summary**\n\n"
        "A clean California housing price prediction app that estimates home value from geographic, structural, "
        "and economic inputs using a trained Random Forest model."
    )
    st.divider()
    st.markdown("**Technologies Used**")
    st.markdown("- Python\n- Streamlit\n- Scikit-learn\n- Pandas\n- NumPy")
    st.divider()
    st.info("Use the input form in the main panel to generate a price estimate.")

# Load the trained model
model = joblib.load("models/random_forest.pkl")

st.markdown('<div class="main-header">California House Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subheader">Enter property details below to estimate a home value with a trained Random Forest Regressor.</div>',
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Property Location</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Enter the geographic position of the house.</div>', unsafe_allow_html=True)
    location_left, location_right = st.columns(2)
    with location_left:
        longitude = st.number_input(
            "Longitude",
            min_value=-124.35,
            max_value=-114.30,
            value=-122.23,
            step=0.01,
            format="%.2f",
            help="Longitude in decimal degrees. Coastal California values are typically more negative.",
        )
    with location_right:
        latitude = st.number_input(
            "Latitude",
            min_value=32.50,
            max_value=42.05,
            value=37.88,
            step=0.01,
            format="%.2f",
            help="Latitude in decimal degrees. Northern California locations are higher.",
        )

    st.divider()
    st.markdown('<div class="section-title">Property Characteristics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Core housing attributes used by the model.</div>', unsafe_allow_html=True)
    characteristics_left, characteristics_right = st.columns(2)
    with characteristics_left:
        housing_median_age = st.number_input(
            "Housing Median Age",
            min_value=1,
            max_value=52,
            value=41,
            step=1,
            help="Median age of houses in the block group, measured in years.",
        )
        total_rooms = st.number_input(
            "Total Rooms",
            min_value=1,
            max_value=40000,
            value=880,
            step=10,
            help="Total number of rooms in the block group.",
        )
        total_bedrooms = st.number_input(
            "Total Bedrooms",
            min_value=1,
            max_value=8000,
            value=129,
            step=5,
            help="Total number of bedrooms in the block group.",
        )
    with characteristics_right:
        population = st.number_input(
            "Population",
            min_value=1,
            max_value=40000,
            value=322,
            step=10,
            help="Population count in the block group.",
        )
        households = st.number_input(
            "Households",
            min_value=1,
            max_value=7000,
            value=126,
            step=5,
            help="Number of households in the block group.",
        )
        median_income = st.number_input(
            "Median Income",
            min_value=0.50,
            max_value=15.00,
            value=8.3252,
            step=0.1,
            format="%.4f",
            help="Median household income in tens of thousands of dollars.",
        )

    st.divider()
    st.markdown('<div class="section-title">Neighborhood Context</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Select the closest coastal category.</div>', unsafe_allow_html=True)
    ocean_proximity = st.selectbox(
        "Ocean Proximity",
        [
            "<1H OCEAN",
            "INLAND",
            "ISLAND",
            "NEAR BAY",
            "NEAR OCEAN",
        ],
        help="Choose the nearest coastal category for the property location.",
    )
    st.markdown('<div class="input-note">This selection is automatically converted into the model’s one-hot encoded inputs.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">This panel highlights the estimated price after prediction.</div>', unsafe_allow_html=True)
    with st.expander("About this Model", expanded=True):
        st.markdown(
            "**Random Forest Regressor**\n\n"
            "This model combines predictions from multiple decision trees to produce a stable house price estimate.\n\n"
            "**California Housing Dataset**\n\n"
            "The dataset contains housing-related features from California block groups, including location, income, and neighborhood context.\n\n"
            "**What the prediction means**\n\n"
            "The displayed value is the model’s estimated median house value for the selected inputs. It is a data-driven estimate, not a guaranteed market appraisal."
        )
    st.divider()

    ocean_1h = 0
    ocean_inland = 0
    ocean_island = 0
    ocean_near_bay = 0
    ocean_near_ocean = 0

    if ocean_proximity == "<1H OCEAN":
        ocean_1h = 1

    elif ocean_proximity == "INLAND":
        ocean_inland = 1

    elif ocean_proximity == "ISLAND":
        ocean_island = 1

    elif ocean_proximity == "NEAR BAY":
        ocean_near_bay = 1

    elif ocean_proximity == "NEAR OCEAN":
        ocean_near_ocean = 1

    st.markdown("#### Ready to Predict")
    st.caption("All inputs are validated with realistic ranges for a polished user experience.")

    button_col_left, button_col_center, button_col_right = st.columns([0.12, 0.76, 0.12])
    with button_col_center:
        predict_pressed = st.button("🔮 Predict House Price")

    if predict_pressed:
        input_data = pd.DataFrame({
            "longitude": [longitude],
            "latitude": [latitude],
            "housing_median_age": [housing_median_age],
            "total_rooms": [total_rooms],
            "total_bedrooms": [total_bedrooms],
            "population": [population],
            "households": [households],
            "median_income": [median_income],
            "ocean_proximity_<1H OCEAN": [ocean_1h],
            "ocean_proximity_INLAND": [ocean_inland],
            "ocean_proximity_ISLAND": [ocean_island],
            "ocean_proximity_NEAR BAY": [ocean_near_bay],
            "ocean_proximity_NEAR OCEAN": [ocean_near_ocean]
        })

        prediction = model.predict(input_data)

        st.markdown('<div class="prediction-shell">', unsafe_allow_html=True)
        st.success("Prediction complete")
        st.markdown(
            f'''
            <div class="metric-card">
                <div class="metric-label">🏠 Estimated House Price</div>
                <div class="metric-value">💰 ${prediction[0]:,.2f}</div>
                <div class="metric-caption">Based on the selected California housing features</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption("The value above is the model’s estimated price for this input combination.")
    else:
        st.info("Submit the form to see a price estimate displayed here.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="footer-wrap">
        <div class="footer-strong">Developed by Omkar Waghmode</div>
        <div>Machine Learning Project</div>
        <div>Built with Streamlit &amp; Scikit-learn</div>
    </div>
    """,
    unsafe_allow_html=True,
)