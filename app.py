import streamlit as st
import pandas as pd
import pycaret.classification as pc

# Load the model
model = pc.load_model('wheat-pipeline')

st.set_page_config(page_title="Wheat Classification", layout="wide")

# Center the title
st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5em; font-weight: bold;'>🌾 Wheat Classification Predictor</h1>
    """,
    unsafe_allow_html=True
)

# Center the banner.gif using Streamlit's built-in layout system
col1, col2, col3 = st.columns([1, 3, 1])  

with col2:
    st.image("misc/banner.gif", use_container_width=True)  

st.divider()

# Center the prediction title
st.markdown(
    """
    <h2 style='text-align: center;'>Wheat Classification Prediction</h2>
    <p style='text-align: center;'>Enter the feature values to predict the wheat type:</p>
    """,
    unsafe_allow_html=True
)

# Center the text boxes and make them shorter
with st.container():
    col1, col2, col3 = st.columns([2, 3, 2])  
    
    with col2:
        area = st.number_input("Area", min_value=0.0, format="%.2f")
        perimeter = st.number_input("Perimeter", min_value=0.0, format="%.2f")
        compactness = st.number_input("Compactness", min_value=0.0, format="%.2f")
        length = st.number_input("Length", min_value=0.0, format="%.2f")
        width = st.number_input("Width", min_value=0.0, format="%.2f")
        asymmetry_coeff = st.number_input("Asymmetry Coefficient", min_value=0.0, format="%.2f")
        groove = st.number_input("Groove", min_value=0.0, format="%.2f")

# Add padding before the button
st.markdown("<br>", unsafe_allow_html=True)

# Create columns to center the button
col1, col2, col3 = st.columns([2, 1, 2])  

with col2:
    if st.button("Predict", use_container_width=True):  
        # Prepare input data as DataFrame
        input_data = pd.DataFrame(
            [[area, perimeter, compactness, length, width, asymmetry_coeff, groove]],
            columns=["Area", "Perimeter", "Compactness", "Length", "Width", "AsymmetryCoeff", "Groove"]
        )
        
        # Make predictions
        prediction = model.predict(input_data)
        
        # Display prediction
        st.success(f"Predicted Wheat Type: {prediction[0]}")

# Center the footer
st.markdown(
    """
    <div style="text-align: center;">
        <hr>
        <p>🛠️ Developed by <b>[Your Name]</b></p>
        <p>📅 Project: AI Wheat Classification</p>
    </div>
    """,
    unsafe_allow_html=True
)
