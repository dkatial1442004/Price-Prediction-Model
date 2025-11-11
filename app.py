import pandas as pd
import streamlit as st
import pickle

# Load the pre-trained models and data
best_model = pickle.load(open('best_house_price_model.pkl', 'rb'))
dataset = pickle.load(open('dataset.pkl', 'rb'))
model_scores = pickle.load(open('model_scores.pkl', 'rb'))

# Extract unique locations for the dropdown
locations = dataset['location'].unique()

st.title("Bangalore House Price Prediction")

# Inject custom CSS to increase font size
st.markdown(
    """
    <style>
    .medium-font {
        font-size: 20px;
    }
    .big-font {
        font-size: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields for user with custom font size
st.markdown('<p class="medium-font">Location</p>', unsafe_allow_html=True)
location = st.selectbox('', locations)

st.markdown('<p class="medium-font">Total Square Foot</p>', unsafe_allow_html=True)
total_sqft = st.number_input('', min_value=0.0, value=1000.0)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<p class="medium-font">Number of Bathrooms</p>', unsafe_allow_html=True)
    bath = st.number_input('', min_value=1, max_value=10, value=2)
with col2:
    st.markdown('<p class="medium-font">Number of Bedrooms (BHK)</p>', unsafe_allow_html=True)
    bhk = st.number_input('', min_value=1, max_value=10, value=2)

if st.button('Predict'):
    input_data = pd.DataFrame([[location, total_sqft, bath, bhk]], columns=['location', 'total_sqft', 'bath', 'BHK'])
    best_pred = best_model.predict(input_data)[0]

    # Display the prediction from the best model
    st.markdown(f'<p class="big-font">The predicted price using the best model ({best_model_name}) is {best_pred:.2f} lakhs.</p>', unsafe_allow_html=True)

    # Display comparison with other models
    st.markdown('<p class="medium-font">### Comparison with Other Models:</p>', unsafe_allow_html=True)
    for name, (model, score) in model_scores.items():
        other_pred = model.predict(input_data)[0]
        difference = other_pred - best_pred
        st.markdown(f'<p class="medium-font"><b>{name}:</b> Predicted Price = {other_pred:.2f} lakhs (Difference: {difference:+.2f} lakhs, R² Score: {score:.4f})</p>', unsafe_allow_html=True)
